import os
import tempfile

import mmcv 
import numpy as np
import torch
from mmcv.runner import load_state_dict, _load_checkpoint
from mmcv.runner import load_checkpoint
from mmpose.core import imshow_keypoints, imshow_multiview_keypoints_3d
from mmpose.core.camera import SimpleCamera
from mmpose.models import builder
from mmpose.models.builder import POSENETS 
from mmpose.models.detectors.base import BasePose

# from mmpose.models.detectors import DetectAndRegress
from multiview_pose.models.gcn_modules import GCNS


@POSENETS.register_module()
class GraphBasedModel(BasePose):
    def __init__(self, 
                 num_joints, 
                 backbone,
                 human_detector,
                 pose_regressor,
                 keypoint_head,
                 pose_refiner, 
                 freeze_2d=True,
                 freeze_keypoint_head=True,
                 test_with_refine=True, 
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(GraphBasedModel, self).__init__()
        if backbone is not None:
            self.backbone = builder.build_backbone(backbone)
        else:
            self.backbone = None

        if keypoint_head is not None:
            self.keypoint_head = builder.build_head(keypoint_head)
        else:
            self.keypoint_head = None

        if self.training and pretrained is not None:
            load_checkpoint(self, pretrained, strict=True)

        if pose_refiner is not None:
            self.pose_refiner = GCNS.build(pose_refiner)
        else:
            self.pose_refiner = None

        self.num_joints = num_joints
        self.freeze_2d = freeze_2d
        self.freeze_keypoint_head = freeze_keypoint_head
        self.test_with_refine = test_with_refine

        self.human_detector = builder.MODELS.build(human_detector)
        self.pose_regressor = builder.MODELS.build(pose_regressor)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
    
    @staticmethod
    def _freeze(model):
        model.eval()
        for param in model.parameters():
            param.requires_grad = False

    def train(self, mode=True):
        """Sets the module in training mode.
        Args:
            mode (bool): whether to set training mode (``True``)
                or evaluation mode (``False``). Default: ``True``.

        Returns:
            Module: self
        """
        super().train(mode)
        if mode and self.freeze_2d:
            if self.backbone is not None:
                self._freeze(self.backbone)
            if self.keypoint_head is not None and self.freeze_keypoint_head:
                self._freeze(self.keypoint_head)
            # for n, p in self.backbone.named_parameters():
            #     print(f'{n}: ', p.requires_grad)
            # for n, p in self.keypoint_head.named_parameters():
            #     print(f'{n}: ', p.requires_grad)

        return self

    @property
    def has_keypoint_2d_loss(self):
        return (not self.freeze_2d) or (self.freeze_2d and not self.freeze_keypoint_head)

    def forward(self,
                img=None,
                img_metas=None,
                return_loss=True,
                target=None,
                mask=None,
                targets_3d=None,
                input_heatmaps=None,
                **kwargs):

        if return_loss:
            return self.forward_train(img, img_metas, target, mask,
                                      targets_3d, input_heatmaps, **kwargs)
        else:
            return self.forward_test(img, img_metas, input_heatmaps, **kwargs)
    
    def train_step(self, data_batch, optimizer, **kwargs):
        losses = self.forward(**data_batch)

        loss, log_vars = self._parse_losses(losses)
        if 'img' in data_batch:
            batch_size = data_batch['img'][0].shape[0]
        else:
            assert 'input_heatmaps' in data_batch
            batch_size = data_batch['input_heatmaps'][0][0].shape[0]

        outputs = dict(loss=loss, log_vars=log_vars, num_samples=batch_size)

        return outputs
    
    def predict_heatmap(self, img):
        output = self.backbone(img)
        output = self.keypoint_head(output)
        return output

    def forward_train(self,
                      img,
                      img_metas,
                      target=None,
                      mask=None,
                      targets_3d=None,
                      input_heatmaps=None,
                      **kwargs):
        if self.backbone is None:
            assert input_heatmaps is not None
            feature_maps = []
            for input_heatmap in input_heatmaps:
                feature_maps.append(input_heatmap[0])
        else:
            feature_maps = []
            assert isinstance(img, list)
            for img_ in img:
                feature_maps.append(self.predict_heatmap(img_)[0])

        losses = dict()
        human_candidates, human_loss = self.human_detector.forward_train(
            None, img_metas, feature_maps, targets_3d, return_preds=True, **kwargs)
        losses.update(human_loss)

        pose_pred, pose_loss = self.pose_regressor.forward_train(
            None,
            img_metas,
            feature_maps=[f[:, -self.num_joints:].detach() for f in feature_maps],
            human_candidates=human_candidates,
            return_preds=True)
        losses.update(pose_loss)
        if self.pose_refiner is not None:
            losses.update(self.pose_refiner.forward_train(pose_pred, feature_maps, img_metas))

        if self.has_keypoint_2d_loss:
            losses_2d = {}
            heatmaps_tensor = torch.cat([f[:, -self.num_joints:] for f in feature_maps], dim=0)
            targets_tensor = torch.cat([t[0] for t in target], dim=0)
            masks_tensor = torch.cat([m[0] for m in mask], dim=0)
            losses_2d_ = self.keypoint_head.get_loss([heatmaps_tensor],
                                                     [targets_tensor], [masks_tensor])
            for k, v in losses_2d_.items():
                losses_2d[k + '_2d'] = v
            losses.update(losses_2d)

        return losses

    def forward_test(
        self,
        img,
        img_metas,
        input_heatmaps=None,
        **kwargs
    ):
        if self.backbone is None:
            assert input_heatmaps is not None
            feature_maps = []
            for input_heatmap in input_heatmaps:
                feature_maps.append(input_heatmap[0])
        else:
            feature_maps = []
            assert isinstance(img, list)
            for img_ in img:
                feature_maps.append(self.predict_heatmap(img_)[0])

        result = {}

        human_candidates_init, human_candidates = self.human_detector.forward_test(
            None, img_metas, feature_maps, **kwargs)
        result['human_detection_3d_init'] = human_candidates_init.cpu().numpy()
        result['human_detection_3d'] = human_candidates.cpu().numpy()

        human_poses_init = self.pose_regressor(
            None,
            img_metas,
            return_loss=False,
            feature_maps=[f[:, -self.num_joints:] for f in feature_maps],
            human_candidates=human_candidates)
        result['pose_3d_init'] = human_poses_init.cpu().numpy()

        if self.pose_refiner is not None and self.test_with_refine:
            human_poses = self.pose_refiner.forward_test(human_poses_init, feature_maps, img_metas)
        else:
            human_poses = human_poses_init.clone()
        result['pose_3d'] = human_poses.cpu().numpy()

        result['sample_id'] = [img_meta['sample_id'] for img_meta in img_metas]

        return result

    def show_result(self,
                img,
                img_metas,
                visualize_2d=False,
                input_heatmaps=None,
                dataset_info=None,
                radius=4,
                thickness=2,
                out_dir=None,
                show=False):
        """Visualize the results."""
        result = self.forward_test(
            img, img_metas, input_heatmaps=input_heatmaps)
        pose_3d = result['pose_3d']
        sample_id = result['sample_id']
        batch_size = pose_3d.shape[0]
        # get kpts and skeleton structure

        for i in range(batch_size):
            # visualize 3d results
            img_meta = img_metas[i]
            num_cameras = len(img_meta['camera'])
            pose_3d_i = pose_3d[i]
            pose_3d_i = pose_3d_i[pose_3d_i[:, 0, 3] >= 0]

            num_persons, num_keypoints, _ = pose_3d_i.shape
            pose_3d_list = [p[..., [0, 1, 2, 4]]
                            for p in pose_3d_i] if num_persons > 0 else []
            img_3d = imshow_multiview_keypoints_3d(
                pose_3d_list,
                skeleton=dataset_info.skeleton,
                pose_kpt_color=dataset_info.pose_kpt_color[:num_keypoints],
                pose_link_color=dataset_info.pose_link_color,
                space_size=self.human_detector.space_size,
                space_center=self.human_detector.space_center)
            if out_dir is not None:
                mmcv.image.imwrite(
                    img_3d,
                    os.path.join(out_dir, 'vis_3d', f'{sample_id[i]}_3d.jpg'))

            if visualize_2d:
                for j in range(num_cameras):
                    single_camera = SimpleCamera(img_meta['camera'][j])
                    # img = mmcv.imread(img)
                    if num_persons > 0:
                        pose_2d = np.ones_like(pose_3d_i[..., :3])
                        pose_2d_flat = single_camera.world_to_pixel(
                            pose_3d_i[..., :3].reshape((-1, 3)))
                        pose_2d[..., :2] = pose_2d_flat.reshape(
                            (num_persons, -1, 2))
                        pose_2d_list = [pose for pose in pose_2d]
                    else:
                        pose_2d_list = []
                    with tempfile.TemporaryDirectory() as tmpdir:
                        if 'image_file' in img_meta:
                            img_file = img_meta['image_file'][j]
                        else:
                            img_size = img_meta['center'][j] * 2
                            img = np.zeros(
                                [int(img_size[1]),
                                 int(img_size[0]), 3],
                                dtype=np.uint8)
                            img.fill(255)  # or img[:] = 255
                            img_file = os.path.join(tmpdir, 'tmp.jpg')
                            mmcv.image.imwrite(img, img_file)
                        img = imshow_keypoints(
                            img_file, pose_2d_list, dataset_info.skeleton, 0.0,
                            dataset_info.pose_kpt_color[:num_keypoints],
                            dataset_info.pose_link_color, radius, thickness)
                    if out_dir is not None:
                        mmcv.image.imwrite(
                            img,
                            os.path.join(out_dir,
                                         f'{sample_id[i]}_{j}_2d.jpg'))
                    # TODO: show image

    def forward_dummy(self, img, input_heatmaps=None, num_candidates=5):
        """Used for computing network FLOPs."""
        if self.backbone is None:
            assert input_heatmaps is not None
            feature_maps = []
            for input_heatmap in input_heatmaps:
                feature_maps.append(input_heatmap[0])
        else:
            feature_maps = []
            assert isinstance(img, list)
            for img_ in img:
                feature_maps.append(self.predict_heatmap(img_)[0])

        _ = self.human_detector.forward_dummy(feature_maps)

        _ = self.pose_regressor.forward_dummy(feature_maps, num_candidates)