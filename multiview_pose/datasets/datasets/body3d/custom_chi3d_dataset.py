# Copyright (c) OpenMMLab. All rights reserved.
import copy
import os
import glob
import os.path as osp
import json
import warnings
import pickle

from collections import OrderedDict
import tempfile
import mmcv
import numpy as np
from matplotlib import pyplot as plt

from mmcv import Config, deprecated_api_warning

from mmpose.core.camera import SimpleCamera
from mmpose.datasets.builder import DATASETS
from mmpose.datasets.datasets.base import Kpt3dMviewRgbImgDirectDataset 

from easymocap.mytools.camera_utils import read_cameras
from easymocap.mytools.file_utils import save_numpy_dict

body25topanoptic15 = [1,0,8,5,6,7,12,13,14,2,3,4,9,10,11]

@DATASETS.register_module()
class CustomCHI3DDataset(Kpt3dMviewRgbImgDirectDataset):
    ALLOWED_METRICS = {'mpjpe', 'mAP'}

    def __init__(self,
                ann_file,
                img_prefix,
                data_cfg,
                pipeline,
                dataset_info=None,
                test_mode=False):

        if dataset_info is None:
            warnings.warn(
                'dataset_info is missing. '
                'Check https://github.com/open-mmlab/mmpose/pull/663 '
                'for details.', DeprecationWarning)
            cfg = Config.fromfile('configs/_base_/datasets/chi3d_body3d.py')
            dataset_info = cfg._cfg_dict['dataset_info']

        super().__init__(
            ann_file,
            img_prefix,
            data_cfg,
            pipeline,
            dataset_info=dataset_info,
            test_mode=test_mode)
        
        self.joint_type = 'body25'

        self.load_config(data_cfg)
        self.ann_info['use_different_joint_weights'] = False

        if ann_file is None:
            self.db_file = osp.join(
                img_prefix, f'mvpose_{self.subset}_cam{self.num_cameras}.pkl')
        else:
            self.db_file = ann_file

        seq_list = os.listdir(self.img_prefix)
        new_seq_list = [x for x in seq_list if x[:3] in self.seq_list]
        self.seq_list = new_seq_list

        if osp.exists(self.db_file):
            with open(self.db_file, 'rb') as f:
                info = pickle.load(f)
            assert info['sequence_list'] == self.seq_list
            assert info['interval'] == self.seq_frame_interval
            assert info['cam_list'] == self.cam_list
            assert info['joint_type'] == self.joint_type
            self.db = info['db']
        else:
            self.db = self._get_db()
            info = {
                'sequence_list': self.seq_list,
                'interval': self.seq_frame_interval,
                'cam_list': self.cam_list,
                'joint_type': self.joint_type,
                'db': self.db
            }
            with open(self.db_file, 'wb') as f:
                pickle.dump(info, f)

        self.db_size = len(self.db)

        print(f'=> load {len(self.db)} samples')
    
    def load_config(self, data_cfg):
        self.num_joints = data_cfg['num_joints']
        assert self.num_joints <= 19
        self.seq_list = data_cfg['seq_list']
        self.cam_list = data_cfg['cam_list']
        self.num_cameras = data_cfg['num_cameras']
        assert self.num_cameras == len(self.cam_list)
        self.seq_frame_interval = data_cfg.get('seq_frame_interval', 1)
        self.subset = data_cfg.get('subset', 'train')
        self.need_camera_param = True
        self.root_id = data_cfg.get('root_id', 0)
        self.max_persons = data_cfg.get('max_num', 10)

    def _get_cam(self, seq):
        calib = read_cameras(osp.join(self.img_prefix, seq))

        cameras = {}
        for k, v in calib.items():
            # if k not in self.cam_list: continue
            sel_cam = {}
            R_w2c = np.array(v['R'])
            T_w2c = np.array(v['T']).reshape((3, 1)) * 1000.0
            R_c2w = R_w2c.T
            T_c2w = -R_w2c.T @ T_w2c
            sel_cam['R'] = R_c2w.tolist()
            sel_cam['T'] = T_c2w.tolist()
            sel_cam['K'] = np.array(v['K'])[:2]
            distCoef = np.array(v['dist']).flatten()
            sel_cam['k'] = [distCoef[0], distCoef[1], distCoef[4]]
            sel_cam['p'] = [distCoef[2], distCoef[3]]
            cameras[k] = sel_cam

        return cameras 
    
    def _get_db(self):
        width = 900
        height = 900
        db = []
        sample_id = 0
        for seq in self.seq_list:
            cameras = self._get_cam(seq)
            curr_anno = osp.join(self.img_prefix, seq, self.joint_type)
            anno_files = sorted(glob.iglob('{:s}/*.json'.format(curr_anno)))
            print(f'load sequence: {seq}', flush=True)
            for i, file in enumerate(anno_files):
                if i % self.seq_frame_interval == 0:
                    with open(file) as dfile:
                        bodies = json.load(dfile)
                    if len(bodies) == 0:
                        continue 
                
                    for k, cam_param in cameras.items():
                        single_view_camera = SimpleCamera(cam_param)
                        image_file = osp.join(seq, 'images', k, osp.basename(file))
                        image_file = image_file.replace('json', 'jpg')
                        
                        all_poses_3d = np.zeros(
                            (self.max_persons, self.num_joints, 3),
                            dtype=np.float32)
                        all_poses_vis_3d = np.zeros(
                            (self.max_persons, self.num_joints, 3),
                            dtype=np.float32)
                        all_roots_3d = np.zeros((self.max_persons, 3),
                                                dtype=np.float32)
                        all_poses = np.zeros(
                            (self.max_persons, self.num_joints, 3),
                            dtype=np.float32)
                        
                        cnt = 0
                        person_ids = -np.ones(self.max_persons, dtype=np.int64)
                        for body in bodies:
                            if cnt >= self.max_persons:
                                break
                            pose3d = np.array(body['keypoints3d'])
                            if pose3d.shape[-1] == 4:
                                pose3d = pose3d[..., :3]
                            pos3d = pose3d.reshape(-1, 3)
                            if self.joint_type == 'body25':
                                pose3d = pose3d[body25topanoptic15]
                            pose3d = pose3d[:self.num_joints]
                            pose3d = np.concatenate([pose3d, np.ones_like(pose3d[:, :1])], axis=-1)

                            joints_vis = pose3d[:, -1] > 0.1

                            if not joints_vis[self.root_id]:
                                continue

                            pose3d[:, :3] = pose3d[:, :3] * 1000.0
                            
                            all_poses_3d[cnt] = pose3d[:, :3]
                            all_roots_3d[cnt] = pose3d[self.root_id, :3]
                            all_poses_vis_3d[cnt] = np.repeat(
                                np.reshape(joints_vis, (-1, 1)), 3, axis=1)

                            pose2d = np.zeros((pose3d.shape[0], 3))
                            # get pose_2d from pose_3d
                            pose2d[:, :2] = single_view_camera.world_to_pixel(
                                pose3d[:, :3])
                            x_check = np.bitwise_and(pose2d[:, 0] >= 0,
                                                     pose2d[:, 0] <= width - 1)
                            y_check = np.bitwise_and(
                                pose2d[:, 1] >= 0, pose2d[:, 1] <= height - 1)
                            check = np.bitwise_and(x_check, y_check)
                            joints_vis[np.logical_not(check)] = 0
                            pose2d[:, -1] = joints_vis

                            all_poses[cnt] = pose2d
                            person_ids[cnt] = body['id']
                            cnt += 1

                        if cnt > 0:
                            db.append({
                                'seq': seq,
                                # '_'.join([seq, k, osp.basename(file).split('.')[0]]),
                                'frame': osp.basename(file).split('.')[0],
                                'image_file':
                                osp.join(self.img_prefix, image_file),
                                'joints_3d':
                                all_poses_3d,
                                'person_ids':
                                person_ids,
                                'joints_3d_visible':
                                all_poses_vis_3d,
                                'joints': [all_poses],
                                'roots_3d':
                                all_roots_3d,
                                'camera':
                                cam_param,
                                'num_persons':
                                cnt,
                                'sample_id':
                                sample_id,
                                'center':
                                np.array((width / 2, height / 2),
                                         dtype=np.float32),
                                'scale':
                                self._get_scale((width, height))
                            })
                            sample_id += 1
        return db

    def evaluate(self, outputs, res_folder=None, metric='mpjpe', **kwargs):
        """

        Args:
            outputs list(dict(pose_3d, sample_id)):
                pose_3d (np.ndarray): predicted 3D human pose
                sample_id (np.ndarray): sample id of a frame.
            res_folder (str): Path of directory to save the results.
            metric (str | list[str]): Metric to be performed.
                Defaults: 'mpjpe'.
            **kwargs:

        Returns:

        """
        pose_3ds = np.concatenate([output['pose_3d'] for output in outputs],
                                  axis=0)
        center_3ds = np.concatenate([output['human_detection_3d'][..., None, :]
                                     for output in outputs],
                                    axis=0)

        sample_ids = []
        for output in outputs:
            sample_ids.extend(output['sample_id'])
        _outputs = [
            dict(sample_id=sample_id, pose_3d=pose_3d, center_3d=center_3d)
            for (sample_id, pose_3d, center_3d) in zip(sample_ids, pose_3ds, center_3ds)
        ]
        _outputs = self._sort_and_unique_outputs(_outputs, key='sample_id')

        metrics = metric if isinstance(metric, list) else [metric]
        for _metric in metrics:
            if _metric not in self.ALLOWED_METRICS:
                raise ValueError(
                    f'Unsupported metric "{_metric}"'
                    f'Supported metrics are {self.ALLOWED_METRICS}')

        # if res_folder is not None:
        #     tmp_folder = None
        #     res_file = osp.join(res_folder, 'result_keypoints.json')
        # else:
        #     tmp_folder = tempfile.TemporaryDirectory()
        #     res_file = osp.join(tmp_folder.name, 'result_keypoints.json')

        # mmcv.dump(_outputs, res_file)

        results = dict()
        results.update(self._evaluate(_outputs, metrics))

        results.update(self._evaluate(_outputs, metrics,
                                      eval_name='center_3d',
                                      suffix='_c',
                                      joint_ids=[self.root_id]))
        # if tmp_folder is not None:
        #     tmp_folder.cleanup()

        return results
    
    def visualize(self, outputs, res_folder=None, **kwargs):
        pose_3ds = np.concatenate([output['pose_3d'] for output in outputs],
                                  axis=0)
        pose_3d_inits = np.concatenate([output['pose_3d_init'] for output in outputs],
                                       axis=0)
        center_3ds = np.concatenate([output['human_detection_3d'][..., None, :]
                                    for output in outputs],
                                    axis=0)
        center_3d_inits = np.concatenate([output['human_detection_3d_init'][..., None, :]
                                         for output in outputs],
                                         axis=0)

        sample_ids = []
        for output in outputs:
            sample_ids.extend(output['sample_id'])
        _outputs = [
            dict(
                sample_id=sample_id, 
                pose_3d=pose_3d, 
                center_3d=center_3d, 
                pose_3d_init=pose3d_init,
                center_3d_init=center_3d_init)
            for (sample_id, pose_3d, center_3d, pose3d_init, center_3d_init) in \
                zip(sample_ids, pose_3ds, center_3ds, pose_3d_inits, center_3d_inits)
        ]
        _outputs = self._sort_and_unique_outputs(_outputs, key='sample_id')

        if res_folder is not None:
            tmp_folder = None
            vis_folder = osp.join(res_folder, 'blenderfig') 
            os.makedirs(vis_folder, exist_ok=True)
        else:
            tmp_folder = tempfile.TemporaryFile()
            vis_folder = osp.join(tmp_folder.name, 'blenderfig')

        # gt_num = len(_outputs)
        gt_num = self.db_size // self.num_cameras
        assert len(_outputs) == gt_num, f'number mismatch: {len(_outputs)}, {gt_num}'
        
        for i in range(gt_num):
            index = self.num_cameras * i
            db_rec = copy.deepcopy(self.db[index])
            joints_3d = db_rec['joints_3d']
            joints_3d_vis = db_rec['joints_3d_visible']

            if joints_3d_vis.sum() < 1:
                continue

            gt = []
            for (joint, joint_vis) in zip(joints_3d, joints_3d_vis):
                vis = joint_vis[:, 0] > 0
                if vis.sum() < 1:
                   continue 
                gt.append(joint)
            gt = np.stack(gt)
            gt = np.concatenate([gt / 1000.0, np.ones_like(gt[..., :1])], axis=-1)
                
            pred = _outputs[i]['pose_3d'].copy()
            pred = pred[pred[:, 0, 3] >= 0]
            pred = np.concatenate([pred[..., :3] / 1000.0, np.ones_like(pred[..., :1])], axis=-1)

            pred_init = _outputs[i]['pose_3d_init'].copy()
            pred_init = pred_init[pred_init[:, 0, 3] >= 0]
            pred_init = np.concatenate([pred_init[..., :3] / 1000.0, np.ones_like(pred_init[..., :1])], axis=-1)

            center = _outputs[i]['center_3d'].copy()
            center = center[center[:, 0, 3] >= 0]
            center = np.concatenate([center[..., :3] / 1000.0, np.ones_like(center[..., :1])], axis=-1)

            center_init = _outputs[i]['center_3d_init'].copy()
            center_init = center_init[center_init[:, 0, 3] >= 0]
            center_init = np.concatenate([center_init[..., :3] / 1000.0, np.ones_like(center_init[..., :1])], axis=-1)

            data = {
                'gt': gt,
                'pred': pred,
                'center': center,
                'pred_init': pred_init,
                'center_init': center_init,
            }
            key = db_rec['key'].split('_')
            name = '_'.join([key[0], key[1], key[-1]]) + '.json'
            name = osp.join(vis_folder, name)
            save_numpy_dict(name, data)

            # if 'Hug' in key[1]:
            #     plt.figure(0, figsize=(4, 4))
            #     plt.subplots_adjust(left=0.05, right=0.95, bottom=0.05,
            #                         top=0.95, wspace=0.05, hspace=0.15)
            #     num_persons = db_rec['num_persons']
            #     ax = plt.subplot(1, 1, 1, projection='3d')
            #     for n in range(num_persons):
            #         joint = gt[n]
            #         for k in self.ann_info['skeleton']:
            #             # if joint_vis[k[0], 0] and joint_vis[k[1], 0]:
            #             x = [float(joint[k[0], 0]), float(joint[k[1], 0])]
            #             y = [float(joint[k[0], 1]), float(joint[k[1], 1])]
            #             z = [float(joint[k[0], 2]), float(joint[k[1], 2])]
            #             ax.plot(x, y, z, c='r', lw=1.5, marker='o', markerfacecolor='w', markersize=2,
            #                     markeredgewidth=1)
            #             # else:
            #             #     x = [float(joint[k[0], 0]), float(joint[k[1], 0])]
            #             #     y = [float(joint[k[0], 1]), float(joint[k[1], 1])]
            #             #     z = [float(joint[k[0], 2]), float(joint[k[1], 2])]
            #             #     ax.plot(x, y, z, c='r', ls='--', lw=1.5, marker='o', markerfacecolor='w', markersize=2,
            #             #             markeredgewidth=1)
            #     colors = ['b', 'g', 'c', 'y', 'm', 'orange', 'pink', 'royalblue', 'lightgreen', 'gold']
            #     for n in range(len(pred)):
            #         joint = pred[n]
            #         # if joint[0, 3] >= 0:
            #         for k in self.ann_info['skeleton']:
            #             x = [float(joint[k[0], 0]), float(joint[k[1], 0])]
            #             y = [float(joint[k[0], 1]), float(joint[k[1], 1])]
            #             z = [float(joint[k[0], 2]), float(joint[k[1], 2])]
            #             ax.plot(x, y, z, c=colors[int(n % 10)], lw=1.5, marker='o', markerfacecolor='w', markersize=2,
            #                     markeredgewidth=1)
            #     plt.savefig(name.replace('json', 'jpg'))
            #     plt.close(0)

    def _evaluate(self, _outputs, metrics, eval_name='pose_3d', suffix='', joint_ids=None):
        eval_list = []
        gt_num = self.db_size // self.num_cameras
        assert len(
            _outputs) == gt_num, f'number mismatch: {len(_outputs)}, {gt_num}'

        total_gt = 0
        for i in range(gt_num):
            index = self.num_cameras * i
            db_rec = copy.deepcopy(self.db[index])
            joints_3d = db_rec['joints_3d']
            joints_3d_vis = db_rec['joints_3d_visible']

            if joints_3d_vis.sum() < 1:
                continue

            pred = _outputs[i][eval_name].copy()
            pred = pred[pred[:, 0, 3] >= 0]
            for pose in pred:
                pjpes = []
                mpjpes = []
                for (gt, gt_vis) in zip(joints_3d, joints_3d_vis):
                    if joint_ids is not None:
                        gt = gt[joint_ids]
                        gt_vis = gt_vis[joint_ids]
                    vis = gt_vis[:, 0] > 0
                    if vis.sum() < 1:
                        break
                    pjpe = np.sqrt(
                        np.sum((pose[vis, 0:3] - gt[vis])**2, axis=-1))
                    pjpes.append(pjpe)
                    mpjpe = np.mean(pjpe)
                    mpjpes.append(mpjpe)
                min_gt = np.argmin(mpjpes)
                min_mpjpe = np.min(mpjpes)
                score = pose[0, 4]
                eval_list.append({
                    'mpjpe': float(min_mpjpe),
                    'score': float(score),
                    'gt_id': int(total_gt + min_gt),
                    'pjpes': pjpes
                })

            total_gt += (joints_3d_vis[:, :, 0].sum(-1) >= 1).sum()

        mpjpe_threshold = np.arange(25, 155, 25)
        aps = []
        ars = []
        for t in mpjpe_threshold:
            ap, ar = self._eval_list_to_ap(eval_list, total_gt, t)
            aps.append(ap)
            ars.append(ar)

        name_value_tuples = []
        for _metric in metrics:
            if _metric == 'mpjpe':
                stats_names = ['RECALL 500mm', 'MPJPE 500mm']
                for i, stats_name in enumerate(stats_names):
                    stats_names[i] = stats_name + suffix
                info_str = list(
                    zip(stats_names, [
                        self._eval_list_to_recall(eval_list, total_gt),
                        self._eval_list_to_mpjpe(eval_list)
                    ]))
            elif _metric == 'mAP':
                stats_names = [
                    'AP 25', 'AP 50', 'AP 75', 'AP 100', 'AP 125', 'AP 150',
                    'mAP', 'AR 25', 'AR 50', 'AR 75', 'AR 100', 'AR 125',
                    'AR 150', 'mAR'
                ]
                for i, stats_name in enumerate(stats_names):
                    stats_names[i] = stats_name + suffix
                mAP = np.array(aps).mean()
                mAR = np.array(ars).mean()
                info_str = list(zip(stats_names, aps + [mAP] + ars + [mAR]))
            else:
                raise NotImplementedError
            name_value_tuples.extend(info_str)

        return OrderedDict(name_value_tuples)

    @staticmethod
    def _sort_and_unique_outputs(outputs, key='sample_id'):
        """sort outputs and remove the repeated ones."""
        outputs = sorted(outputs, key=lambda x: x[key])
        num_outputs = len(outputs)
        for i in range(num_outputs - 1, 0, -1):
            if outputs[i][key] == outputs[i - 1][key]:
                del outputs[i]

        return outputs
    
    @staticmethod
    def _eval_list_to_ap(eval_list, total_gt, threshold):
        """Get Average Precision (AP) and Average Recall at a certain
        threshold."""

        eval_list.sort(key=lambda k: k['score'], reverse=True)
        total_num = len(eval_list)

        tp = np.zeros(total_num)
        fp = np.zeros(total_num)
        gt_det = []
        for i, item in enumerate(eval_list):
            if item['mpjpe'] < threshold and item['gt_id'] not in gt_det:
                tp[i] = 1
                gt_det.append(item['gt_id'])
            else:
                fp[i] = 1
        tp = np.cumsum(tp)
        fp = np.cumsum(fp)
        recall = tp / (total_gt + 1e-5)
        precise = tp / (tp + fp + 1e-5)
        for n in range(total_num - 2, -1, -1):
            precise[n] = max(precise[n], precise[n + 1])

        precise = np.concatenate(([0], precise, [0]))
        recall = np.concatenate(([0], recall, [1]))
        index = np.where(recall[1:] != recall[:-1])[0]
        ap = np.sum((recall[index + 1] - recall[index]) * precise[index + 1])

        return ap, recall[-2]

    @staticmethod
    def _eval_list_to_mpjpe(eval_list, threshold=500):
        """Get MPJPE within a certain threshold."""
        eval_list.sort(key=lambda k: k['score'], reverse=True)
        gt_det = []

        mpjpes = []
        for i, item in enumerate(eval_list):
            if item['mpjpe'] < threshold and item['gt_id'] not in gt_det:
                mpjpes.append(item['mpjpe'])
                gt_det.append(item['gt_id'])

        return np.mean(mpjpes) if len(mpjpes) > 0 else np.inf
    
    @staticmethod
    def _eval_list_to_recall(eval_list, total_gt, threshold=500):
        """Get Recall at a certain threshold."""
        gt_ids = [e['gt_id'] for e in eval_list if e['mpjpe'] < threshold]

        return len(np.unique(gt_ids)) / total_gt

    def __getitem__(self, idx):
        results = {}
        for c in range(self.num_cameras):
            result = copy.deepcopy(self.db[self.num_cameras * idx + c])
            result['ann_info'] = self.ann_info
            width = 900
            height = 900
            result['mask'] = [np.ones((height, width), dtype=np.float32)]
            results[c] = result

        return self.pipeline(results)