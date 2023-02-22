import torch

model1 = torch.load('checkpoints/resnet_50_deconv.pth.tar')
model2 = torch.load('checkpoints/pose_resnet50_coco_rh_mvpose.pth')
# for k, v in model1.items():
#     print('{}: {}'.format(k, v.shape))
#     print('{}: {}'.format(k, model2[k].shape))
# print(len(model1)) 
# print(len(model2)) 