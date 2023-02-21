import argparse
from mmcv import Config
from multiview_pose.datasets import build_dataset

parser = argparse.ArgumentParser()
parser.add_argument('config')
args = parser.parse_args()
cfg = Config.fromfile(args.config)
train_dataset = build_dataset(cfg.data.train)
val_dataset = build_dataset(cfg.data.val)