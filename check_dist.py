import numpy as np 
import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('path')
args = parser.parse_args()

with open(args.path, 'r') as f:
    data = json.load(f)
preds = np.array(data['pred'])[..., :3]
gts = np.array(data['gt'])[..., :3]

for pred in preds:
    pjpes = []
    mpjpes = []
    for gt in gts:
        pjpe = np.linalg.norm(pred - gt, axis=-1, ord=2)
        pjpes.append(pjpe)
        mpjpe = np.mean(pjpe) 
        mpjpes.append(mpjpe)
    print(pjpes[0])
    print(pjpes[1])
    print(mpjpes)