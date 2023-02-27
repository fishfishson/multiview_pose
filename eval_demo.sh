EXP=$1
EPOCH=$2
python tools/demo.py \
    configs/body/3d_kpt_mview_rgb_img/graph_pose/demo/crash.py \
    work_dirs/$EXP/epoch_$EPOCH.pth \
    --out exp-out/$EXP-crash@$EPOCH
python tools/demo.py \
    configs/body/3d_kpt_mview_rgb_img/graph_pose/demo/dance.py \
    work_dirs/$EXP/epoch_$EPOCH.pth \
    --out exp-out/$EXP-dance@$EPOCH
python tools/demo.py \
    configs/body/3d_kpt_mview_rgb_img/graph_pose/demo/fight.py \
    work_dirs/$EXP/epoch_$EPOCH.pth \
    --out exp-out/$EXP-fight@$EPOCH
python tools/demo.py \
    configs/body/3d_kpt_mview_rgb_img/graph_pose/demo/511.py \
    work_dirs/$EXP/epoch_$EPOCH.pth \
    --out exp-out/$EXP-511@$EPOCH
