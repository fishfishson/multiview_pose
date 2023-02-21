#!/bin/bash

# NOTE: Lines starting with "#SBATCH" are valid SLURM commands or statements,
#       while those starting with "#" and "##SBATCH" are comments.  Uncomment
#       "##SBATCH" line means to remove one # and start with #SBATCH to be a
#       SLURM command or statement.


#SBATCH -J MVPOSE-CHI3D #Slurm job name

# Set the maximum runtime, uncomment if you need it
##SBATCH -t 48:00:00 #Maximum runtime of 48 hours

# Enable email notificaitons when job begins and ends, uncomment if you need it
##SBATCH --mail-user=user_name@ust.hk #Update your email address
##SBATCH --mail-type=begin
##SBATCH --mail-type=end

# Choose partition (queue) "x-gpu" or "x-gpu-share"
#SBATCH -p x-gpu

# To use 4 cpu cores and 2 rtx2080ti gpu devices in a node -- gpu resources can be gpu:rtx2080ti or gpu:rtx6000
#SBATCH -N 1 -n 8 --gres=gpu:rtx2080ti:4

# Setup runtime environment if necessary 
# or you can source ~/.bashrc or ~/.bash_profile 

# Go to the job submission directory and run your application 
# cd $HOME/xgpu-scratch/app/
# Execute applications in parallel 
srun -n 1 --cpus-per-task 8 --gres=gpu:rtx2080ti:4 bash tools/dist_train.sh \
    configs/body/3d_kpt_mview_rgb_img/graph_pose/chi3d/gcn_cpn80x80x20_chi3d_cam4_end2end_split.py 4
