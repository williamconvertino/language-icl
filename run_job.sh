#!/bin/bash
#SBATCH --job-name=icl-language
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --partition=scavenger-gpu
#SBATCH --exclude=dcc-youlab-gpu-28,dcc-gehmlab-gpu-56
#SBATCH --nodelist=dcc-allenlab-gpu-[01-04],dcc-allenlab-gpu-[05-12],dcc-majoroslab-gpu-[01-08],dcc-yaolab-gpu-[01-08],dcc-wengerlab-gpu-01,dcc-engelhardlab-gpu-[01-04],dcc-motesa-gpu-[01-04],dcc-pbenfeylab-gpu-[01-04],dcc-vossenlab-gpu-[01-04],dcc-youlab-gpu-[01-56],dcc-mastatlab-gpu-01,dcc-viplab-gpu-01,dcc-youlab-gpu-57
#HIGH VRAM --nodelist=dcc-allenlab-gpu-[01-04],dcc-allenlab-gpu-[05-12],dcc-majoroslab-gpu-[01-08]
#SBATCH --requeue
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -e
source ~/.bashrc
conda activate icl

cd ..
python main.py "$@"