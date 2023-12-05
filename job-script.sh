#!/bin/bash
#SBATCH --account=def-ssfels
#SBATCH --gres=gpu:4
#SBTACH --mem=32G
#SBATCH --cpus-per-task=12  
#SBATCH --mem-per-cpu=16384
#SBTACH --nodes=2
#SBTACH --time=2:00:00
#SBTACH --mail-user=zxia0101@student.ubc.ca
#SBTACH --mail-type=ALL

cd /home/yrebuilt/projects/def-ssfels/yrebuilt/eagle-eyes-hackathon
module purge
module load python/3.8.10 scipy-stack
source ~/eagle/bin/activate

python train_coment.py
