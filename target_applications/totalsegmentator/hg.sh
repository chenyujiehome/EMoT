#!/bin/bash
#SBATCH --job-name=suprem-total

#SBATCH -N 1
#SBATCH -n 12
#SBATCH -G a100:1
##SBATCH --exclusive
#SBATCH --mem=100G
#SBATCH -p general
#SBATCH -t 5-00:00:00
#SBATCH -q public

#SBATCH -o %x_slurm_%j.out     
#SBATCH -e %xslurm_%j.err      
#SBATCH --mail-type=ALL 
#SBATCH --mail-user=zzhou82@asu.edu

module load mamba/latest # only for Sol

# mamba create -n suprem python=3.9
source activate suprem

# pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
# pip install monai[all]==0.9.0
# pip install -r requirements.txt

# cd pretrained_weights/
# wget https://www.dropbox.com/scl/fi/gd1d7k9mac5azpwurds66/supervised_suprem_swinunetr_2100.pth?rlkey=xoqr7ey52rnese2k4hwmrlqrt
# mv supervised_suprem_swinunetr_2100.pth\?rlkey\=xoqr7ey52rnese2k4hwmrlqrt supervised_suprem_swinunetr_2100.pth
# cd ../

RANDOM_PORT=$((RANDOM % 64512 + 1024))
datapath=/scratch/zzhou82/data/Totalsegmentator_dataset/Totalsegmentator_dataset/ 
# change to /path/to/your/data/TotalSegmentator
arch=$1 
# support swinunetr, unet, and segresnet
target_task=$2
num_target_class=$3
fold=$4
# the maximum number of target annotations is 1081 for the whole training dataset
if [ "$arch" == "segresnet" ]; then suprem_path=pretrained_weights/supervised_suprem_segresnet_2100.pth; elif [ "$arch" == "unet" ]; then suprem_path=pretrained_weights/supervised_suprem_unet_2100.pth; else echo "Error in arch"; fi
log_name=train.$arch.$target_task.fold$fold
checkpoint_path=out/$log_name/best_model.pth

### Training 

python -W ignore -m torch.distributed.launch --nproc_per_node=1 --master_port=$RANDOM_PORT train.py --dist --model_backbone $arch --log_name $log_name --map_type $target_task --num_class $num_target_class --dataset_path $datapath --num_workers 12 --batch_size 8 --pretrain $suprem_path --fold $fold

# for arch in segresnet; do for fold in 1 2 3 4 5; do sbatch --error=logs/train.$arch.cardiac.fold$fold.out --output=logs/train.$arch.cardiac.fold$fold.out hg.sh  $arch cardiac 19 $fold; done; done

### Testing

# python -W ignore -m torch.distributed.launch --nproc_per_node=1 --master_port=$RANDOM_PORT test.py --dist  --model_backbone $arch --log_name efficiency.$arch.$target_task.number$num_target_annotation --map_type $target_task --num_class $num_target_class --dataset_path $datapath --num_workers 12 --batch_size 2 --pretrain $checkpoint_path 

# # for num_target_annotation in 64 128 256 512 1024; do sbatch --error=logs/test.organs.$num_target_annotation.out --output=logs/test.organs.$num_target_annotation.out hg.sh organs 18 $num_target_annotation; done

# # for num_target_annotation in 64 128 256 512 1024; do sbatch --error=logs/test.muscles.$num_target_annotation.out --output=logs/test.muscles.$num_target_annotation.out hg.sh muscles 22 $num_target_annotation; done

# # for num_target_annotation in 64 128 256 512 1024; do sbatch --error=logs/test.cardiac.$num_target_annotation.out --output=logs/test.cardiac.$num_target_annotation.out hg.sh cardiac 19 $num_target_annotation; done

# # for num_target_annotation in 64 128 256 512 1024; do sbatch --error=logs/test.vertebrae.$num_target_annotation.out --output=logs/test.vertebrae.$num_target_annotation.out hg.sh vertebrae 25 $num_target_annotation; done

# # for num_target_annotation in 64 128 256 512 1024; do sbatch --error=logs/test.ribs.$num_target_annotation.out --output=logs/test.ribs.$num_target_annotation.out hg.sh ribs 25 $num_target_annotation; done