# Single GPU

cd /mnt/amlt_code/SuPreM/target_applications/totalsegmentator
RANDOM_PORT=$((RANDOM % 64512 + 1024))
datapath=/mnt/amlt_code/prostate_data/ # change to /path/to/your/data/TotalSegmentator
checkpoint=tang
python download.py  --checkpoint $checkpoint
target_task=vertebrae
num_target_class=3
num_target_annotation=64
fold=5
for i in $(seq 0 $((fold-1))); do
    python -W ignore -m torch.distributed.launch --nproc_per_node=1 --master_port=$RANDOM_PORT train.py --dist False --model_backbone $arch --log_name efficiency.$arch.$target_task.number$num_target_annotation --map_type $target_task --num_class $num_target_class --dataset_path $datapath --num_workers 8 --batch_size 2 --pretrain $suprem_path --percent $num_target_annotation --fold_t $i --fold $fold
done 
