# Single GPU

cd /mnt/amlt_code/SuPreM/target_applications/totalsegmentator
datapath=/mnt/amlt_code/prostate_data/ # change to /path/to/your/data/TotalSegmentator
checkpoints="genesis sup_unet sup_swin tang"
for checkpoint in $checkpoints; do
python download_pretrain.py  --checkpoint $checkpoint
target_task=prostate
num_target_class=3
num_target_annotation=64
checkpoint_path=out/efficiency.$checkpoint.$target_task.number$num_target_annotation/best.pth
python -W ignore -m torch.distributed.launch --nproc_per_node=1  train.py --dist False --checkpoint $checkpoint --log_name efficiency.$checkpoint.$target_task.number$num_target_annotation --map_type $target_task --num_class $num_target_class --dataset_path $datapath --num_workers 8 --batch_size 2  --percent $num_target_annotation 
python -W ignore -m torch.distributed.launch --nproc_per_node=1  test.py --dist False --checkpoint $checkpoint --log_name efficiency.$checkpoint.$target_task.number$num_target_annotation --map_type $target_task --num_class $num_target_class --dataset_path $datapath --num_workers 8 --batch_size 2 --pretrain $checkpoint_path --train_type efficiency 
done