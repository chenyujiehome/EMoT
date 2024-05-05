# Single GPU

cd /home/azureuser/SuPreM/target_applications/totalsegmentator
datapath=/home/azureuser/prostate_data/ # change to /path/to/your/data/TotalSegmentator
checkpoint=sup_unet
python download_pretrain.py  --checkpoint $checkpoint
target_task=prostate
num_target_class=3
num_target_annotation=64
fold=5
for i in $(seq 0 $((fold-1))); do
    checkpoint_path=/home/azureuser/pretrain_mri/pretrian_k_fold_prostate/efficiency.$checkpoint.$target_task.number$num_target_annotation/model_fold_$i.pth
    # python -W ignore -m torch.distributed.launch --nproc_per_node=1  train.py  --checkpoint $checkpoint --log_name efficiency.$checkpoint.$target_task.number$num_target_annotation --map_type $target_task --num_class $num_target_class --dataset_path $datapath --num_workers 8 --batch_size 2  --percent $num_target_annotation --fold_t $i --fold $fold
    python -W ignore -m torch.distributed.launch --nproc_per_node=1  test.py  --checkpoint $checkpoint --log_name efficiency.$checkpoint.$target_task.number$num_target_annotation --map_type $target_task --num_class $num_target_class --dataset_path $datapath --num_workers 8 --batch_size 2 --pretrain $checkpoint_path --train_type efficiency --fold_t $i --fold $fold
done 
