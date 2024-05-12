
# 定义一个数组
checkpoints=('tang' 'jose' 'univ_swin' 'voco' 'sup_swin' 'genesis' 'unimiss_tiny' 'unimiss_small' 'med3d' 'dodnet' 'univ_unet' 'sup_unet' 'sup_seg')
cd /home/fanlinghuang/TAD-chenyujie/SuPreM/target_applications/totalsegmentator
# 遍历数组
for checkpoint in "${checkpoints[@]}"; do
    # 调用 download_pretrain.py 并传递 checkpoint 参数
    python3 download_pretrain.py --save_path /home/fanlinghuang/TAD-chenyujie/dataset/checkpointpath/ --checkpoint $checkpoint
done