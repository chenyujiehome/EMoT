import os
import fnmatch
import argparse
import torch
import json
import sys
sys.path.append('/home/fanlinghuang/TAD-chenyujie/SuPreM/target_applications/totalsegmentator')
from model.SwinUNETR import SwinUNETR
from model.unet3d import UNet3D
from monai.networks.nets import SegResNet


# 目录路径
def main():
    directory_path = args.pretrain_path
    file_keys_dict = {}
    # 遍历目录下所有以.pt或.pth结尾的文件
    for root, dirs, files in os.walk(directory_path):
        for file in files:
            if fnmatch.fnmatch(file, '*.pt') or fnmatch.fnmatch(file, '*.pth'):
                file_path = os.path.join(root, file)
                # 加载文件
                data = torch.load(file_path)
                # 获取文件中所有的键
                keys = list(data.keys())
                # 将文件名和键添加到字典中
                file_keys_dict[file] = keys
                keys_to_check = ['state_dict', 'net', 'model', 'teacher']
                for key in keys_to_check:
                    if key in data:
                        file_keys_dict[file+'['+key+']'] = list(data[key].keys())
                        del file_keys_dict[file]
    model = SwinUNETR(img_size=(96,96,96),
                        in_channels=2,
                        out_channels=3,
                        feature_size=48,
                        drop_rate=0.0,
                        attn_drop_rate=0.0,
                        dropout_path_rate=0.0,
                        use_checkpoint=False
                        )
    store_dict = model.state_dict()
    file_keys_dict['swinunter'] = list(store_dict.keys())
    model = UNet3D(n_class=3)    
    store_dict = model.state_dict()
    file_keys_dict['unet'] = list(store_dict.keys())
    model = SegResNet(
                    blocks_down=[1, 2, 2, 4],
                    blocks_up=[1, 1, 1],
                    init_filters=16,
                    in_channels=1,
                    out_channels=3,
                    dropout_prob=0.0,
                    )
    store_dict = model.state_dict()
    file_keys_dict['segres'] = list(store_dict.keys())
    with open('file_keys_dict.json', 'w') as f:
        json.dump(file_keys_dict, f,indent=4)
    print(file_keys_dict)
            
if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Load pretrain model')
    parser.add_argument('--pretrain_path', type=str, default="/home/fanlinghuang/TAD-chenyujie/dataset/checkpointpath/", help='pretrain model path')
    # parser.add_argument('--save_path', type=str, default="/home/fanlinghuang/TAD-chenyujie/dataset/pretrain_mri/", help='pretrain model path')
    args = parser.parse_args()
    main()
    # print(args.pretrain_path)
    print("Done!")