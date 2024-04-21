import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import argparse
import time
import csv
import glob
import nibabel as nib
from collections import OrderedDict

import warnings
warnings.filterwarnings("ignore")

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from tensorboardX import SummaryWriter

from monai.losses import DiceCELoss
from monai.inferers import sliding_window_inference
from monai.data import load_decathlon_datalist, decollate_batch, DistributedSampler
from monai.transforms import AsDiscrete
from monai.metrics import DiceMetric

from model.SwinUNETR import SwinUNETR
from model.unet3d import UNet3D
from monai.networks.nets import SegResNet
from dataset.dataloader import get_loader
from utils.utils_test import dice_score, check_data, TEMPLATE, get_key, NUM_CLASS, surface_dice
from optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR

torch.multiprocessing.set_sharing_strategy('file_system')

dice_metric = DiceMetric(include_background=True, reduction="mean_batch", get_not_nans=False)
loss_function = DiceCELoss(to_onehot_y=True, softmax=True)

def validation(model, ValLoader, args):
    model.eval()
    dice_results = []  
    nsd_results = [] 
    post_label = AsDiscrete(to_onehot=args.num_class)
    post_pred = AsDiscrete(argmax=True, to_onehot=args.num_class)    
    spacing_dict = {}
    save_tensor={}
    for index, batch in enumerate(tqdm(ValLoader)):
        image, name_list = batch["image"].to(args.device), batch["name"]
        # Loop over each name in name list
        with torch.no_grad():
            val_outputs = sliding_window_inference(image,(args.roi_x, args.roi_y, args.roi_z), 1, model, overlap=args.overlap, mode='gaussian')
        
        val_outputs_list = decollate_batch(val_outputs)
        val_output_convert = [post_pred(val_pred_tensor) for val_pred_tensor in val_outputs_list]
        for i in range(len(name_list)):
            save_tensor[name_list[i]] = val_output_convert[i] 
    torch.save(save_tensor, f'save_tensor_{args.checkpoint}_{args.model_backbone}.pt')
    return 0,0



def process(args):
    rank = 0

    if args.dist:
        dist.init_process_group(backend="nccl", init_method="env://")
        rank = args.local_rank
    args.device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(args.device)

    # prepare the 3D model
    if args.model_backbone == 'segresnet':
        model = SegResNet(
                    blocks_down=[1, 2, 2, 4],
                    blocks_up=[1, 1, 1],
                    init_filters=16,
                    in_channels=1,
                    out_channels=args.num_class,
                    dropout_prob=0.0,
                    )
        store_dict = model.state_dict()
        model_dict = torch.load(args.pretrain)['net']
        amount = 0
        for key in model_dict.keys():
            new_key = '.'.join(key.split('.')[1:])
            if new_key in store_dict.keys():
                store_dict[new_key] = model_dict[key]   
                amount += 1
        print(amount, len(store_dict.keys()))
        model.load_state_dict(store_dict)
        print(f'Load SegResNet transfer learning weights')

    if args.model_backbone == 'swinunetr':
        model = SwinUNETR(img_size=(args.roi_x, args.roi_y, args.roi_z),
                    in_channels=2,
                    out_channels=args.num_class,
                    feature_size=48,
                    drop_rate=0.0,
                    attn_drop_rate=0.0,
                    dropout_path_rate=0.0,
                    use_checkpoint=False
                    )
        store_dict = model.state_dict()
        model_dict = torch.load(args.pretrain)['net']
        store_dict = model.state_dict()
        amount = 0
        for key in model_dict.keys():
            new_key = '.'.join(key.split('.')[1:])
            if new_key in store_dict.keys():
                store_dict[new_key] = model_dict[key]   
                amount += 1
        model.load_state_dict(store_dict)
        print(amount, len(store_dict.keys()))
        print(f'Load Swin UNETR transfer learning weights')


    if args.model_backbone == 'unet':
        model = UNet3D(n_class=args.num_class)
        model_dict = torch.load(args.pretrain)['net']
        store_dict = model.state_dict()
        amount = 0
        for key in model_dict.keys():
            new_key = '.'.join(key.split('.')[1:])
            if new_key in store_dict.keys():
                store_dict[new_key] = model_dict[key]   
                amount += 1
        model.load_state_dict(store_dict)
        print(amount, len(store_dict.keys()))
        print(f'Load Unet transfer learning weights')

    model.to(args.device)
    if args.dist:
        model = DistributedDataParallel(model, device_ids=[args.device])

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = LinearWarmupCosineAnnealingLR(optimizer, warmup_epochs=args.warmup_epoch, max_epochs=args.max_epoch)

    if args.resume:
        checkpoint = torch.load(args.resume)
        if args.dist:
            model.load_state_dict(checkpoint['net'])
        else:
            store_dict = model.state_dict()
            model_dict = checkpoint['net']
            for key in model_dict.keys():
                store_dict['.'.join(key.split('.'))] = model_dict[key]
            model.load_state_dict(store_dict)
        optimizer.load_state_dict(checkpoint['optimizer'])
        args.epoch = checkpoint['epoch']
        scheduler.load_state_dict(checkpoint['scheduler'])
        
        print('success resume from ', args.resume)

    torch.backends.cudnn.benchmark = True

    train_loader, train_sampler, val_loader, test_loader = get_loader(args)

    if rank == 0:
        writer = SummaryWriter(log_dir='out/' + args.log_name)
        print('Writing Tensorboard logs to ', 'out/' + args.log_name)

    mean_dice, std = validation(model, test_loader, args)
    print("Mean dice is:", mean_dice)
    dist.destroy_process_group()
    # assert 0 == 1
def main():
    parser = argparse.ArgumentParser()
    ## for distributed training
    parser.add_argument('--dist', dest='dist', type=bool, default=False,
                        help='distributed training or not')
    parser.add_argument("--local_rank", type=int)
    parser.add_argument("--device")
    parser.add_argument("--epoch", default=0)
    ## logging
    parser.add_argument('--log_name', default='efficiency.genesis.prostate.number64', help='The path resume from checkpoint')
    ## model load
    parser.add_argument('--resume', default=None, help='The path resume from checkpoint')
    parser.add_argument('--pretrain', default='/home/azureuser/SuPreM/target_applications/totalsegmentator/out/efficiency.genesis.prostate.number64/model.pth', 
                        help='The path of pretrain model')
    parser.add_argument('--trans_encoding', default='rand_embedding', 
                        help='the type of encoding: rand_embedding or word_embedding')
    parser.add_argument('--word_embedding', default='./pretrained_weights/word_embedding.pth', 
                        help='The path of word embedding')
    ## hyperparameter
    parser.add_argument('--max_epoch', default=1, type=int, help='Number of training epoches')
    parser.add_argument('--warmup_epoch', default=20, type=int, help='number of warmup epochs')
    parser.add_argument('--lr', default=1e-4, type=float, help='Learning rate')
    parser.add_argument('--weight_decay', default=1e-5, help='Weight Decay')
    ## dataset
    parser.add_argument('--dataset_list', nargs='+', default=['total_set'])
    # change here
    parser.add_argument('--data_root_path', default='...', help='data root path')
    parser.add_argument('--data_txt_path', default='./dataset/dataset_list/', help='data txt path')
    # default =2
    parser.add_argument('--batch_size', default= 2,  type=int, help='batch size')
    parser.add_argument('--num_workers', default=8, type=int, help='workers numebr for DataLoader')
    parser.add_argument('--a_min', default=-250, type=float, help='a_min in ScaleIntensityRanged')
    parser.add_argument('--a_max', default=250, type=float, help='a_max in ScaleIntensityRanged')
    parser.add_argument('--b_min', default=0.0, type=float, help='b_min in ScaleIntensityRanged')
    parser.add_argument('--b_max', default=1.0, type=float, help='b_max in ScaleIntensityRanged')
    parser.add_argument('--space_x', default=1.0, type=float, help='spacing in x direction')
    parser.add_argument('--space_y', default=1.0, type=float, help='spacing in y direction')
    parser.add_argument('--space_z', default=1.0, type=float, help='spacing in z direction')
    parser.add_argument('--roi_x', default=96, type=int, help='roi size in x direction')
    parser.add_argument('--roi_y', default=96, type=int, help='roi size in y direction')
    parser.add_argument('--roi_z', default=96, type=int, help='roi size in z direction')
    parser.add_argument('--num_samples', default=1, type=int, help='sample number in each ct')
    parser.add_argument('--map_type', default='prostate', help='sample number in each ct')
    parser.add_argument('--num_class', default=3, type=int, help='class num')

    parser.add_argument('--phase', default='train', help='train or validation or test')
    parser.add_argument('--uniform_sample', action="store_true", default=False, help='whether utilize uniform sample strategy')
    parser.add_argument('--datasetkey', nargs='+', default=['01', '02', '03', '04', '05', 
                                            '07', '08', '09', '12', '13', '10_03', 
                                            '10_06', '10_07', '10_08', '10_09', '10_10'],
                                            help='the content for ')
    parser.add_argument('--cache_dataset', action="store_true", default=False, help='whether use cache dataset')
    parser.add_argument('--cache_rate', default=0.005, type=float, help='The percentage of cached data in total')
    parser.add_argument('--overlap', default=0.5, type=float, help='overlap for sliding_window_inference')
    parser.add_argument('--dataset_path', default='/home/azureuser/prostate_data/', help='dataset path')
    parser.add_argument("--weight_std", default=True)
    parser.add_argument('--model_backbone', default=None, help='model backbone, also avaliable for swinunetr')
    parser.add_argument('--train_type', default='efficiency', help='either train from scratch or transfer')
    parser.add_argument('--percent', default=1081, type=int, help='pre-training using numbers of images')
    parser.add_argument('--seed', default=0, type=int, help='random seed')
    parser.add_argument('--checkpoint', default='genesis', help='pretrain checkpoint name') 
    args = parser.parse_args()
    assert args.checkpoint in ['tang', 'jose', 'univ_swin', 'voco','sup_swin', 'genesis', 'unimiss_tiny', 'unimiss_small', 'med3d', 'dodnet', 'univ_unet', 'sup_unet', 'sup_seg',]
    pre_dict={'tang':'self_supervised_nv_swin_unetr_5050.pt',
              "jose":'self_supervised_nv_swin_unetr_50000.pth',
              "univ_swin":'supervised_clip_driven_universal_swin_unetr_2100.pth',
             'sup_swin':'supervised_suprem_swinunetr_2100.pth',
             'genesis':'self_supervised_models_genesis_unet_620.pt', 
             "unimiss_tiny":'self_supervised_unimiss_nnunet_tiny_5022.pth',
             "unimiss_small":'self_supervised_unimiss_nnunet_small_5022.pth',
             "med3d":'supervised_med3D_residual_unet_1623.pth',
             "dodnet":'supervised_dodnet_unet_920.pth',
             "univ_unet":'supervised_clip_driven_universal_unet_2100.pth',
             "sup_unet":'supervised_suprem_unet_2100.pth',
             "sup_seg":'supervised_suprem_segresnet_2100.pth',
             "voco":"VoCo_10k.pt",
             }
    back_dict= {
    'tang': 'swinunetr',
    'jose': 'swinunetr',
    'univ_swin': 'swinunetr',
    'sup_swin': 'swinunetr',
    'voco': 'swinunetr',
    'genesis': 'unet',
    'unimiss_tiny': 'unet',
    'unimiss_small': 'unet',
    'med3d': 'unet',
    'dodnet': 'unet',
    'univ_unet': 'unet',
    'sup_unet': 'unet',
    'sup_seg': 'segresnet'
    }

    if args.model_backbone is  None:
        args.model_backbone = back_dict[args.checkpoint]
    process(args=args)


if __name__ == "__main__":
    main()



    
