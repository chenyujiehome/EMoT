'''
python -W ignore plot_video_multiprocessing.py --abdomen_atlas /Users/zongwei.zhou/Dropbox\ \(ASU\)/PublicResource/SuPreM/AbdomenAtlas/AbdomenAtlas1.0 --png_save_path /Users/zongwei.zhou/Desktop/AbdomenAtlas1.0PNG --video_save_path /Users/zongwei.zhou/Desktop/AbdomenAtlas1.0AVI --gif_save_path /Users/zongwei.zhou/Desktop/AbdomenAtlas1.0GIF
'''

import numpy as np 
import os 
import cv2
import argparse
import nibabel as nib 
from tqdm import tqdm 
from PIL import Image
import imageio
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count
import torch
import os
import shutil
import tempfile
import time
import matplotlib.pyplot as plt
from monai.apps import DecathlonDataset
from monai.config import print_config
from monai.data import DataLoader, decollate_batch,DistributedSampler, pad_list_data_collate,list_data_collate
from torch.utils.data import random_split
from monai.handlers.utils import from_engine
from monai.losses import DiceLoss
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric
from monai.networks.nets import SegResNet
from monai.transforms import (
    Activations,
    Activationsd,
    AsDiscrete,
    AsDiscreted,
    Compose,
    Invertd,
    LoadImaged,
    MapTransform,
    NormalizeIntensityd,
    Orientationd,
    RandFlipd,
    RandScaleIntensityd,
    RandShiftIntensityd,
    RandSpatialCropd,
    Spacingd,
    EnsureTyped,
    EnsureChannelFirstd,
    SpatialPadd,
)
from monai.utils import set_determinism

import torch
class ScaleIntensity(MapTransform):

    def __init__(
        self,
        keys ,
        allow_missing_keys: bool = True,
        channel_wise: bool = True,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.channel_wise = channel_wise

    def scaler(self, img):
        if self.channel_wise:
            # Assume the channel dimension is the first dimension
            for i in range(img.shape[0]):
                if torch.max(img[i]) != 0:
                    img[i] = img[i] / torch.max(img[i])
        else:
            if torch.max(img) != 0:
                img = img / torch.max(img)
        return img

    def __call__(self, data) :
        d = dict(data)
        for key in self.key_iterator(d):
            d[key] = self.scaler(d[key])
        return d
class NameData(MapTransform):


    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            d['name'] = os.path.splitext(os.path.splitext(os.path.basename(d[key]))[0])[0]
        return d
class ConvertToMultiChannelBasedOnBratsClassesd(MapTransform):
    """
    Convert labels to multi channels based on brats classes:
    label 1 is the peritumoral edema
    label 2 is the GD-enhancing tumor
    label 3 is the necrotic and non-enhancing tumor core
    The possible classes are TC (Tumor core), WT (Whole tumor)
    and ET (Enhancing tumor).

    """

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            d[key] = d[key].unsqueeze(0)
        return d
low_range = 0
high_range = 250
# 除以max 乘以255
class_of_interest = ['PZ',
                     'TZ',
                    ]

CLASS_IND = {
    'PZ': 1,
    'TZ': 2,
}

def add_colorful_mask(image, mask, class_index):
    
    image[mask == class_index['PZ'], 0] = 255   # spleen (255,0,0) 
    image[mask == class_index['TZ'], 1] = 255   # spleen (255,0,0) 
    
    
    # image[mask == class_index['kidney_right'], 1] = 255   # kidney_right (0,255,0)

    # image[mask == class_index['liver'], 2] = 255   # liver (255,0,255)
  
    
    return image

def load_individual_maps(segmentation_dir):
    
    mask_path = args.mask_path
    c_mask = torch.load(mask_path)[segmentation_dir[:11]].cpu().clone().numpy()
    mask=np.zeros(c_mask.shape[1:],dtype=np.uint8)
    for c in class_of_interest:
        mask[c_mask[CLASS_IND[c],:,:,:] == 1] = CLASS_IND[c]
    
    return mask
    
def full_make_png(case_name, args,data):
    
    for plane in ['axial', 'coronal', 'sagittal']:
        if not os.path.exists(os.path.join(args.png_save_path, plane, case_name)):
            os.makedirs(os.path.join(args.png_save_path, plane, case_name))


    image_path = os.path.join(args.abdomen_atlas, case_name)

    # single case
    image =data[case_name[:11]][args.image_channel,:,:,:]

    mask = load_individual_maps( case_name)
    
    high_range = torch.max(image)
    image[image > high_range] = high_range
    image[image < low_range] = low_range
    image = np.round((image / high_range)  * 255.0).astype(np.uint8)
    image = np.repeat(image.reshape(image.shape[0],image.shape[1],image.shape[2],1), 3, axis=3)
    
    image_mask = add_colorful_mask(image, mask, CLASS_IND)
    image_mask= np.rot90(image_mask, k=1, axes=(0, 1))
    for z in range(mask.shape[2]):
        Image.fromarray(image_mask[:,:,z,:]).save(os.path.join(args.png_save_path, 'axial', case_name, str(z)+'.png'))

    for z in range(mask.shape[1]):
        Image.fromarray(image_mask[:,z,:,:]).save(os.path.join(args.png_save_path, 'sagittal', case_name, str(z)+'.png'))

    for z in range(mask.shape[0]):
        Image.fromarray(image_mask[z,:,:,:]).save(os.path.join(args.png_save_path, 'coronal', case_name, str(z)+'.png'))
        
def make_avi(case_name, plane, args,data):

    if not os.path.exists(os.path.join(args.video_save_path, plane)):
        os.makedirs(os.path.join(args.video_save_path, plane))
    if not os.path.exists(os.path.join(args.gif_save_path, plane)):
        os.makedirs(os.path.join(args.gif_save_path, plane))
    
    image_folder = os.path.join(args.png_save_path, plane, case_name)
    video_name = os.path.join(args.video_save_path, plane, case_name+'.avi')
    gif_name = os.path.join(args.gif_save_path, plane, case_name+'.gif')
    
    images = [img for img in os.listdir(image_folder) if img.endswith(".png")]

    for i in range(len(images)):
        images[i] = images[i].replace('.png','')
        images[i] = int(images[i])
    images.sort()

    frame = cv2.imread(os.path.join(image_folder, str(images[0])+'.png'))
    height, width, layers = frame.shape

    video = cv2.VideoWriter(video_name, 0, args.FPS, (width,height))

    imgs = []
    for image in images:
        img = cv2.imread(os.path.join(image_folder, str(image)+'.png'))
        video.write(img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        imgs.append(img)

    cv2.destroyAllWindows()
    video.release()
    imageio.mimsave(gif_name, imgs, duration=args.FPS*0.4)
    
def event(folder, args):

    test_transform = Compose(
    [
        NameData(keys=["image"]),
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys="image"),
        EnsureTyped(keys=["image"]),
        # ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
        Orientationd(keys=["image"], axcodes="RAS"),
        Spacingd(
            keys=["image"],
            pixdim=(1.0,1.0,1.0),
            mode=("bilinear"),
        ),
        ScaleIntensity(keys=["image"],channel_wise=True),
    ]
)
    test_dataset = DecathlonDataset(
            root_dir="/home/azureuser/prostate_data/",
            task="Task05_Prostate",
            section="test",
            transform=test_transform,
            download=False,
            cache_rate=0.0,
            num_workers=4,
        )
    test_dict={}
    for i in range(len(test_dataset)):
        test_dict[test_dataset[i]['name']]=test_dataset[i]['image']
    test_dataset=test_dict
    full_make_png(folder, args,test_dataset)
    for plane in ['axial', 'coronal', 'sagittal']:
        make_avi(folder, plane, args,test_dataset)
        
def main(args):
    folder_names = [name for name in os.listdir(args.abdomen_atlas)]
    print('>> {} CPU cores are secured.'.format(cpu_count()))
    for folder in folder_names:
        event(folder, args)

        
                
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--abdomen_atlas', dest='abdomen_atlas', type=str, default='/home/azureuser/prostate_data/Task05_Prostate/imagesTs',
                        help='the directory of the AbdomenAtlas dataset',
                       )
    parser.add_argument("--png_save_path", dest='png_save_path', type=str, default='./materials',
                        help='the directory of png for each CT slice',
                       )
    parser.add_argument("--video_save_path", dest='video_save_path', type=str, default='./videos',
                        help='the directory for saving videos',
                       )
    parser.add_argument("--gif_save_path", dest='gif_save_path', type=str, default='./gifs',
                        help='the directory for saving gifs',
                       )
    parser.add_argument("--FPS", dest='FPS', type=float, default=20,
                        help='the FPS value for videos',
                       )
    parser.add_argument("--image_channel", type=int, default=0,
                       )
    parser.add_argument("--mask_path",  type=str, default="/home/azureuser/SuPreM/target_applications/totalsegmentator/save_tensor_genesis_unet.pt",
                       )
    args = parser.parse_args()

    if not os.path.exists(args.png_save_path):
        os.makedirs(args.png_save_path)

    if not os.path.exists(args.video_save_path):
        os.makedirs(args.video_save_path)

    if not os.path.exists(args.gif_save_path):
        os.makedirs(args.gif_save_path)
    
    main(args)