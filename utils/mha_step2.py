# convert pengwin to nii
import SimpleITK as sitk
import nibabel as nib
import os
import argparse
def main(args):
# 加载MHA文件
    image = sitk.ReadImage('/home/azureuser/pg/001.mha')
    # nii=nib.load('/home/azureuser/output_filename.nii.gz')
    # 保存为NII.GZ格式
    sitk.WriteImage(image, 'output_filename.nii.gz')
    # assert image.GetDimension() == 3
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--abdomen_atlas', dest='abdomen_atlas', type=str, default='/home/fanlinghuang/TAD-chenyujie/Task05_Prostate/imagesTs',
                        help='the directory of the AbdomenAtlas dataset',
                       )
    parser.add_argument("--png_save_path", dest='png_save_path', type=str, default='./materials',
                        help='the directory of png for each CT slice',
                       )
    parser.add_argument("--video_save_path", dest='video_save_path', type=str, default='./videos',
                        help='the directory for saving videos',
                       )
    args = parser.parse_args()

    if not os.path.exists(args.png_save_path):
        os.makedirs(args.png_save_path)

    if not os.path.exists(args.video_save_path):
        os.makedirs(args.video_save_path)

    if not os.path.exists(args.gif_save_path):
        os.makedirs(args.gif_save_path)
    
    main(args)