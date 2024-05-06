import SimpleITK as sitk
import os
import argparse
from concurrent.futures import ThreadPoolExecutor

def process_file(filename, input_dir, output_dir):
    # Read the .mha file
    image = sitk.ReadImage(os.path.join(input_dir, filename))
    
    # Convert the .mha file to .nii.gz format
    nii_image = sitk.GetImageFromArray(sitk.GetArrayFromImage(image))
    
    # Create a new directory named "s{n}"
    new_dir = os.path.join(output_dir, 's' + filename[:-4])
    os.makedirs(new_dir, exist_ok=True)
    
    # Save the .nii.gz file in the new directory
    sitk.WriteImage(nii_image, os.path.join(new_dir, 'ct.nii.gz'))
    
    # Create a "segmentation" subdirectory in the new directory
    seg_dir = os.path.join(new_dir, 'segmentation')
    os.makedirs(seg_dir, exist_ok=True)
    
    # Read the corresponding .mha file from the "segmentation" directory
    seg_image = sitk.ReadImage(os.path.join(input_dir, 'segmentation', filename))
    
    # Convert the .mha file to .nii.gz format
    seg_nii_image = sitk.GetImageFromArray(sitk.GetArrayFromImage(seg_image))
    
    # Save the .nii.gz file in the "segmentation" subdirectory
    sitk.WriteImage(seg_nii_image, os.path.join(seg_dir, 'label.nii.gz'))

def main(args):
    # Using ThreadPoolExecutor to process files in parallel
    with ThreadPoolExecutor() as executor:
        for filename in os.listdir(args.input):
            if filename.endswith('.mha'):
                executor.submit(process_file, filename, args.input, args.output)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='/home/azureuser/pg',
                        help='the directory of the AbdomenAtlas dataset')
    parser.add_argument("--output", type=str, default='/mnt/nii',
                        help='the directory of png for each CT slice')
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        os.makedirs(args.input)

    if not os.path.exists(args.output):
        os.makedirs(args.output)
    
    main(args)
