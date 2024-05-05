# convert pengwin to nii
import SimpleITK as sitk
import nibabel as nib

# 加载MHA文件
image = sitk.ReadImage('/home/azureuser/pg/001.mha')
# nii=nib.load('/home/azureuser/output_filename.nii.gz')
# 保存为NII.GZ格式
sitk.WriteImage(image, 'output_filename.nii.gz')
# assert image.GetDimension() == 3
