<h1 align="center">SuPreM</h1>
<h3 align="center" style="font-size: 20px; margin-bottom: 4px">Benchmark your own backbone on AbdomenAtlas 1.0</h3>
<p align="center">
    <a href='https://www.zongweiz.com/dataset'><img src='https://img.shields.io/badge/Project-Page-Green'></a> 
    <a href='https://www.cs.jhu.edu/~alanlab/Pubs23/li2023suprem.pdf'><img src='https://img.shields.io/badge/Paper-PDF-purple'></a> 
    <a href='document/promotion_slides.pdf'><img src='https://img.shields.io/badge/Slides-PDF-orange'></a> 
    <a href='document/dom_wse_poster.pdf'><img src='https://img.shields.io/badge/Poster-PDF-blue'></a> 
    <a href='https://www.cs.jhu.edu/news/ai-and-radiologists-unite-to-map-the-abdomen/'><img src='https://img.shields.io/badge/WSE-News-yellow'></a>
    <br/>
    <a href="https://github.com/MrGiovanni/SuPreM"><img src="https://img.shields.io/github/stars/MrGiovanni/SuPreM?style=social" /></a>
    <a href="https://twitter.com/bodymaps317"><img src="https://img.shields.io/twitter/follow/BodyMaps" alt="Follow on Twitter" /></a>
</p>

### 0. Create a virtual environment (optional)

```bash
conda create -n suprem python=3.8
source activate suprem
```

### 1. Clone the GitHub repository

```bash
git clone https://github.com/MrGiovanni/SuPreM
cd SuPreM/benchmark_backbones/
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
pip install monai[all]==0.9.0
pip install -r requirements.txt
```

### 2. Download the AbdomenAtlas 1.0 dataset

Please reach out to Zongwei Zhou (`zzhou82@jh.edu`) for a beta-test of this dataset.

### 3. Train a U-Net (as an example) on AbdomenAtlas 1.0

```bash
RANDOM_PORT=$((RANDOM % 64512 + 1024))
backbone=unet
datapath=/scratch/zzhou82/data/AbdomenAtlasMini1.0 # need modification

# Single GPU
# A batch size of 2 requires 6G GPU memory (U-Net)
python -W ignore -m torch.distributed.launch --nproc_per_node=1 --master_port=$RANDOM_PORT train.py --dist  --data_root_path $datapath --num_workers 12 --log_name AbdomenAtlas1.0.$backbone --backbone $backbone --lr 1e-4 --warmup_epoch 20 --batch_size 2 --max_epoch 800 --cache_dataset

# Multi GPU (e.g., 4)
python -W ignore -m torch.distributed.launch --nproc_per_node=4 --master_port=$RANDOM_PORT train.py --dist  --data_root_path $datapath --num_workers 12 --log_name AbdomenAtlas1.0.$backbone --backbone $backbone --lr 1e-4 --warmup_epoch 20 --batch_size 2 --max_epoch 800 --cache_dataset
```

### 4. Change to your own backbone

##### 4.1 Create a Python file of your backbone in the folder `model/`

Here we provided two examples, U-Net (`model/unet3d.py`) and Swin UNETR (`model/SwinUNETR.py`), for your reference.

##### 4.2 Import your backbone at the beginning of `train.py`

For example,
```python
from model.unet3d import UNet
from model.SwinUNETR import SwinUNETR # add this line
```

##### 4.3 Add an IF branch to set up your backbone in the `train.py`

For example, in Line [72](https://github.com/MrGiovanni/SuPreM/blob/fef9725c86d39caed2d79bcca0a69683c847b00f/benchmark_backbones/train.py#L72)
```python
# add this IF branch
if args.backbone == 'swinunetr':
        model = SwinUNETR(
                          img_size=(args.roi_x, args.roi_y, args.roi_z),
                          in_channels=1,
                          out_channels=args.num_class,
                          feature_size=48,
                          drop_rate=0.0,
                          attn_drop_rate=0.0,
                          dropout_path_rate=0.0,
                          use_checkpoint=False,
                         )
```

##### 4.4 Modify the `--backbone` argument when running the code

For example,
```bash
backbone=swinunetr # modify here
```
