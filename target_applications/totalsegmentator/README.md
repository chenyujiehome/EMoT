<h1 align="center">SuPreM</h1>
<h3 align="center" style="font-size: 20px; margin-bottom: 4px">For Anatomical Segmentation on the TotalSegmentator</h3>
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
The results are organized as

```
totalsegmentator
    ├── checkpoints
    │   └── $logname
    │       ├── best_model.pth
    │       └── model.pth
    └── out
        └── $logname
            ├── TensorBoardLogs
            ├── nsd_validation_results.csv
            └── dice_validation_results.csv
```

##### 0. Create a virtual environment (optional)

```bash
conda create -n suprem python=3.8
source activate suprem
```

##### 1. Clone the GitHub repository

```bash
git clone https://github.com/chenyujiehome/SuPreM.git
cd SuPreM/
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
pip install monai[all]==0.9.0
pip install -r requirements.txt
```

##### 2. Download the pre-trained  checkpoint

```bash
cd target_applications/totalsegmentator/pretrained_weights/
wget https://huggingface.co/MrGiovanni/SuPreM/resolve/main/supervised_suprem_unet_2100.pth
wget https://huggingface.co/MrGiovanni/SuPreM/resolve/main/supervised_suprem_segresnet_2100.pth

cd ../../../
```

##### 3. Download the TotalSegmentator (v2.0.1) dataset

from [Zenodo](https://doi.org/10.5281/zenodo.6802613) (1,228 subjects) (v2.0.1) and save it to `/path/to/your/data/TotalSegmentator`

##### 4. Fine-tune pretraining methods (U-Net and SegResNet) on TotalSegmentator 
```bash
# Single GPU

cd target_applications/totalsegmentator/
RANDOM_PORT=$((RANDOM % 64512 + 1024))
pretraining_method_name=suprem
datapath=/path/to/your/data/TotalSegmentator/ # change to /path/to/your/data/TotalSegmentator
target_task=cardiac
num_target_class=19
for arch in unet segresnet; do
    if [ "$arch" = "unet" ]; then
    case $pretraining_method_name in 
    suprem)
        pretrain_path="pretrained_weights/supervised_suprem_unet_2100.pth"
        ;;
    genesis)
        pretrain_path="pretrained_weights/self_supervised_models_genesis_unet_620.pt"
        ;;
    med3d)
        pretrain_path="pretrained_weights/supervised_med3D_residual_unet_1623.pth"
        ;;
    scratch)
        echo "from scratch"
        ;;
    *)
        echo "unkown: $pretraining_method_name"
        ;;
    esac

    elif [ "$arch" = "segresnet" ]; then
    case $pretraining_method_name in 
    suprem)
        pretrain_path="pretrained_weights/supervised_suprem_segresnet_2100.pth"
        ;;
    scratch)
        echo "from scratch"
        ;;
    *)
        echo "unkown: $pretraining_method_name"
        ;;
    esac
        
    else
        echo "unkown : $arch"
    fi

    for fold in {1..5}; do
        RANDOM_PORT=$((RANDOM % 64512 + 1024))
        python -W ignore -m torch.distributed.launch --nproc_per_node=1 --master_port=$RANDOM_PORT train.py --dist  --model_backbone $arch --log_name $pretraining_method_name.$arch.$target_task.fold$fold --map_type $target_task --num_class $num_target_class --dataset_path $datapath --num_workers 8 --batch_size 2 --pretrain $pretrain_path --fold $fold --pretraining_method_name $pretraining_method_name
    done
done
```

##### 5. Evaluate the performance per class of pretraining methods

```bash
# Single GPU

cd target_applications/totalsegmentator/
RANDOM_PORT=$((RANDOM % 64512 + 1024))
pretraining_method_name=suprem
for arch in unet segresnet; do
for fold in {1..5}
do
datapath=/path/to/your/data/TotalSegmentator/ # change to /path/to/your/data/TotalSegmentator
target_task=cardiac
num_target_class=19
checkpoint_path=checkpoints/$pretraining_method_name.$arch.$target_task.fold$fold/best_model.pth



python -W ignore -m torch.distributed.launch --nproc_per_node=1 --master_port=$RANDOM_PORT test.py --dist  --model_backbone $arch --log_name $pretraining_method_name.$arch.$target_task.fold$fold --map_type $target_task --num_class $num_target_class --dataset_path $datapath --num_workers 8 --batch_size 2 --pretrain $checkpoint_path  --fold $fold --pretraining_method_name $pretraining_method_name
done
done
```

##### 6. Fine-tune the from-scratch models (U-Net and SegResNet) using TotalSegmentator

```bash
# Single GPU

cd target_applications/totalsegmentator/
RANDOM_PORT=$((RANDOM % 64512 + 1024))
pretraining_method_name=scratch
datapath=/path/to/your/data/TotalSegmentator/ # change to /path/to/your/data/TotalSegmentator
for arch in unet segresnet; do
target_task=cardiac
num_target_class=19
for fold in {1..5}
do
python -W ignore -m torch.distributed.launch --nproc_per_node=1 --master_port=$RANDOM_PORT train.py --dist  --model_backbone $arch --log_name $pretraining_method_name.$arch.$target_task.fold$fold --map_type $target_task --num_class $num_target_class --dataset_path $datapath --num_workers 8 --batch_size 2  --fold $fold --pretraining_method_name $pretraining_method_name
done
done
```

##### 7. Evaluate the per-class performance of the model trained from scratch

```bash
# Single GPU

cd target_applications/totalsegmentator/
RANDOM_PORT=$((RANDOM % 64512 + 1024))
pretraining_method_name=scratch
for arch in unet segresnet; do
for fold in {1..5}
do
datapath=/path/to/your/data/TotalSegmentator/ # change to /path/to/your/data/TotalSegmentator
target_task=cardiac
num_target_class=19
checkpoint_path=checkpoints/$pretraining_method_name.$arch.$target_task.fold$fold/best_model.pth



python -W ignore -m torch.distributed.launch --nproc_per_node=1 --master_port=$RANDOM_PORT test.py --dist  --model_backbone $arch --log_name $pretraining_method_name.$arch.$target_task.fold$fold --map_type $target_task --num_class $num_target_class --dataset_path $datapath --num_workers 8 --batch_size 2 --pretrain $checkpoint_path  --fold $fold --pretraining_method_name $pretraining_method_name
done
done
```


