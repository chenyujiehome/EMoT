import os
import requests
from train import pre_dict
import argparse
# 指定下载的 URL 和文件路径
def main(args):
    file_name=pre_dict[args.checkpoint]
    url = f"https://huggingface.co/jethro682/pretrain_mri/resolve/main/{file_name}?download=true"
    file_path = "pretrained_weights/"+file_name  # 替换为你的目标文件夹路径

    # 检查文件是否已经存在
    if not os.path.exists(file_path):
        # 文件不存在，开始下载
        response = requests.get(url, allow_redirects=True)
        response.raise_for_status()  # 确保请求成功

        # 将文件写入到指定的路径
        with open(file_path, "wb") as f:
            f.write(response.content)
        print(f"File downloaded to {file_path}")
    else:
        print(f"File {file_path} already exists. No download needed.")
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', default='tang', help='The path resume from checkpoint')
    args = parser.parse_args()
    main(args)