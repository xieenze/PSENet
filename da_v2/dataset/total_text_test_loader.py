# dataloader add 3.0 scale
# dataloader add filer text
import sys
sys.path.append('../')
import numpy as np
from PIL import Image
from torch.utils import data
import util
import cv2
import random
import torchvision.transforms as transforms
import torch
import pyclipper
import Polygon as plg
_debug = 0
if _debug == 0:
    total_text_root_dir = './data/total_text/'
    total_text_test_data_dir = total_text_root_dir + 'totaltext/Images/Test/'
else:
    total_text_root_dir = '../data/total_text/'
    total_text_test_data_dir = total_text_root_dir + 'totaltext/Images/Test/'

random.seed(123456)

def get_img(img_path):
    try:
        img = cv2.imread(img_path)
        img = img[:, :, [2, 1, 0]]
    except Exception as e:
        print img_path
        raise
    return img

def scale(img, short_size=512):
    h, w = img.shape[0:2]
    scale = short_size * 1.0 / min(h, w)
    # img = cv2.resize(img, dsize=None, fx=scale, fy=scale)
    h = (int)(h * scale + 0.5)
    w = (int)(w * scale + 0.5)
    if h % 32 != 0:
        h = h + (32 - h % 32)
    if w % 32 != 0:
        w = w + (32 - w % 32)
    img = cv2.resize(img, dsize=(w, h))
    return img


class TOTALTEXTTestloader(data.Dataset):
    def __init__(self, short_size=512):
        data_dirs = [total_text_test_data_dir]
        
        self.img_paths = []
        
        for data_dir in data_dirs:
            img_names = util.io.ls(data_dir, '.jpg')
            img_names.extend(util.io.ls(data_dir, '.png'))
            # img_names.extend(util.io.ls(data_dir, '.gif'))

            img_paths = []
            for idx, img_name in enumerate(img_names):
                img_path = data_dir + img_name
                img_paths.append(img_path)

            self.img_paths.extend(img_paths)
        self.short_size = short_size

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        img_path = self.img_paths[index]

        img = get_img(img_path)
        
        scaled_img = scale(img, self.short_size)
        scaled_img = Image.fromarray(scaled_img)
        scaled_img = scaled_img.convert('RGB')
        scaled_img = transforms.ToTensor()(scaled_img)
        scaled_img = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(scaled_img)

        return img, scaled_img

if __name__ == "__main__":
    from IPython import embed
    loader = TOTALTEXTTestloader()
    embed()
