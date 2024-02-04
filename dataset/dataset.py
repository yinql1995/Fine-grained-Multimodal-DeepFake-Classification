from PIL import Image
from torch.nn.functional import batch_norm
from torch.utils.data import Dataset
import torch
from torchvision import transforms as T
import os
import torchvision.transforms.functional as TF
import random
import numpy as np
from tqdm import tqdm
import albumentations as A
import cv2
import shutil
from albumentations import (
    HorizontalFlip, Perspective, ShiftScaleRotate, CLAHE, RandomRotate90,
    Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,
    MotionBlur, MedianBlur,
    Sharpen, Emboss, RandomBrightnessContrast, Flip, OneOf, Compose, ReplayCompose
)
import soundfile as sf
from torch import Tensor
import dataset_collate
# import torch.multiprocessing
#
# torch.multiprocessing.set_sharing_strategy('file_system')


trans = {300:T.Compose([T.Resize(300), T.ToTensor()]),
         128:T.Compose([T.Resize((128, 128)), T.ToTensor()]),
         299:T.Compose([T.ToTensor(), T.Resize(299)]),
         256:T.Compose([T.ToTensor(), T.Resize((256, 256))]),
         224:T.Compose([T.ToTensor(), T.Resize((224, 224))]),
         192:T.Compose([T.ToTensor(),T.Resize((192, 192))])}

class Multimodal_dataset(Dataset):
    def __init__(self, image_size, type, txt_path, num_frame):
        assert image_size in trans.keys()
        fh = open(txt_path, 'r')
        img_aud = []
        for line in fh:
            # print(line)
            line = line.rstrip()
            img_path = line[:-4]

            temp_path = os.path.split(img_path)[0]

            aud_path, _ = os.path.split(temp_path)
            aud_path = os.path.join(aud_path, '16k_'+_+'.wav')
            # print(img_path)
            # print(aud_path)
            # print(int(line[-3]))
            img_aud.append((img_path, aud_path, int(line[-3]), int(line[-1])))
        self.img_aud = img_aud          # imgs[0:len(imgs):10]
        self.trans = trans[image_size]
        self.image_size = image_size
        self.num_frame = num_frame
        self.phase = type
        # self.slice_index = np.arange(0, 10, 1)
        # random.shuffle(self.slice_index)

    def __getitem__(self, index):
        # img_data = torch.zeros((self.num_frame*10, 3, self.image_size, self.image_size))
        img_data = []
        fn_img, fn_aud, label, start = self.img_aud[index]

        temp = os.path.split(fn_img)[0]
        filename = temp
        index = int(os.path.split(fn_img)[1][:-4])

        temp1 = ''
        # pp = True

        ## 随机每10帧内取4帧
        slice_index = np.arange(0, 10, 1)
        random.shuffle(slice_index)
        slice_index = slice_index[:4]
        slice_index.sort()
        slice_index = slice_index.repeat(10)
        slice_index = slice_index.reshape(4, 10).transpose(1, 0)
        a = np.arange(0, 100, 10).reshape(10, 1)
        slice_index = slice_index + a
        # print(slice_index)
        slice_index = slice_index.reshape(-1)

        for i in slice_index:

            fn = temp + '/' + str(index + i) + '.png'

            if not os.path.exists(fn):
                fn = temp1
            try:
                img = Image.open(fn).convert('RGB')

                # if self.phase == 'train':
                #     # aug = strong_aug2(0.5)
                #     img = np.array(img)
                #     # if not pp:
                #     #     pp = aug(image=img)
                #     #     img = A.ReplayCompose.replay(pp['replay'], image=img)['image']
                #     img = Image.fromarray(img)
                # if self.phase == 'test':
                #     img = np.array(img)
                #     img = gaussian_blur(img, 9)
                #     img = Image.fromarray(img)

                img = self.trans(img)
                img_data.append(img.unsqueeze(0))
                temp1 = fn
            except:
                print('........................')
                print(filename)

        img_data = torch.cat(img_data, dim=0)
        img_data = img_data.view(-1, self.image_size, self.image_size)
        # print(img_data.size())
        aud_data, _ = sf.read(fn_aud, start=start*16000, stop=(start+4)*16000)
        aud_data = Tensor(aud_data)

        # print(aud_data.size(0))

        if label == 0:
            video_label = 0
            audio_label = 0
            total_label = 0
        elif label == 1:
            video_label = 1
            audio_label = 1
            total_label = 1
        elif label == 2 :
            video_label = 1
            audio_label = 0
            total_label = 2
        else:
            video_label = 0
            audio_label = 1
            total_label = 3
        # print(filename)
        return filename, img_data, aud_data, total_label, video_label, audio_label

    def __len__(self):
        return len(self.img_aud)

