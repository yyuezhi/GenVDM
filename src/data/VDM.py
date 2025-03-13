#Copyright 2024-2025 Adobe Inc.

import os
import json
import numpy as np
import webdataset as wds
import pytorch_lightning as pl
import torch
from torch.utils.data import Dataset
from torch.utils.data.distributed import DistributedSampler
from PIL import Image
from pathlib import Path
import torchvision.transforms.functional as F
from src.utils.train_util import instantiate_from_config
import torchvision.transforms as transforms
import random
from einops import rearrange
from torchvision.transforms import v2

def change_hue(tensor, alpha,hue_factor,brightness_factor,saturation_factor):
    """Change the hue of the image tensor."""
    # Convert to PIL image, apply hue adjustment, and convert back to tensor
    tensor_list = []
    alpha = alpha.unsqueeze(1).repeat(1, 3, 1, 1)
    bg_color = tensor[alpha == 0]
    for i in range(tensor.size(0)):
        pil_image = F.to_pil_image(tensor[i])
        if i == 2:
            pil_image = F.adjust_hue(pil_image, hue_factor)
            pil_image = F.adjust_brightness(pil_image, brightness_factor)
            #pil_image = F.adjust_saturation(pil_image, saturation_factor)
        tensor_list.append(F.to_tensor(pil_image))
    tensor = torch.stack(tensor_list)
    tensor[alpha == 0] = bg_color
    return tensor

# Define the transformation pipeline
class CustomTransform:
    def __init__(self):
        pass

    def __call__(self, tensor,alpha,crop_size, hue_factor,brightness_factor,saturation_factor,hue_flag = False,crop_flag = False, flip_flag = False):
        # Change hue
        if hue_flag:
            tensor = change_hue(tensor, alpha,hue_factor,brightness_factor,saturation_factor)
        # Random crop
        if crop_flag:
            tensor = transforms.CenterCrop(crop_size)(tensor)
        # Vertical flip

        if random.random() > 0.5 and flip_flag:
            tensor = F.vflip(tensor)
            tmp = tensor[5].detach().clone()
            tensor[5] = tensor[6]
            tensor[6] = tmp

        if random.random() > 0.5 and flip_flag:
            tensor = F.hflip(tensor)
            tmp = tensor[1].detach().clone()
            tensor[1] = tensor[3]
            tensor[3] = tmp
            tmp = tensor[4].detach().clone()
            tensor[4] = tensor[0]
            tensor[0] = tmp
        return tensor

class DataModuleFromConfig(pl.LightningDataModule):
    def __init__(
        self, 
        batch_size=8, 
        num_workers=4, 
        train=None, 
        validation=None, 
        test=None, 
        **kwargs,
    ):
        super().__init__()

        self.batch_size = batch_size
        self.num_workers = num_workers

        self.dataset_configs = dict()
        if train is not None:
            self.dataset_configs['train'] = train
        if validation is not None:
            self.dataset_configs['validation'] = validation
        if test is not None:
            self.dataset_configs['test'] = test
    
    def setup(self, stage):

        if stage in ['fit']:
            self.datasets = dict((k, instantiate_from_config(self.dataset_configs[k])) for k in self.dataset_configs)
        else:
            raise NotImplementedError

    def train_dataloader(self):

        sampler = DistributedSampler(self.datasets['train'])
        return wds.WebLoader(self.datasets['train'], batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False, sampler=sampler)

    def val_dataloader(self):

        sampler = DistributedSampler(self.datasets['validation'])
        return wds.WebLoader(self.datasets['validation'], batch_size=4, num_workers=self.num_workers, shuffle=False, sampler=sampler)

    def test_dataloader(self):

        return wds.WebLoader(self.datasets['test'], batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)




class VDMData(Dataset):
    def __init__(self,
        image_dir="/mnt/localssd/render_ortho_normal",
        validation=False,
        hue_flag = False,
        crop_flag = True, 
        flip_flag = True
    ):
        self.color_image_dir = image_dir.replace("normal","color")
        self.image_data_base_path = image_dir#"/mnt/localssd/render_new_vertical_image"
        image_list = list(set([i[:-6] for i in os.listdir(f"{self.image_data_base_path}") if ".png" in i]))
        for k in range(7):
            image_list = [i for i in image_list if os.path.exists(f"{self.color_image_dir}/{i}_{k}.png")]
        for k in range(7):
            image_list = [i for i in image_list if os.path.exists(f"{self.image_data_base_path}/{i}_{k}.png")]
        image_list = [i for i in image_list if os.path.exists(f"{self.image_data_base_path}/{i}_6.png")] * 10
        import random
        random.shuffle(image_list)
        self.paths = image_list
        self.paths.sort()
        if validation:
            self.paths = self.paths[-16:] # used last 16 as validation
        else:
            self.paths = self.paths[:-16]
        self.transform = CustomTransform()
        print('============= length of dataset %d =============' % len(self.paths))
        self.hue_flag = hue_flag
        self.crop_flag = crop_flag
        self.flip_flag = flip_flag
        print("hue_flag",self.hue_flag)
        print("crop_flag",self.crop_flag)
        print("flip_flag",self.flip_flag)
        print(self.color_image_dir,self.image_data_base_path)


    def __len__(self):
        return len(self.paths)

    def load_im(self, path, color):
        pil_img = Image.open(path)

        image = np.asarray(pil_img, dtype=np.float32) / 255.
        alpha = image[:, :, 3:]
        image = image[:, :, :3] * alpha + color * (1 - alpha)

        image = torch.from_numpy(image).permute(2, 0, 1).contiguous().float()
        alpha = torch.from_numpy(alpha).permute(2, 0, 1).contiguous().float()
        return image, alpha
    
    def get_crop_size(self, alpha):
        batch_size = alpha.shape[0]
        width = alpha.shape[1]
        height = alpha.shape[2]
        center_y, center_x = height / 2, width / 2

        # Create a meshgrid of coordinates
        y_coords, x_coords = torch.meshgrid(torch.arange(height), torch.arange(width), indexing='ij')


        # Calculate distances from the center for each pixel
        distances = torch.max(torch.stack([torch.abs(x_coords - center_x),torch.abs(y_coords - center_y)]),dim=0)[0]
        non_transparent_mask = alpha > 0
        non_transparent_distances = distances * non_transparent_mask
        max_distances = non_transparent_distances.max()
        return max_distances*2
            
    def __getitem__(self, index):
        # print("getting item",index)
        while True:
            image_basename = self.paths[index]#os.path.join(self.root_dir, self.image_dir, self.paths[index])

            '''background color, default: white'''
            bkg_color = [1., 1., 1.]

            img_list = []
            alpha_list = []
            try:
                for idx in range(7):
                    if idx != 2:
                        img, alpha = self.load_im(f'{self.image_data_base_path}/{image_basename}_{idx}.png', bkg_color)
                    else:
                        k = random.randint(0, 2) 
                        img, alpha = self.load_im(f'{self.color_image_dir}/{image_basename}_{k}.png', bkg_color)
                    img_list.append(img)
                    alpha_list.append(alpha[0])

            except Exception as e:
                print(e)
                index = np.random.randint(0, len(self.paths))
                continue

            break
        
        imgs = torch.stack(img_list, dim=0).float()
        alpha = torch.stack(alpha_list).float()
        min_crop_size = self.get_crop_size(alpha)

        hue_factor = random.random() -0.5  # Change hue by this factor
        brightness_factor = random.random() + 0.5  # Change brightness by this factor
        saturation_factor = random.random() + 0.5  # Change saturation by this factor
        

        crop_size = random.randint(min_crop_size, 512)  # Size of the crop
        imgs = self.transform(imgs,alpha,crop_size,hue_factor,brightness_factor,saturation_factor,self.hue_flag,self.crop_flag,self.flip_flag)
        imgs = v2.functional.resize(imgs, 512, interpolation=3, antialias=True).clamp(0, 1)
        
        ###to make 5 images to 7 images
        last_img = imgs[-1]
        front_view_img = imgs[2]
        output_imgs = torch.cat([imgs[0:2], imgs[3:7],], dim=0)

        data = {
            'cond_imgs': front_view_img,           # (3, H, W)
            'target_imgs': output_imgs,        # (6, 3, H, W)
        }
        
        return data
