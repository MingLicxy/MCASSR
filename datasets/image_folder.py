import os
import json
import random
from PIL import Image

import pickle
import imageio
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from datasets import register

# 训练时从单个文件夹中取得图像，再通过wrapper定义的数据处理获取HR-LR pairs
# out-of-scale SR测试时同上
@register('image-folder')
class ImageFolder(Dataset): # 后续其他的数据提取方式都是以这种为基础

    def __init__(self,
                 root_path,
                 split_file=None,  # json文件包含不同数据集划分
                 split_key=None,
                 first_k=None, # 取数据集前k张
                 repeat=1, # 数据集重复次数
                 cache='none' # 缓存模式：['none'(不缓存), 'bin'(二进制), 'in_memory'(加载进入内存)]
                 ):
        self.repeat = repeat
        self.cache = cache

        if split_file is None:
            filenames = sorted(os.listdir(root_path))
        else:
            with open(split_file, 'r') as f:
                filenames = json.load(f)[split_key]
        if first_k is not None:
            filenames = filenames[:first_k]

        self.files = []
        for filename in filenames:
            file = os.path.join(root_path, filename)

            if cache == 'none':
                self.files.append(file)

            elif cache == 'bin':
                bin_root = os.path.join(os.path.dirname(root_path),
                    '_bin_' + os.path.basename(root_path))
                if not os.path.exists(bin_root):
                    os.mkdir(bin_root)
                    print('mkdir', bin_root)
                bin_file = os.path.join(
                    bin_root, filename.split('.')[0] + '.pkl')
                if not os.path.exists(bin_file):
                    with open(bin_file, 'wb') as f:
                        pickle.dump(imageio.imread(file), f)
                    print('dump', bin_file)
                self.files.append(bin_file)
            ##################################### BUG 无论图像是单通道还是多通道通通转化为三通道RGB ################################
            elif cache == 'in_memory': # 不用先将.png读取为numpy数组再转化为Tensor
                self.files.append(transforms.ToTensor()(
                    Image.open(file).convert('RGB'))) # [3,H,W]
            elif cache == 'in_memory_L': 
                self.files.append(transforms.ToTensor()(
                    Image.open(file).convert('L'))) # [1,H,W]
            elif cache == 'mat': 
                 self.files.append(file)

    def __len__(self):
        return len(self.files) * self.repeat

    def __getitem__(self, idx):
        x = self.files[idx % len(self.files)]
        
        #TODO 在__getitem__中将numpy数组转化为Tensor
        if self.cache == 'none':
            return transforms.ToTensor()(Image.open(x).convert('RGB'))
        
        elif self.cache == 'bin': # 这里只能处理[H,W,C]形状的numpy数组
            with open(x, 'rb') as f:
                x = pickle.load(f)
            x = np.ascontiguousarray(x.transpose(2, 0, 1)) # [C,H,W]
            x = torch.from_numpy(x).float() / 255 # 归一化
            return x

        elif self.cache == 'in_memory': # 转化提前完成了
            return x
        elif self.cache == 'in_memory_L': 
            return x
        
        #elif self.cache == 'mat': # .mat文件里可以存numpy数组/Tensor，返回的是字典



# in-scale SR测试时HR和LR分别从两个文件夹中取得（多对比MRI训练时采用）
@register('paired-image-folders')
class PairedImageFolders(Dataset):

    def __init__(self, root_path_1, root_path_2, **kwargs):
        self.dataset_1 = ImageFolder(root_path_1, **kwargs)
        self.dataset_2 = ImageFolder(root_path_2, **kwargs)

    def __len__(self):
        return len(self.dataset_1)

    def __getitem__(self, idx):
        return self.dataset_1[idx], self.dataset_2[idx]
    
#TODO 用于多对比MRI加载数据（在测试时使用）
@register('mc-paired-image-folders')
class MRIPairedImageFolders(Dataset):

    def __init__(self, root_path_1, root_path_2, root_path_3, **kwargs):
        self.dataset_1 = ImageFolder(root_path_1, **kwargs)
        self.dataset_2 = ImageFolder(root_path_2, **kwargs)
        self.dataset_3 = ImageFolder(root_path_3, **kwargs)

    def __len__(self):
        return len(self.dataset_1)

    def __getitem__(self, idx):
        return self.dataset_1[idx], self.dataset_2[idx], self.dataset_3[idx]




############################## code for real arbitrary SR dataset(COZ) ############################
idx_with_6 = [154,155,156,159,160,161,162,163,164,165,166,167,168,169,170,171,172,173
                ,174,175,176,177,178,179,180,181,182,183,184,186,187,188,189,190]

class RealImageFolder(Dataset):

    def __init__(self, root_path,first_k=None,
                 repeat=1, cache='none', data_type='LR'):
        self.repeat = repeat
        self.cache = cache
        self.data_type = data_type

        self.idx_without_6 = []
        self.test_scale = 0

        if self.data_type == 'HR':
            self.files = []
            filenames = sorted(os.listdir(root_path))
            if first_k is not None:
                filenames = filenames[:first_k]

            for filename in filenames:
                file = os.path.join(root_path, filename)

                if cache == 'none':
                    self.files.append(file)

                elif cache == 'bin':
                    bin_root = os.path.join(os.path.dirname(root_path),
                        '_bin_' + os.path.basename(root_path))
                    if not os.path.exists(bin_root):
                        os.mkdir(bin_root)
                        print('mkdir', bin_root)
                    bin_file = os.path.join(
                        bin_root, filename.split('.')[0] + '.pkl')
                    if not os.path.exists(bin_file):
                        with open(bin_file, 'wb') as f:
                            pickle.dump(imageio.imread(file), f)
                        print('dump', bin_file)
                    self.files.append(bin_file)

                elif cache == 'in_memory':
                    self.files.append(transforms.ToTensor()(
                        Image.open(file).convert('RGB')))
        
        else:
            self.files = []
            filename_list = []
            dirnames = sorted(os.listdir(root_path))

            for dirname in dirnames:
                dir_path = os.path.join(root_path,dirname)
                file_list = sorted([os.path.join(dirname, filename) for filename in os.listdir(dir_path)])
                if file_list:
                    filename_list.append(file_list)                  
                
              
            if first_k is not None:
                filename_list = filename_list[:first_k]

            for filenames in filename_list:
                files = []
                # print(filenames)
                for filename in filenames:
                    file = os.path.join(root_path, filename)

                    if cache == 'none':
                        files.append(file)

                    elif cache == 'bin':
                        bin_root = os.path.join(os.path.dirname(root_path),
                            '_bin_' + os.path.basename(root_path))
                        if not os.path.exists(bin_root):
                            os.mkdir(bin_root)
                            print('mkdir', bin_root)
                        bin_file = os.path.join(
                            bin_root, filename.split('.')[0] + '.pkl')
                        if not os.path.exists(bin_file):
                            with open(bin_file, 'wb') as f:
                                pickle.dump(imageio.imread(file), f)
                            print('dump', bin_file)
                        files.append(bin_file)

                    elif cache == 'in_memory':
                        files.append(transforms.ToTensor()(
                            Image.open(file).convert('RGB')))

                self.files.append(files)                  
            

    def set_test_scale(self,test_scale):
        self.test_scale = test_scale


    def __len__(self):
        if self.test_scale == 6:
            return len(idx_with_6) *self.repeat
        
        return len(self.files) * self.repeat

    def __getitem__(self, idx):
        if self.test_scale == 6:
            x = self.files[idx_with_6[idx % len(idx_with_6)] - 154]
        else:
            x = self.files[idx % len(self.files)]
        # print(idx)

        if self.data_type =='LR':
            if self.test_scale ==0:
                x = random.choice(x)
            else:
                x = x[self.test_scale-2]

        if self.cache == 'none':
            return transforms.ToTensor()(Image.open(x).convert('RGB'))

        elif self.cache == 'bin':
            with open(x, 'rb') as f:
                x = pickle.load(f)
            x = np.ascontiguousarray(x.transpose(2, 0, 1))
            x = torch.from_numpy(x).float() / 255
            return x

        elif self.cache == 'in_memory':
            return x

@register('paired-real-image-folders')
class PairedRealImageFolders(Dataset):

    def __init__(self, root_path_1, root_path_2, **kwargs):
        self.dataset_1 = RealImageFolder(root_path_1, **kwargs)
        self.dataset_2 = RealImageFolder(root_path_2, **kwargs, data_type='HR')


    def set_test_scale(self,test_scale):
        self.dataset_1.set_test_scale(test_scale)
        self.dataset_2.set_test_scale(test_scale)

    def __len__(self):
        return len(self.dataset_1)

    def __getitem__(self, idx):
        return self.dataset_1[idx], self.dataset_2[idx]





    
    

class RealImageFolderTest(Dataset):

    def __init__(self, root_path,first_k=None,
                 repeat=1, cache='none', data_type='LR'):
        self.repeat = repeat
        self.cache = cache
        self.data_type = data_type

        self.idx_without_6 = []
        self.test_scale = 0

        if self.data_type == 'HR':
            self.files = []
            filenames = sorted(os.listdir(root_path))
            if first_k is not None:
                filenames = filenames[:first_k]

            for filename in filenames:
                file = os.path.join(root_path, filename)

                if cache == 'none':
                    self.files.append(file)

                elif cache == 'bin':
                    bin_root = os.path.join(os.path.dirname(root_path),
                        '_bin_' + os.path.basename(root_path))
                    if not os.path.exists(bin_root):
                        os.mkdir(bin_root)
                        print('mkdir', bin_root)
                    bin_file = os.path.join(
                        bin_root, filename.split('.')[0] + '.pkl')
                    if not os.path.exists(bin_file):
                        with open(bin_file, 'wb') as f:
                            pickle.dump(imageio.imread(file), f)
                        print('dump', bin_file)
                    self.files.append(bin_file)

                elif cache == 'in_memory':
                    self.files.append(transforms.ToTensor()(
                        Image.open(file).convert('RGB')))
        
        else:
            self.files = []
            filename_list = []
            dirnames = sorted(os.listdir(root_path))

            for dirname in dirnames:
                dir_path = os.path.join(root_path,dirname)
                file_list = sorted([os.path.join(dirname, filename) for filename in os.listdir(dir_path)])
                if file_list:
                    filename_list.append(file_list)                  
                
              
            if first_k is not None:
                filename_list = filename_list[:first_k]

            for filenames in filename_list:
                files = []
                # print(filenames)
                for filename in filenames:
                    file = os.path.join(root_path, filename)

                    if cache == 'none':
                        # img = cv2.imread(file)
                        # cv2.imwrite(file, img)
                        files.append(file)

                    elif cache == 'bin':
                        bin_root = os.path.join(os.path.dirname(root_path),
                            '_bin_' + os.path.basename(root_path))
                        if not os.path.exists(bin_root):
                            os.mkdir(bin_root)
                            print('mkdir', bin_root)
                        bin_file = os.path.join(
                            bin_root, filename.split('.')[0] + '.pkl')
                        if not os.path.exists(bin_file):
                            with open(bin_file, 'wb') as f:
                                pickle.dump(imageio.imread(file), f)
                            print('dump', bin_file)
                        files.append(bin_file)

                    elif cache == 'in_memory':
                        files.append(transforms.ToTensor()(
                            Image.open(file).convert('RGB')))

                self.files.append(files)      
                
                            
            

    def set_test_scale(self,test_scale):
        self.test_scale = test_scale


    def __len__(self):
        if self.test_scale == 6:
            return len(idx_with_6) *self.repeat
        
        return len(self.files) * self.repeat

    def __getitem__(self, idx):
        if self.test_scale == 6:
            x = self.files[idx_with_6[idx % len(idx_with_6)] - 154]
        else:
            x = self.files[idx % len(self.files)]
        # print(x)

        if self.data_type =='LR':
            if self.test_scale ==0:
                x = random.choice(x)
            else:
                x = x[self.test_scale-2]

        if self.cache == 'none':
            return x

        elif self.cache == 'bin':
            with open(x, 'rb') as f:
                x = pickle.load(f)
            x = np.ascontiguousarray(x.transpose(2, 0, 1))
            x = torch.from_numpy(x).float() / 255
            return x

        elif self.cache == 'in_memory':
            return x

@register('paired-real-image-folders-test')
class PairedRealImageFolders(Dataset):

    def __init__(self, root_path_1, root_path_2, **kwargs):
        self.dataset_1 = RealImageFolderTest(root_path_1, **kwargs)
        self.dataset_2 = RealImageFolderTest(root_path_2, **kwargs, data_type='HR')


    def set_test_scale(self,test_scale):
        self.dataset_1.set_test_scale(test_scale)
        self.dataset_2.set_test_scale(test_scale)

    def __len__(self):
        return len(self.dataset_1)

    def __getitem__(self, idx):
        fileh = self.dataset_2[idx]
        filel = self.dataset_1[idx]

        hr_image = Image.open(fileh)
        hr_width, hr_height = hr_image.size
        lr_image = Image.open(filel)
        lr_width, lr_height = lr_image.size
        s = hr_width/lr_width

        w_lr = 48
        x0 = random.randint(0, lr_width - w_lr)
        y0 = random.randint(0, lr_height - w_lr)
        try:
            crop_lr = lr_image.crop([ x0, y0,  x0 + w_lr,y0 + w_lr])
        except: 
            print(filel)
        w_hr = round(w_lr * s)
        x1 = round(x0 * s)
        y1 = round(y0 * s)
        try:
            crop_hr = hr_image.crop([x1,  y1,x1 + w_hr, y1 + w_hr])
        except:
            print(fileh)

        return transforms.ToTensor()(crop_lr.convert('RGB')), transforms.ToTensor()(crop_hr.convert('RGB'))
    
    
    
    
class RealImageFolderRandom(Dataset):

    def __init__(self, root_path,first_k=None,
                 repeat=1, cache='none', data_type='LR'):
        self.repeat = repeat
        self.cache = cache
        self.data_type = data_type


        self.files = []
        filename_list = []
        dirnames = sorted(os.listdir(root_path))

        for dirname in dirnames:
            dir_path = os.path.join(root_path,dirname)
            file_list = sorted([os.path.join(dirname, filename) for filename in os.listdir(dir_path)])
            if file_list:
                filename_list.append(file_list)                  
            
            
        if first_k is not None:
            filename_list = filename_list[:first_k]

        for filenames in filename_list:
            files = []
            # print(filenames)
            for filename in filenames:
                file = os.path.join(root_path, filename)
                files.append(file)

            self.files.append(files)      
                
                            
        

    def __len__(self):
        
        return len(self.files) * self.repeat

    def __getitem__(self, idx):

        x = self.files[idx % len(self.files)]

        x = random.sample(x,2)

        return x

@register('paired-real-image-folders-random')
class PairedRealImageFoldersRandom(Dataset):

    def __init__(self, root_path_1, **kwargs):
        self.dataset_1 = RealImageFolderRandom(root_path_1, **kwargs)


    def __len__(self):
        return len(self.dataset_1)

    def __getitem__(self, idx):
        filel,fileh = self.dataset_1[idx]

        lr_image = Image.open(filel)
        hr_image = Image.open(fileh)

        if (hr_image.size[0] < lr_image.size[0]):
            t = hr_image
            hr_image = lr_image
            lr_image = t

        hr_width, hr_height = hr_image.size
        lr_width, lr_height = lr_image.size
        
        s = hr_width/lr_width

        w_lr = 48
        x0 = random.randint(0, lr_width - w_lr)
        y0 = random.randint(0, lr_height - w_lr)
        try:
            crop_lr = lr_image.crop([ x0, y0,  x0 + w_lr,y0 + w_lr])
        except: 
            print(filel)
        w_hr = round(w_lr * s)
        x1 = round(x0 * s)
        y1 = round(y0 * s)
        try:
            crop_hr = hr_image.crop([x1,  y1,x1 + w_hr, y1 + w_hr])
        except:
            print(fileh)

        return transforms.ToTensor()(crop_lr.convert('RGB')), transforms.ToTensor()(crop_hr.convert('RGB'))
    