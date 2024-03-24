# -*- coding: UTF-8 -*-
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm


class ImageStack(object):
    def __init__(self,img_dir):

        self.IMAGE_DIR = img_dir
        self.files = os.listdir(self.IMAGE_DIR)
        self.IMAGE_SIZE = 384
        self.landSet_name_train = ['002053','002054','011002','011247','029040','032029','034034','035034',
                             '039034','044010','045026','047023','059014','061017',
                                   '063016','064014','064017','066017']

        self.landSet_name_test = ['035035', '063013', '035029', '032037', '066014', '029041', '032035', '029032',
                             '064012', '034037', '034029', '003052', '064015', '039035', '018008', '029044',
                             '034033', '032030', '050024', '063012']
        time_patches_eachImage = {}
        num_patches_eachImage = {}
        shape_paches_eachImage = {}

        for file in self.files:
            for name in self.landSet_name_test:
                if '_'+name+'_' in file:
                    if name in num_patches_eachImage.keys():
                        num_patches_eachImage[name] += 1
                    else:
                        num_patches_eachImage[name] = 1

                    if name not in time_patches_eachImage.keys():
                        time_patches_eachImage[name] = file.split('_')[-4]+'_'+ file.split('_')[-3]
        num_patches_eachImage['064012'] = 529
        for key in num_patches_eachImage.keys():
            value = str(num_patches_eachImage[key])
            for file in self.files:
                if key in file and '_'+value+'_' in file:
                    file = file.split('_')
                    print('******************************')
                    print(file)
                    print('******************************')
                    shape_paches_eachImage[key] = file[3]+'_'+file[5]
        self.time_patches_eachImage = time_patches_eachImage
        self.num_patches_eachImage = num_patches_eachImage
        self.shape_paches_eachImage = shape_paches_eachImage
        print(num_patches_eachImage)
        print(shape_paches_eachImage)
        print(time_patches_eachImage)

    def stack_image(self):
        for name in tqdm(self.landSet_name_test):
            row, col = [int(i) for i in self.shape_paches_eachImage[name].split('_')]
            print (row,col)
            to_image = np.zeros((row * self.IMAGE_SIZE, col * self.IMAGE_SIZE))
            index = 1
            for y in range(1, row + 1):
                for x in range(1, col + 1):
                    # blue_patch_100_5_by_12_LC08_L1TP_064015_20160420_20170223_01_T1.TIF
                    image_file_name = 'gt_patch_{}_{}_by_{}_LC08_L1TP_{}_{}_01_T1.TIF'\
                        .format(index,y, x, name,self.time_patches_eachImage[name])
                    index += 1
                    # for file in self.files:
                    #     if image_file_name in file:
                    #         image_file_name = file
                    #         break  #这里很重要
                    image_path = os.path.join(self.IMAGE_DIR, image_file_name)
                    from_image = Image.open(image_path)
                    mm = np.asarray(from_image)
                    to_image[(y - 1) * self.IMAGE_SIZE:y * self.IMAGE_SIZE,
                    (x - 1) * self.IMAGE_SIZE:x * self.IMAGE_SIZE] = mm

            to_image = np.asarray(to_image)*255
            to_image = Image.fromarray((to_image).astype(np.uint8))
            # to_image.save('/media/estar/Data1/Lgy/FasterNet/results/{}.png'.format(name))
            # to_image = None
            #print(to_image.shape,to_image.max())
            plt.imsave('./results/{}.png'.format(name),to_image,cmap=plt.cm.gray)

if __name__ == '__main__':
    img_s = ImageStack('./result/')
    img_s.stack_image()