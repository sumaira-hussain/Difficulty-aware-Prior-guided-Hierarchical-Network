from __future__ import print_function
from keras.preprocessing.image import ImageDataGenerator
import numpy as np 
import os
import glob
import skimage.io as io
import skimage.transform as trans
import cv2
from PIL import Image

#np.set_printoptions(threshold=np.nan)

def adjustData(img,mask,flag_multi_class,num_class):
    img = img / 255
    mask = mask /255
    mask[mask > 0.5] = 1
    mask[mask <= 0.5] = 0
    return (img,mask)


def trainGenerator(batch_size,train_path,image_folder,mask_folder,aug_dict,image_color_mode = "grayscale",
                    mask_color_mode = "grayscale",image_save_prefix  = "image",mask_save_prefix  = "mask",
                    flag_multi_class = False,num_class = 2,save_to_dir = None ,target_size = (256,256),seed = 1):
    '''
    can generate image and mask at the same time
    use the same seed for image_datagen and mask_datagen to ensure the transformation for image and mask is the same
    if you want to visualize the results of generator, set save_to_dir = "your path"
    '''
    image_datagen = ImageDataGenerator(**aug_dict)
    mask_datagen = ImageDataGenerator(**aug_dict)
    image_generator = image_datagen.flow_from_directory(
        train_path,
        classes = [image_folder],
        class_mode = None,
        color_mode = image_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = save_to_dir,
        save_prefix  = image_save_prefix,
        seed = seed)

    mask_generator = mask_datagen.flow_from_directory(
        train_path,
        classes = [mask_folder],
        class_mode = None,
        color_mode = mask_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = save_to_dir,
        save_prefix  = mask_save_prefix,
        seed = seed)
    train_generator = zip(image_generator, mask_generator)#combined generators into one which yeilds images and masks
    for (img,mask) in train_generator:
#Since batch is 2, two images are returned at a time, that is, img is an array of two gray-scale images, [2,256,256]
        img,mask = adjustData(img,mask,flag_multi_class,num_class)#The returned img is still [2,256,256]
        yield (img,mask)

def testGenerator(test_path, num_image = 186, target_size = (256,256), flag_multi_class = False, as_gray = True):
    for i in range(num_image):
        # j=122
        # i = i+j
        if (i < 9):
            img = io.imread(os.path.join(test_path,"00%d.png"%(i+1)),as_gray = as_gray)
        elif (i < 99):
            img = io.imread(os.path.join(test_path, "0%d.png" % (i + 1)), as_gray=as_gray)
        else:
            img = io.imread(os.path.join(test_path, "%d.png" % (i + 1)), as_gray=as_gray)
        img = img / 255
        img = trans.resize(img,target_size)
        img = np.reshape(img,img.shape+(1,)) if (not flag_multi_class) else img
        img = np.reshape(img,(1,)+img.shape)
        yield img



def saveResult(save_path,npyfile,flag_multi_class = False,num_class = 2):
      for i, item in enumerate(npyfile):
          img = item[:, :, 0]
          print(np.max(img), np.min(img))

          if np.max(img) >= 0.5:
              img[img > 0.5] = 255
              img[img <= 0.5] = 0
          else:
              img[img > 0] = 255
              img[img <= 0] = 0
          print(np.max(img), np.min(img))

          rimage = Image.fromarray(np.uint8(img))
          if (i<9):
              rimage.save(save_path + "00%d" % (i+1) + '.png')
          elif(i<99):
              rimage.save(save_path + "0%d" % (i + 1) + '.png')
          else:
              rimage.save(save_path + "%d" % (i + 1) + '.png')
