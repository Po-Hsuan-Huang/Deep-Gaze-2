#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 10 23:18:50 2021

@author: pohsuanh

Data Augmentation module.


"""
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt



def augment(data_images: tf.data.Dataset , target_images: tf.data.Dataset, augmentations:str)-> tf.data.Dataset:

    # Apply an augmentation only in 25% of the cases.
    
    new_imgs = []
    new_targets = []    
    _dic_ = {'flip':_flip, 'zoom':_zoom, 'color':_color}
    
    for img, tar in zip(data_images, target_images):
        
        for f in augmentations :
            
            if tf.random.uniform([], 0 , 1) > 0.25 :
                
                func = _dic_[f]
                
                img, tar = func( img, tar)
                                        
        new_imgs.append(img)
        new_targets.append(tar)
        
    images = tf.convert_to_tensor(new_imgs, dtype = tf.float16)
    targets = tf.convert_to_tensor(new_targets,dtype = tf.float16)
    
    return images, targets




def _flip(x: tf.Tensor, y: tf.Tensor) -> tf.Tensor:
    """Flip augmentation. The function is not supposed be called directly.

    Args:
        x: Image to flip

    Returns:
        Augmented image
    """
    x = tf.image.flip_left_right(x)
    
    y = tf.image.flip_left_right(y)

    return x, y

def _color(x: tf.Tensor, y:tf.Tensor) -> tf.Tensor:
    """Color augmentation
    The function is not supposed be called directly.

    Args:
        x: Image

    Returns:
        Augmented image
    
    """
    x = tf.image.random_hue(x, 0.03)
    x = tf.image.random_saturation(x, 0.6, 1.6)
    x = tf.image.random_brightness(x, 0.05)
    x = tf.image.random_contrast(x, 0.7, 1.3)
    x = tf.experimental.numpy.clip(x, 0, 255)
        
    return x, y

def _zoom(x: tf.Tensor, y: tf.Tensor) -> tf.Tensor:
    """
    Random Crop and Resize.
    The function is not supposed be called directly.

    Parameters
    ----------
    x : tf.Tensor
        DESCRIPTION.
    y : tf.Tensor
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """

    # Generate 20 crop settings, ranging from a 1% to 20% crop.
    scales = list(np.arange(0.8, 1.0, 0.05))
    boxes = np.zeros((len(scales), 4))

    for i, scale in enumerate(scales):
        x1 = y1 = 0.5 - (0.5 * scale)
        x2 = y2 = 0.5 + (0.5 * scale)
        boxes[i] = [x1, y1, x2, y2]

    def crop(img):
        shape = (img.shape[0], img.shape[1])
        # Create different crops for an image
        crops = tf.image.crop_and_resize([img], boxes=boxes, box_indices=np.zeros(len(scales)), crop_size=shape)
        # Return a random crop
        return crops

    x = crop(x)
    
    y = crop(y)
   
    choice = tf.random.uniform(shape=[], minval=0, maxval=len(scales), dtype=tf.int32)
   
    x, y = x[choice] , y[choice]

    return x, y