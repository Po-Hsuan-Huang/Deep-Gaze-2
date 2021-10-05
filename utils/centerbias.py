#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  9 16:38:39 2021

@author: pohsuanh

Saliency Map Center Bias Preprocessing Module

"""

import numpy as np
import tensorflow as tf
import os

def Array2Dataset(images, targets):
    NUM_TRAINSET = targets.shape[0]
    CENTER_BIAS = tf.reduce_mean( targets, axis =0)
    CENTER_BIAS = ( CENTER_BIAS - tf.reduce_mean(CENTER_BIAS))/tf.math.reduce_std(CENTER_BIAS)
    center_bias = np.expand_dims( CENTER_BIAS, 0)
    center_bias = np.repeat(center_bias, NUM_TRAINSET, axis = 0)
    center_bias = tf.data.Dataset.from_tensor_slices(center_bias)

    images  = tf.data.Dataset.from_tensor_slices(images)

    targets = tf.data.Dataset.from_tensor_slices(targets)

    return images, center_bias, targets



def load_center_bias_from_file( fpath_train, fpath_val):
    """
    Parameters
    ----------
    fpath_train : TYPE
        DESCRIPTION.
    fpath_val : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        CENTER BIAS OF TRAINING SET.
    TYPE
        CENTER BIAS OF VALIDATION SET.

    """

    NUM_TRAINSET = 10000

    NUM_VALSET = 5000

    results = []

    print('LOAD CENTER BIAS')

    paths = [fpath_train, fpath_val]

    nums = [NUM_TRAINSET, NUM_VALSET]

    for path, num in zip(paths, nums):

        center_bias = np.load(path) # Dataset consists of (1,W,H,C) Tensor

        def expand(center_bias):

            center_bias = np.expand_dims( center_bias, 0)

            center_bias = np.repeat(center_bias, num, axis = 0)

            center_bias = tf.convert_to_tensor(center_bias)

            return center_bias

        center_bias = expand(center_bias)

        results.append(center_bias)

    return results[0], results[1]

def save_center_bias_from_images(root_dir, train_center_bias_file, val_center_bias_file, get_probability_cb = False):
    """


    Parameters
    ----------
    root_dir : TYPE
        ROOT DIRECTORY OF THE DATA SET. THE FOLDER SHOULD CONTAIN TWO SUBFOLDERS
        'images', AND 'annotations'. FOR EACH SUBFOLDER, THERE ARE TWO SUBFOLDERS
        'train' and 'val'.

    train_center_bias_file : TYPE
        SAVE PATH OF CENTERBIAS NUMPY ARRAY OF TRAINING IMAGES.

    val_center_bias_file : TYPE
        SAVE PATH OF CENTER BIAS NUMPY ARRY OF VALIDATION IMAGES.

    get_probability_cb : OPTION BOOLEAN
        SAVE PROBABILITY DENSITY OF CENTER_BIAS OR NOT

    Returns
    -------
    CENTER_BIAS : TYPE
        DESCRIPTION.

    """

    if os.path.exists(train_center_bias_file) or os.path.exists(val_center_bias_file) :

        key = input('files alerady exists. Are you sure you want to overwrite?')

        if key == True :

            print('CREATE CENTER BIAS')

            train_tars = tf.keras.preprocessing.image_dataset_from_directory(
                os.path.join(root_dir,'annotations','train'), label_mode = None,
               )

            val_tars = tf.keras.preprocessing.image_dataset_from_directory(
                os.path.join(root_dir,'annotations','val'), label_mode = None,
             )

            # Process fixmaps to create center bias tensor.

            val_imgs = tf.keras.preprocessing.image_dataset_from_directory(
            os.path.join(root_dir,'images','val'), label_mode = None,
            )

            val_tars = tf.keras.preprocessing.image_dataset_from_directory(
            os.path.join(root_dir,'annotations','val'), label_mode = None,
            )

            def get_centerbias(targets):
                """
                CALCULATE CENTER BIAS FROM TARGET IMAGES
                THE OUTPUT TENSOR WILL BE (10000,W,H,C) FOR TRAINING SET,
                AND THATS TOO MUCH FOOTPRINT.
                WE SAVE THE TENSOR AS (1,W,H,C), AND SCALE BACK WHEN LOADING.
                """
                CENTER_BIAS = tf.reduce_mean( targets, axis =0)/255

                def rgb2gray(rgb):
                    color_vec = tf.reshape(tf.convert_to_tensor([0.2989, 0.5870, 0.1140]), (3,1))
                    return tf.linalg.matmul(rgb, color_vec)

                CENTER_BIAS  = rgb2gray(CENTER_BIAS)

                return CENTER_BIAS

            def get_prob_centerbias(targets):

                CENTER_BIAS = tf.reduce_mean( targets, axis =0)/255

                def rgb2gray(rgb):
                    color_vec = tf.reshape(tf.convert_to_tensor([0.2989, 0.5870, 0.1140]), (3,1))
                    return tf.linalg.matmul(rgb, color_vec)

                CENTER_BIAS  = rgb2gray(CENTER_BIAS)

                shape = CENTER_BIAS.shape

                CENTER_BIAS = tf.reshape(CENTER_BIAS, [-1])

                CENTER_BIAS = tf.math.softmax(CENTER_BIAS)

                CENTER_BIAS = tf.reshape(CENTER_BIAS, shape)

                return CENTER_BIAS

            train_bias = train_tars.map(get_centerbias)

            val_bias = val_tars.map(get_centerbias)

            # convert MapDataset to Tensor to save file.

            train_bias_tensor = train_bias.as_numpy_iterator().next()

            val_bias_tensor = val_bias.as_numpy_iterator().next()

            np.save( train_center_bias_file, train_bias_tensor)

            np.save( val_center_bias_file, val_bias_tensor)

            if get_probability_cb == True:

                train_bias = train_tars.map(get_prob_centerbias)

                val_bias = val_tars.map(get_prob_centerbias)

                # convert MapDataset to Tensor to save file.

                train_bias_tensor = train_bias.as_numpy_iterator().next()

                val_bias_tensor = val_bias.as_numpy_iterator().next()

                np.save( train_center_bias_file, train_bias_tensor)

                np.save( val_center_bias_file, val_bias_tensor)