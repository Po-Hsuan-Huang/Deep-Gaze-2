#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  9 15:50:49 2021

@author: pohsuanh

Reimplemtation of DeepGaze II by Matthias Kuemmerer et. al.

This is the VGG like DNN model with VGG16 backbone.

conv5 1, relu5 1, relu5 2, conv5 3, relu5 4 are extracted and concatenated.

before 4 1x1 convolution layers.

Finally, the output of the 1x1 convolutional block is rescaled to size of target image.

"""
import tensorflow as tf
from SaliencyMapData import MIT1003
from utils.smoothing import Gaussian_kernel
from utils.centerbias import load_center_bias_from_file
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow.keras.layers as layers
from tensorflow.keras.layers import Resizing as Resize

gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_visible_devices(gpus, 'GPU')

# Limite GPU Memroy occupancy of Tensorflow
gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.7, allow_growth=True)
sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))

# Set session setting to Keras execution.
tf.compat.v1.keras.backend.set_session(sess)

root_dir = '/home/pohsuanh/Desktop/pohsuan/projects/deep gaze/SALICON/'

mode ='fine-tuning'

BATCH_SIZE = 6

EPOCHS = 25

AUTOTUNE = tf.data.AUTOTUNE
#""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
# PREPROCESSING CENTER BIAS
#""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

train_center_bias_file = os.path.join(root_dir,'train_center_bias_tensor.npy')

val_center_bias_file = os.path.join(root_dir, 'val_center_bias_tensor.npy')



def convert_to_binary(x)->tf.Tensor:
    return tf.cast(x>0, tf.float32)

def convert_to_prob(x):
     x = tf.reshape(x, [8,256*256])
     x = tf.keras.activations.softmax(x)
     x = tf.reshape(x, [8,256,256,1])
     return x
############## READ CNETER BIAS ########################

"""
options = tf.data.Options()
options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA

train_bias_tensor, val_bias_tensor = load_center_bias_from_file(train_center_bias_file, val_center_bias_file)

train_bias = tf.data.Dataset.from_tensor_slices(train_bias_tensor).with_options(options)

val_bias = tf.data.Dataset.from_tensor_slices(val_bias_tensor).with_options(options)

train_bias = train_bias.batch(BATCH_SIZE).prefetch(AUTOTUNE).with_options(options)

val_bias = val_bias.batch(BATCH_SIZE).prefetch(AUTOTUNE).with_options(options)
"""

# del train_bias_tensor
# del val_bias_tensor
#%%
def convert_to_prob_density(target):

    target = tf.reshape(target,[8,-1])
    target = tf.math.softmax(target, axis = 1)
    target = tf.reshape(target, [8,256,256,1])

    return target

options = tf.data.Options()
options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA

train_imgs = tf.keras.preprocessing.image_dataset_from_directory(
    os.path.join(root_dir,'images','train'), batch_size = BATCH_SIZE, label_mode = None, shuffle =False,
  ).map(lambda x:x/255).prefetch(AUTOTUNE).with_options(options)

train_tars = tf.keras.preprocessing.image_dataset_from_directory(
    os.path.join(root_dir,'annotations','train'), batch_size = BATCH_SIZE,image_size=(120,
    160), label_mode = None, shuffle =False,
    ).map(lambda x:x/255).map(tf.image.rgb_to_grayscale).prefetch(AUTOTUNE).with_options(options)

val_imgs = tf.keras.preprocessing.image_dataset_from_directory(
    os.path.join(root_dir,'images','val'), batch_size = BATCH_SIZE, label_mode = None,shuffle =False,
    ).map(lambda x:x/255).prefetch(AUTOTUNE).with_options(options)

val_tars = tf.keras.preprocessing.image_dataset_from_directory(
    os.path.join(root_dir,'annotations','val'), batch_size = BATCH_SIZE,image_size=(120,160
    ), label_mode = None,shuffle =False,
  ).map(lambda x:x/255).map(tf.image.rgb_to_grayscale).prefetch(AUTOTUNE).with_options(options)

#%%
#==============================================================
# Load fixation maps (not gaussian blurred)
# Tried to load the numpy arrays as bmp images, or use ImageDataGenerator.
# The resulting output Tensors are not lossless.
#==============================================================

from glob import glob
def train_npy_gen():
    root_dir = '/media/pohsuanh/Data/SALICON/fixations/fixations/'
    fmap_paths = sorted(glob(os.path.join(root_dir, 'train_np', '0','*.npy')))
    tensor_gen = (tf.convert_to_tensor(np.load(path)) for path in fmap_paths)
    return tensor_gen

def val_npy_gen():
    root_dir = '/media/pohsuanh/Data/SALICON/fixations/fixations/'
    fmap_paths = sorted(glob(os.path.join(root_dir, 'val_np', '0','*.npy')))
    tensor_gen = (tf.convert_to_tensor(np.load(path)) for path in fmap_paths)
    return tensor_gen

train_fmaps = tf.data.Dataset.from_generator(
    train_npy_gen,
    output_signature=(
        tf.TensorSpec(shape=(480,640),dtype=tf.int32))
    ).map(lambda x: tf.expand_dims(x, axis=-1)).map(lambda image:tf.image.resize(image, (120,160))).map(lambda y: tf.where(y>0, 1, 0)).batch(BATCH_SIZE).prefetch(AUTOTUNE).with_options(options)

val_fmaps = tf.data.Dataset.from_generator(
    val_npy_gen,
    output_signature=(
        tf.TensorSpec(shape=(480,640),dtype=tf.int32))
    ).map(lambda x: tf.expand_dims(x,axis = -1)).map(lambda image:tf.image.resize(image, (120,160))).map(lambda y: tf.where( y> 0, 1, 0)).batch(BATCH_SIZE).prefetch(AUTOTUNE).with_options(options)


#%%
lr =0.001

def inverse_epoch_learning_scheduler(epoch, lr , decay_rate = 0.5, decay_cycle = 2):
    """
    Inverse Epoch Learning Rate Scheduler
    """
    if epoch < 2:
      return lr
    else:
      return  lr / (1 + decay_rate * epoch / decay_cycle)

def exponential_learning_scheduler(epoch, lr):
  if epoch < 10:
    return lr
  else:
    return lr * tf.math.exp(-0.1)

#%% Build a simple off the shelf vgg16 model, and check the layer names
def _create_model():

    vgg16 = tf.keras.applications.vgg16.VGG16(
            include_top=False, weights='imagenet', input_tensor=None,
            input_shape=(256,256,3), pooling=True)

    for layer in vgg16.layers:

            # names = ['block4_conv1',
            #         'block4_conv2',
            #         'block4_conv3',
            #         'block4_pool',
            #         'block5_conv1',
            #         'block5_conv2',
            #         'block5_conv3',
            #         'block5_pool']
            # if layer.name not in names:
                layer.trainable = False

    #""" Extend the base vgg16 model to a FCN-8 to generate fine-semantic map. """
    # inputs = tf.keras.Input(shape=(256,256,3))
    # x1 = tf.keras.applications.vgg16.preprocess_input(inputs)
    x5 = vgg16.output # 7*7*512


    # The selections of the feature maps are slightly different from the original
    # pepers because Tensroflow based Keras pre-built vgg16 doesn't allow access to
    # the conv layers before relu activation.
    # Instead, I chose conv4_3 after relu, and conv4_pool.
    # The 'conv' prefix here is equiavalent to 'relu' in the original paper
    # because all these layers go through relu activations.

    x_conv4_3 = vgg16.get_layer('block4_conv3').output # 28*28*512

    x_conv4_pool = vgg16.get_layer('block4_pool').output # 14*14*512

    x_conv5_1 = vgg16.get_layer('block5_conv1').output # 14*14*512

    x_conv5_2 = vgg16.get_layer('block5_conv2').output # 14*14*512

    x_conv5_3 = vgg16.get_layer('block5_conv3').output # 14*14*512


    #resize all 5 layers into 128x128

    feature_pyramid = [x_conv4_3, x_conv4_pool, x_conv5_1, x_conv5_2, x_conv5_3]

    # feature_pyramid = [ x_conv5_1, x_conv5_2, x_conv5_3]


    x6 = [Resize(128,128)(x) for x in feature_pyramid]  # 128 x 128 x 512*3

    x7 = layers.Concatenate(name='block7_cat1')(x6)

    # Readout Network
    l1 = tf.keras.regularizers.L1(0.001)
    l2 = tf.keras.regularizers.L2(1e-2)
    x8_1 = tf.keras.layers.Conv2D(16,(1,1), activation ='relu', kernel_regularizer=l2, name='block8_conv1')(x7) #128*128*16
    x8_2 = tf.keras.layers.Conv2D(32,(1,1), activation ='relu', kernel_regularizer=l2, name='block8_conv2')(x8_1) #128*128*32
    x8_3 = tf.keras.layers.Conv2D(2 ,(1,1), activation ='relu', kernel_regularizer=l2, name='block8_conv3')(x8_2) #128*128*2
    x8_4 = tf.keras.layers.Conv2D(1 ,(1,1), activation ='relu', kernel_regularizer=l2, name='block8_conv4')(x8_3) #128*128*1
    x8_5 = Resize(256,256, name = 'block8_rs')(x8_4)

    # Gaussian Smoothing
    gaussian_kernel = tf.convert_to_tensor(Gaussian_kernel(l=10, sig =5))
    gaussian_kernel = tf.expand_dims(gaussian_kernel, axis = -1)
    gaussian_kernel = tf.expand_dims( gaussian_kernel, axis = -1 )
    x9 = tf.nn.conv2d(x8_5, gaussian_kernel, strides = [1,1,1,1], padding='SAME', name='block9_conv1' )
    x9_1 = Resize(28,28, name='block9_rs1')(x9)
    #Add center_bias in form of log probability of the whole training set.
    # center_bias = tf.keras.Input(shape=[256, 256, 1], name='block10_in')
    # cb = Resize(28,28, name='block10_rs1')(center_bias)
    # cb_2 = layers.Flatten(name='block10_flat1')(cb)
    # p_cb = layers.Softmax(name='block10_soft1')(cb_2)
    # def logFunc(x):
    #     return tf.math.log(x)
    # p_cb_2 = layers.Lambda(logFunc, name='block10_lambda')(p_cb)
    # p_cb_3 = layers.Reshape((28,28,1), name='block10_rs')(p_cb_2)
    # p_cb_3 = 0.0*cb
    # x10 = layers.Add(name='block10_add')([x9_1, p_cb_3])
    x= layers.BatchNormalization(name='block10_bn')(x9_1)
    x = layers.Activation('sigmoid', name='block10_out')(x)

    # Covnert to probability
    # x = layers.Flatten()(x10)
    # x = layers.Softmax()(x)
    # x = layers.Reshape((28,28,1))(x)
    # Resize
    outputs = Resize(120,160)(x)

    DeepGaze = tf.keras.Model(vgg16.input, outputs)

    return DeepGaze

#%%
from utils.metrics_classes import AUC_Borji_v2
from utils.losses import borji_auc_loss_fn


def L_NSS(y_true, y_pred, axis = -1, batch_size = 8): # Normalized Scan Path Loss
    """
    y_true : fixations
    y_pred : predicted saliency map
    """

    x = tf.reshape(y_true,[batch_size, -1])
    y = tf.reshape(y_pred, [batch_size, -1])

    y_norm = (y - tf.reduce_mean(y, axis = axis))/tf.math.reduce_variance(y)

    return tf.reduce_mean(-1 * tf.reduce_sum(y_norm*x, axis = -1)/tf.reduce_sum(x, axis = -1))


def total_loss(y_true, y_pred):
    KLD = tf.keras.losses.KLD(y_true, y_pred)
    BCE = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    # CC = correlationLoss(y_true, y_pred)
    MSE = tf.keras.losses.MSE(y_true, y_pred)
    return 0.001*KLD+BCE

strategy = tf.distribute.MirroredStrategy()
print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

if mode  == 'training':

    with strategy.scope():
        b_auc =AUC_Borji_v2(replica_in_sync=strategy.num_replicas_in_sync)
        bc = tf.keras.metrics.BinaryCrossentropy(
        name="binary_crossentropy", dtype=None, from_logits=False, label_smoothing=0
        )
        auc = tf.keras.metrics.AUC(name = 'auc')
        acc = tf.keras.metrics.Accuracy(name ='Accuracy')
        adam = tf.keras.optimizers.Adam(learning_rate = 0.0001)
        sgd = tf.keras.optimizers.SGD(learning_rate = lr, momentum=0.9, nesterov=True )

        DeepGaze = _create_model()
        DeepGaze.compile(optimizer = sgd, loss = tf.keras.losses.MSE, metrics = [ acc, auc, b_auc ], run_eagerly=True)


    checkpoint_path = "./pretrained_model/best_model_small/pretrain_model_small.ckpt"


    # DeepGaze.compile(optimizer = sgd, loss = borji_auc_loss_fn, metrics = [ b_auc])

    try :

        DeepGaze.load_weights("./pretrained_model/best_model_small/pretain_model_small.ckpt/variables/variables")

        print('Load pretraiend weights.')

    except:

        print('No pretrained weights.')
    # DeepGaze.summary()
    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
    # train_in = tf.data.Dataset.zip((train_imgs, train_bias)).with_options(options)
    # val_in = tf.data.Dataset.zip((val_imgs, val_bias)).with_options(options)

    # for training
    train_data =tf.data.Dataset.zip((train_imgs, train_tars)).with_options(options)
    val_data = tf.data.Dataset.zip((val_imgs, val_tars)).with_options(options)


    lr_callback = tf.keras.callbacks.LearningRateScheduler(inverse_epoch_learning_scheduler)

    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                     save_freq = 'epoch',
                                                     save_weights_only = False,
                                                     save_best_only = True,
                                                     mode = 'max',
                                                     monitor= 'auc',
                                                     verbose=1)

    tb_callback = tf.keras.callbacks.TensorBoard(log_dir="./logs",
                                                histogram_freq=1,
                                                write_graph=True,
                                                write_images=True,
                                                write_steps_per_second=False,
                                                update_freq="epoch",
                                                profile_batch=2,
                                                embeddings_freq=2,
                                                embeddings_metadata='./embedding'
                                                )

    DeepGaze.fit( train_data, validation_data=val_data,epochs= 20, workers= 14 ,
              use_multiprocessing =True,
              steps_per_epoch = 400,
              validation_steps= 50,
              callbacks=[lr_callback, cp_callback, tb_callback ])

if mode  == 'fine-tuning':

    checkpoint_path = "./pretrained_model/best_model_small/best_model_small.ckpt"


    # DeepGaze.compile(optimizer = sgd, loss = borji_auc_loss_fn, metrics = [ b_auc])

    try :

        DeepGaze.load_weights("./pretrained_model/best_model_small/best_model_small.ckpt/variables/variables")

        print('Load pretraiend weights.')

    except:

        print('No pretrained weights.')

    with strategy.scope():
            b_auc =AUC_Borji_v2(replica_in_sync=strategy.num_replicas_in_sync)
            bc = tf.keras.metrics.BinaryCrossentropy(
            name="binary_crossentropy", dtype=None, from_logits=False, label_smoothing=0
            )
            auc = tf.keras.metrics.AUC(name = 'auc')
            acc = tf.keras.metrics.Accuracy(name ='Accuracy')
            adam = tf.keras.optimizers.Adam(learning_rate = 0.0001)
            sgd = tf.keras.optimizers.SGD(learning_rate = lr, momentum=0.9, nesterov=True )

            DeepGaze = _create_model()
            DeepGaze.compile(optimizer = sgd, loss = borji_auc_loss_fn, metrics = [auc, b_auc], run_eagerly=True)


    # DeepGaze.summary()
    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
    # train_in = tf.data.Dataset.zip((train_imgs, train_bias)).with_options(options)
    # val_in = tf.data.Dataset.zip((val_imgs, val_bias)).with_options(options)

    # for fine tuning
    train_data =tf.data.Dataset.zip((train_imgs, train_fmaps)).with_options(options)
    val_data = tf.data.Dataset.zip((val_imgs, val_fmaps)).with_options(options)

    lr_callback = tf.keras.callbacks.LearningRateScheduler(inverse_epoch_learning_scheduler)

    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                     save_freq = 'epoch',
                                                     save_weights_only = False,
                                                     save_best_only = True,
                                                     mode = 'max',
                                                     monitor= 'auc',
                                                     verbose=1)

    tb_callback = tf.keras.callbacks.TensorBoard(log_dir="./logs",
                                                histogram_freq=1,
                                                write_graph=True,
                                                write_images=True,
                                                write_steps_per_second=False,
                                                update_freq="epoch",
                                                profile_batch=2,
                                                embeddings_freq=2,
                                                embeddings_metadata='./embedding'
                                                )

    DeepGaze.fit( train_data, validation_data= val_data, epochs= 20, workers= 14 ,
              use_multiprocessing =True,
              steps_per_epoch = 400,
              validation_steps= 50,
              callbacks=[lr_callback, cp_callback, tb_callback ])
#%%
# ==============================================================================
# Evaluate the result
# ==============================================================================
from utils.metrics_functions import AUC_Judd, AUC_shuffled, AUC_Borji, CC, NSS, SIM

_queue_metrics_ = {'AUC_shuffled':[], 'AUC_Judd':[], 'AUC_Borji':[],'NSS':[], 'CC':[], 'SIM':[] }
_mean_metrics_ = {'AUC_shuffled':[], 'AUC_Judd':[], 'AUC_Borji':[],'NSS':[], 'CC':[], 'SIM':[]  }
_names_  = list(_queue_metrics_.keys())

other_fmaps= train_fmaps.shuffle(buffer_size =20).take(20)
val_data = tf.data.Dataset.zip((val_imgs, val_tars, val_fmaps, other_fmaps))

DeepGaze = _create_model()

DeepGaze.load_weights("./pretrained_model/best_model_small/best_model_small.ckpt/variables/variables")
#%%
for h, (img, tar, fmap, other_fmap) in enumerate(val_data.take(1)):

    """ each array is actually a mini batch"""

    pred = DeepGaze.predict(img, workers =12)
    tar = tar.numpy()
    fmap = fmap.numpy()
    other_fmap = other_fmap.numpy()
    # fig, axes = plt.subplots(3,3,sharex = True, sharey=True)
    # for i in range(3):
    #     axes[0][i].imshow(fmap[i])
    #     axes[0][i].set_title('fmap')
    #     axes[1][i].imshow(pred[i])
    #     axes[1][i].set_title('pred')
    #     axes[2][i].imshow(other_fmap[i])
    #     axes[2][i].set_title('other_fmap')

    # fig, axes = plt.subplots(6,2, sharex = True, sharey = True)

    # for i in range(6):
    score, (tp, fp)=  AUC_shuffled(fmap, pred, other_fmap)
    _queue_metrics_['AUC_shuffled'].append(score)
    #     tp = np.asanyarray(tp)
    #     fp = np.asanyarray(fp)
    #     axes[i][0].plot(np.sum(fp, axis=0), np.sum(tp, axis=0))
    #     axes[i][0].set_title('img{}_sAUC'.format(i))
    score = AUC_Judd(pred, fmap, jitter=True)
    _queue_metrics_['AUC_Judd'].append(score)

    score , (tp, fp)= AUC_Borji(fmap, pred)
    #     tp = np.asanyarray(tp)
    #     fp = np.asanyarray(fp)
    _queue_metrics_['AUC_Borji'].append(score)
    #     axes[i][1].plot(np.sum(fp, axis=0), np.sum(tp, axis=0))
    #     axes[i][1].set_title('img{}_Borji_AUC'.format(i))


    score = NSS(fmap, pred )
    _queue_metrics_['NSS'].append(score)

    score = CC(pred, tar)
    _queue_metrics_['CC'].append(score)

    score = SIM(pred, tar)
    _queue_metrics_['SIM'].append(score)


for i in range(6):
    mean =np.mean(np.asanyarray(_queue_metrics_[_names_[i]]))
    _mean_metrics_[_names_[i]].append(mean)

for i in range(6):
    print("{}: {}".format(list(_mean_metrics_.keys())[i], list(_mean_metrics_.values())[i]))

#%%
#==============================================================================
#Visualization
#==============================================================================
test_n = 1 #one element, but batch size is four.
BATCH_SIZE  = 6

fig = plt.figure(figsize =(3*4,8*4) )
# Reconstruct Data sets for ploting
fig.subplots_adjust(hspace=0.001)

_test_imgs = val_imgs.take(test_n)

_test_tars = val_tars.take(test_n)

for j, (img, tar_im) in enumerate( tf.data.Dataset.zip((_test_imgs, _test_tars))):

    for i in range(BATCH_SIZE):

        print('show images', img.shape )

        _img = tf.expand_dims(img[i], axis = 0)

        _tar_img = tf.expand_dims(tar_im[i], axis = 0)

        test_pred = DeepGaze(_img, training = False) #  list of Tensors

        fmap = test_pred.numpy() # np array

        ax = plt.subplot(BATCH_SIZE, 3, 1 + 3*i)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.imshow(_img[0].numpy().squeeze())  # original

        ax = plt.subplot(BATCH_SIZE, 3, 2 + 3*i)
        ax.set_xticks([])
        ax.set_yticks([])
        # plt.imshow(fmap[0].squeeze()-_cb[0].numpy().squeeze()) # predicted fixation
        plt.imshow(fmap[0].squeeze()) # predicted fixation

        ax = plt.subplot(BATCH_SIZE, 3, 3 + 3*i)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.imshow(_tar_img[0].numpy().squeeze()) # ground truth

plt.show()

# generate_images(vae, test_inputs, test_targets, test_n)
#%%

#==============================================================================
#Visualization with center bias
#==============================================================================
def visualize():

    test_n = 1 #one element, but batch size is four.

    fig = plt.figure(figsize =(3*4,8*4) )
    # Reconstruct Data sets for ploting
    fig.subplots_adjust(hspace=0.001)

    _test_imgs = val_imgs.take(test_n)

    _test_bias = val_bias.take(test_n)

    _test_tars = val_tars.take(test_n)

    _test_data = tf.data.Dataset.zip((_test_imgs, _test_bias))

    for j, ((img,cb), tar_im) in enumerate( tf.data.Dataset.zip((_test_data, _test_tars))):

        for i in range(BATCH_SIZE):

            print('show images', img.shape )

            _img = tf.expand_dims(img[i], axis = 0)

            _cb = tf.expand_dims(cb[i], axis = 0)

            _tar_img = tf.expand_dims(tar_im[i], axis = 0)

            test_pred = DeepGaze([_img, _cb], training = False) #  list of Tensors

            fmap = test_pred.numpy() # np array

            ax = plt.subplot(BATCH_SIZE, 3, 1 + 3*i)
            ax.set_xticks([])
            ax.set_yticks([])
            plt.imshow(_img[0].numpy().squeeze())  # original

            ax = plt.subplot(BATCH_SIZE, 3, 2 + 3*i)
            ax.set_xticks([])
            ax.set_yticks([])
            # plt.imshow(fmap[0].squeeze()-_cb[0].numpy().squeeze()) # predicted fixation
            plt.imshow(fmap[0].squeeze()) # predicted fixation

            ax = plt.subplot(BATCH_SIZE, 3, 3 + 3*i)
            ax.set_xticks([])
            ax.set_yticks([])
            plt.imshow(_tar_img[0].numpy().squeeze()) # ground truth

    plt.show()

# generate_images(vae, test_inputs, test_targets, test_n)
#%%
#==============================================================================
feats = []
for layer in DeepGaze.layers:
    # check for convolutional layer
    if hasattr(layer, 'name'):

        # get filter weights
        if layer.name =='block8_conv1':
            weights = layer.get_weights()[0]
            print(weights.shape)