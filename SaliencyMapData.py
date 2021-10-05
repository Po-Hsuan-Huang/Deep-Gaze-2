#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 20 11:12:16 2021

@author: pohsuanh

MIT1003 Tensorflow Dataset 

Dependency: 
 
 SALICON DATA API : https://github.com/NUS-VIP/salicon-api    
 
 
 
"""
import os
import tensorflow as tf
import numpy as np
from glob import glob 
from salicon.salicon import SALICON
import skimage
from glob import glob as glob
import urllib
import zipfile
#Load image files into tf.Tensor

class MIT1003():
    
    def __init__(self, root_dir=
            '/home/pohsuanh/Documents/deep_gaze/Datasets/Context Saliency/DeepGaze/mit1003_dataset',
            data_type = 'jpeg', target_type= 'jpg'
            ):

        self.root_dir = '/home/pohsuanh/Desktop/pohsuan/projects/deep gaze/MIT1003/'

        # self.root_dir = root_dir
        
        self.data_type = data_type
        
        self.target_type = target_type
        
        self.img_path = os.path.join(self.root_dir, 'ALLSTIMULI')
    
        self.label_path = os.path.join(self.root_dir, 'ALLFIXATIONMAPS')
    
        assert os.path.isdir(self.img_path), 'path does not exist.'
            
        assert os.path.isdir(self.label_path), 'path does not exist.'
     
    def load_and_preprocess(self, paths:list)-> tf.Tensor:
    
        frames = []
        
        for path in paths :
            v = tf.io.read_file(path)
            img_tensor = tf.io.decode_image(v, channels = 3)
            img_final = tf.image.resize(img_tensor, (224, 224))
            img_final = img_final/255.0 
            frames.append(img_final)            
            
        tf_data = tf.convert_to_tensor(frames)
            
        return tf_data
    
    def load_data(self):
        
        tensor_imgs = self.load_and_preprocess(sorted(glob(os.path.join(self.img_path,'*.'+self.data_type))))

        return tensor_imgs

    def load_target(self):
        
        tensor_targets = self.load_and_preprocess(sorted(glob(os.path.join(self.label_path,'*fixMap.jpg'))))   

        tensor_targets = tf.math.reduce_mean(tensor_targets, axis = 3, keepdims = True)
        
        return tensor_targets
        
    def load(self):
        
        data, target = self.load_data(), self.load_target()
        
        return data, target
    
    
    
class CAT2000():
    
    def __init__(self,root_dir=
            '/home/pohsuanh/Documents/deep_gaze/Datasets/Context Saliency/DeepGaze/cat2000_dataset',
            data_type = 'jpg', target_type= 'jpg'):
        
        self.root_dir = '/home/pohsuanh/Desktop/pohsuan/projects/deep gaze/CAT2000/'
        
        self.train_folder = 'trainSet'
        
        self.test_folder = 'testSet'
        
        self.FixMaps_folder = 'FIXATIONMAPS'
        
        self.Stimuli_folder = 'Stimuli'
        
        
    def train_set(self):
        
        #Training Set
        
        dir_path_fixmaps = os.path.join(self.root_dir, self.train_folder, self.FixMaps_folder)
        
        dir_path_stimli = os.path.join(self.root_dir, self.train_folder, self.Stimuli_folder)
        
        train_data = tf.keras.preprocessing.image_datatset_from_directory(dir_path_stimli, labels = None, image_size = (224, 224))
        
        train_target = tf.keras.preprocessing.image_dataset_from_directory(dir_path_fixmaps, labels = None, image_size = (224, 224))
        
        return train_data, train_target
        
    def test_set(self): 
        # Test Set
        
        dir_path_fixmaps = os.path.join(self.root_dir, self.test_folder, self.FixMaps_folder)
        
        dir_path_stimli = os.path.join(self.root_dir, self.test_folder, self.Stimuli_folder)
        
        test_data = tf.keras.preprocessing.image_datatset_from_directory(dir_path_stimli, labels = None, image_size = (224, 224))
        
        test_target = tf.keras.preprocessing.image_dataset_from_directory(dir_path_fixmaps, labels = None, image_size = (224, 224))
        
        return test_data, test_target
    
class SALICON_2014():
    """
    SALICON Dataset 
    """
    def __init__(self,root_dir = None, partition= None):
        """

        Parameters
        ----------
        root_dir : TYPE, optional
                THE ROOT DIRECTORY OF SALICON DATA SET. The default is None.
        **kwargs : TYPE
            DESCRIPTION.
            
        partition : 
            THE PARTITION OF DATA SET TO USE. THERE ARE 10 PARTITION.
            EACH CONSISTS OF 1000 IMAGES.

        Returns
        -------
        None.

        """
        
        self.train_folder = 'train'
        
        self.val_folder = 'val'
        
        self.test_folder = 'test'
        
        self.FixMaps_folder = 'annotations'
        
        self.Stimuli_folder = 'images'
        
        self.Annotation_folder = 'annotations'
        
        self.root_dir_ls =[ '/home/pohsuanh/Documents/deep_gaze/Datasets/Context Saliency/DeepGaze/SALICON',
                           '/home/pohsuanh/Desktop/pohsuan/projects/deep gaze/SALICON/' # GPU9 PC
                           ]
        
        if root_dir :
            self.root_dir = root_dir
        else: 
            self.root_dir = self.root_dir_ls[1]
            
            
        if partition : # If partition of the data set is specified
        
            self.partition = partition
                        
            train_ids_txt = ['train_ids_aa, train_ids_ab', 'train_ids_ac', 
                             'train_ids_ad', 'train_ids_ae', 'train_ids_af',
                             'train_ids_ag', 'train_ids_ah', 'train_ids_ai',
                             'train_ids_aj']
            
            val_ids_txt = ['val_ids_aa, val_ids_ab', 'val_ids_ac', 
                             'val_ids_ad', 'val_ids_ae', 'val_ids_af',
                             'val_ids_ag', 'val_ids_ah', 'val_ids_ai',
                             'val_ids_aj']
            
            train_id_path = os.path.join(self.root_dir, self.Stimuli_folder, train_ids_txt[partition])
            
            
            val_id_path = os.path.join(self.root_dir, self.Stimuli_folder, val_ids_txt[partition])

            self.train_ids = [line.rstrip() for line in open(train_id_path)]
 
            self.val_ids = [line.rstrip() for line in open(val_id_path)]
            
            self.split_ID_dict = {'train': self.train_ids, 'val': self.val_ids} 

     
    def load_images(self, split):
        """
        args:
            split : List 
                'train','val', 'test'
        """
        self.split = split
                
        for split in self.split:
            if split == 'train':
                split = self.train_folder
                self.data_dir = os.path.join(self.root_dir, self.Stimuli_folder, split)
            if split == 'val':
                split = self.val_folder
                self.data_dir = os.path.join(self.root_dir, self.Stimuli_folder, split)
            if split == 'test':
                split = self.test_folder
                self.data_dir = os.path.join(self.root_dir, self.Stimuli_folder, split)
            
        try:     
            assert os.path.isdir(self.data_dir), 'No such directoroy'
        except AssertionError:
            # create directory folders
            if not os.path.exists(os.path.join( self.root_dir, self.Stimuli_folder)):
                print ('creating ../images to host images in SALICON (Microsoft COCO) dataset...')
                os.mkdir(os.path.join(self.root_dir, self.Stimuli_folder))
                os.mkdir(os.path.join(self.root_dir, self.Stimuli_folder, self.train_folder))
                os.mkdir(os.path.join(self.root_dir, self.Stimuli_folder, self.val_folder))
                os.mkdir(os.path.join(self.root_dir, self.Stimuli_folder, self.test_folder))
                print ('done')
            
            print ("The following steps help you download images and annotations.")
            print ("Given the size of zipped image files, manual download is recommended at http://mscoco.org/download")
            # download train images
            if self.query_yes_no("Do you want to download zipped training images [1.5GB] under ./images/train/?", default='no'):
                url = 'https://www.dropbox.com/s/cy96zvud8fdpwde/train.zip?dl=1'
                dst = os.path.join(self.root_dir, self.Stimuli_folder, self.train_folder, 'train2015r1.zip')
                self.download( url, dst)
                with zipfile.ZipFile(dst, 'r') as zip_ref:
                    zip_ref.extractall(os.path.dirname(dst))
            # download val images
            if self.query_yes_no("Do you want to download zipped validation images [0.8GB] under ./images/val?", default='no'):
                url = 'https://www.dropbox.com/s/9jzzwsaxnwnbdmg/val.zip?dl=1'
                dst = os.path.join(self.root_dir, self.Stimuli_folder, self.val_folder,'val2015r1.zip')
                self.download(url, dst)
                with zipfile.ZipFile(dst, 'r') as zip_ref:
                    zip_ref.extractall(os.path.dirname(dst))
            
            if self.query_yes_no("Do you want to download zipped test images [0.8GB] under ./images/test?", default='no'):
                url = 'https://www.dropbox.com/s/4gzn0hs1tw4ydlu/test.zip?dl=1'
                dst = os.path.join(self.root_dir, self.Stimuli_folder, self.test_folder,'test2015r1.zip')
                self.download(url,dst)
                with zipfile.ZipFile(dst, 'r') as zip_ref:
                                    zip_ref.extractall(os.path.dirname(dst))
                                    
                                    
        results = []
                        
        for split in self.split:    
                        
            if not self.partition:
                im_paths = sorted(glob(os.path.join( self.root_dir, self.Stimuli_folder, split,'*.jpg')) )
                batch = []
                for path in im_paths:
                            
                    im = skimage.io.imread(path)
                    
                    # Make Greyscale pictures into RGB
                    if len(im.shape)==2:
                        im = np.expand_dims(im, axis = -1)
                        im = np.repeat(im, repeats = 3, axis = -1)
                    batch.append(im/np.max(im))  # Normalize image from [0, 255] to [0, 1]
                    
                array = np.asarray(batch)
                results.append(array)
            
            if self.partition:
                
                batch = []
                for PATTERN in self.split_ID_dict[split]:
                    path = sorted(glob(os.path.join( self.root_dir, self.Stimuli_folder, split, '*' + PATTERN + '.jpg')))
                    im = skimage.io.imread(path[0])
                    # Make Greyscale pictures into RGB
                    if len(im.shape)==2:
                        im = np.expand_dims(im, axis = -1)
                        im = np.repeat(im, repeats = 3, axis = -1)
                    batch.append(im/np.max(im))  # Normalize image from [0, 255] to [0, 1]
                
                array = np.asarray(batch)
                
                results.append(array)
    
        
        try:
            return results[0], results[1], results[2]
        
        except IndexError:
            
            try:    
                return results[0], results[1]
        
            except IndexError:
                
                return results[0]
            
                
    def __download_annotations__(self):
        
        for split in self.split:
            if split == 'train':
                split = self.train_folder
            if split == 'val':
                split = self.val_folder
            if split == 'test':
                split = self.test_folder
            self.data_dir = os.path.join(self.root_dir, self.Stimuli_folder, split)
        try:     
            assert os.path.isdir(self.data_dir), 'No such directoroy'
        except AssertionError:
            # create directory folder
                
            if not os.path.exists(os.path.join(self.root_dir, self.FixMaps_folder)):
                print ('creating ../annotations to host annotations in SALICON dataset')
                os.mkdir(os.path.join(self.root_dir, self.Annotation_folder))
                os.mkdir(os.path.join(self.root_dir, self.Annotation_folder, self.train_folder))
                os.mkdir(os.path.join(self.root_dir, self.Annotation_folder, self.val_folder))
                print ('done')
                
            # download annotations
            for split in self.split:
                anno = 'fixations'
                if split == 'train' and anno == 'fixations':
                    size = '818'
                    if self.query_yes_no("Do you want to download %s split for %s annotations [%sMB] under ./annotations?" % (
                            split, anno, size), default='yes'):
                        fname = os.path.join(self.root_dir, self.Annotation_folder, self.train_folder, '%s_%s2015r1.json' % ( anno, split))
                        url = 'https://www.dropbox.com/s/7t2sc4m92hhtzm2/fixations_train2014.json.zip?dl=1'
                        self.download(url, fname)
                        
        
                elif split == 'val' and anno == 'fixations':
                    size = '459'
                    if self.query_yes_no("Do you want to download %s split for %s annotations [%sMB] under ./annotations?" % (
                            split, anno, size), default='yes'):
                        fname = os.path.join(self.root_dir, self.Annotation_folder, self.val_folder,'%s_%s2015r1.json' % (anno, split))
                        url = 'https://www.dropbox.com/s/bgo91bnoqk3m5ja/fixations_val2014.json.zip?dl=1'
                        self.download(url, fname)
        
    def load_fixmaps(self, split):
        """
        args:
            split : List 
                'train' or 'val'. Thetest set doesn't have annotations.
        """
        self.split = split
        
        if not os.path.exists(os.path.join(self.root_dir, self.Annotation_folder)):
                self.__download_annotations__()
        
        
        if not os.path.exists(os.path.join(self.root_dir, self.FixMaps_folder)):
                print ('creating ../annotations to host annotations in SALICON dataset')
                os.mkdir(os.path.join(self.root_dir, self.FixMaps_folder))
                os.mkdir(os.path.join(self.root_dir, self.FixMaps_folder, self.train_folder))
                os.mkdir(os.path.join(self.root_dir, self.FixMaps_folder, self.val_folder))
                
    
        # build images of fixation maps if dst_dir does not contain any.
        for split in self.split:
            
            if not glob(os.path.join(self.root_dir, self.FixMaps_folder,split,'*.jpg')):
            
                if os.listdir(os.path.join(self.root_dir, self.FixMaps_folder,split)):
    
                    anno_file = sorted(glob(os.path.join(self.root_dir, self.Annotation_folder, split,'*.json')))[0] #There should be only one json file in each split folder. 
                    
                    output_dir = os.path.join(self.root_dir, self.FixMaps_folder, split)
                    
                    self.__build_fixmaps__( anno_file, output_dir)
            
           
                    
            
        # build arrays of shape N,H,W,C from JPEG files.
        results = []
        
        for split in self.split:
            
            if not self.partition:
            
                batch = []
                img_paths = sorted(glob(os.path.join(self.root_dir, self.FixMaps_folder, split, '*.jpg')))
                
                for path in img_paths:
                    im = skimage.io.imread(path)
                    batch.append(im/np.max(im)) # Normalize image from [0 255] to [0 1]
                
                results.append(np.asarray(batch))
                
                
            elif self.partition:
            
                batch = []
                for PATTERN in self.split_ID_dict[split]:
                    path = sorted(glob(os.path.join( self.root_dir, self.FixMaps_folder, split, '*' + PATTERN + '.jpg')))
                    im = skimage.io.imread(path[0])                           
                    batch.append(im/np.max(im))  # Normalize image from [0, 255] to [0, 1]
                                        
                results.append(np.asarray(batch))
                
                                                
        try:    
            return results[0], results[1]
    
        except IndexError:
            
            return results[0]
 
                
    def query_yes_no(self, question, default="yes"):
        """Ask a yes/no question via raw_input() and return their answer.
        "question" is a string that is presented to the user.
        "default" is the presumed answer if the user just hits <Enter>.
            It must be "yes" (the default), "no" or None (meaning
            an answer is required of the user).
        The "answer" return value is one of "yes" or "no".
        """
        valid = {"yes": True, "y": True, "ye": True,
                 "no": False, "n": False}
        if default is None:
            prompt = " [y/n] "
        elif default == "yes":
            prompt = " [Y/n] "
        elif default == "no":
            prompt = " [y/N] "
        else:
            raise ValueError("invalid default answer: '%s'" % default)
    
        while True:
            print (question + prompt)
            choice = input().lower()
            if default is not None and choice == '':
                return valid[default]
            elif choice in valid:
                return valid[choice]
            else:
                print(  "Please respond with 'yes' or 'no' (or 'y' or 'n').")

    def download(self, url, dst):
       '''
       command line progress bar tool
       :param url: url link for the file to download
       :param dst: dst file name
       :return:
       '''
       if os.path.exists(dst):
           print('File downloaded')
           return 1
   
       u = urllib.request.urlopen(url)
       f = open(dst, 'wb')
       meta = u.info()
       file_size = int(meta.getheaders("Content-Length")[0])
       print ("Downloading: %s Bytes: %s " % (dst, file_size))
   
       file_size_dl = 0
       block_sz = 8192
       while True:
           buffer = u.read(block_sz)
           if not buffer:
               break
   
           file_size_dl += len(buffer)
           f.write(buffer)
           status = r"%10d  [%3.2f%%] " % (file_size_dl, file_size_dl * 100. / file_size)
           status = status + chr(8) * (len(status) + 1)
           print (status,)
       f.close()       
       
    def __build_fixmaps__(self, in_ann_wo_fixmap, out_ann_w_fixmap):
        """
        build a full self-contained annoation json file with fixations maps inside.
        :param in_ann_wo_fixmap: the original annotation without fixation maps
        :param out_ann_w_fixmap: the path to the output annotation with fixation maps
        """    
        # initialize COCO api for instance annotations
        salicon=SALICON(in_ann_wo_fixmap)
        
        # get all annotation ids from json
        
        AnnIds = salicon.getAnnIds()
        
        # Collect all the Annotations from the same image to a list
        
        while AnnIds :
            first_time = True 
            for _id_ in AnnIds: 
                Anns = salicon.loadAnns(_id_)
                
                if first_time:
                    img_id = Anns[0]['image_id']
                    Anns_per_img = []
                    AnnIds.pop(0)
                    Anns_per_img.append(Anns[0])
                    first_time = False
                   
                elif img_id == Anns[0]['image_id']:
                    AnnIds.pop(0)
                    Anns_per_img.append(Anns[0])
                    
                else:
                    break
            
            
            fixMap = salicon.buildFixMap(Anns_per_img) # fixMap in 2D array
            
            img_filename_first = os.path.basename(in_ann_wo_fixmap).split('.')[0]
            img_filename_second = '%012d' %img_id
            img_filename = img_filename_first + '_fixations_' + img_filename_second + '.jpg' 
                
            # The json file will be in GBs if all trianing fixMaps are converted.
            # Instead, save the fixMaps as JPEG in folder out_ann_w_fixmap
            filename = os.path.join(out_ann_w_fixmap, img_filename)
            skimage.io.imsave(filename, fixMap)
        
#=============================================================================
# Unit Test
#=============================================================================
#%%
if __name__ == "__main__":
    
    import matplotlib.pyplot as plt
    
    
    """ 
    Example 1
    """   
    
    loader = MIT1003()
    
    data, target = loader.load()
    
    img_tensor = loader.load_and_preprocess(sorted(glob(os.path.join('/home/pohsuanh/Desktop/pohsuan/projects/deep gaze/MIT1003/ALLSTIMULI','*.jpeg'))))
    
    for m, image in enumerate(target):
        
        if m <3:
            plt.figure()
            plt.imshow(image.numpy())
            
"""

    root_dir = '/home/pohsuanh/Desktop/pohsuan/projects/deep gaze/SALICON/'
    
    loader = SALICON_2014(root_dir, partition = 1)
    
    img_train, img_val = loader.load_images(split=['train', 'val'])
    
    tar_train, tar_val = loader.load_fixmaps(split=['train', 'val'])
    
    img_train = tf.convert_to_tensor(img_train)
    
    img_val = tf.convert_to_tensor(img_val)
    
    tar_train = tf.convert_to_tensor(tar_train)
    
    tar_val = tf.convert_to_tensor(tar_val)
    
    tar_train = tf.expand_dims(tar_train, axis = -1)
    
    tar_val = tf.expand_dims(tar_val, axis = -1)
    
    augmentations = ['color', 'flip', 'zoom']
    
    # img_train, tar_train = augment(img_train, tar_train, augmentations)
    
    plot_images(img_train,3,3)
    
    plot_images(tar_train,3,3)
    
    train_imgs, train_bias, train_tars = Array2Dataset(img_train, tar_train)
    
    val_imgs, val_bias, val_tars = Array2Dataset(img_val, tar_val)
    
"""    
            
    



