#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 29 00:28:33 2021

Based on https://github.com/cvzoya/saliency/blob/master/code_forMetrics/

metrics classes

tensorflow keras custom metrics
"""
import numpy as np
import tensorflow as tf
from skimage.transform import resize
from utils.evaluation.tools import normalize, match_hist
import tensorflow_probability as tfp
import tensorflow.python.util.deprecation as deprecation


class AUC_Borji_v1(tf.keras.metrics.Metric):
    """
    Non-parallelized version. For-loops are usually not allowed in tensor graphs.
    Replaced them with Tensor multiplications.
    """
    deprecation.deprecated('2023-12-31', 'Use AUC_Borji_v2.')

    def __init__(self, name = 'AUC_Borji',  n_rep= 10, n_split= 21, rand_sampler= None, **kwargs):
        super(AUC_Borji_v1, self).__init__(name=name, **kwargs)
        self.auc_borji = self.add_weight(name="auc_borji", initializer="zeros")

        self.auc = self.add_weight(name='_auc_', shape=n_rep, initializer="zeros")
        self.tp = self.add_weight(name='_tp_', shape = n_split, initializer="zeros")
        self.fp = self.add_weight(name='_fp_', shape = n_split, initializer="zeros")
        self.rand_sampler = rand_sampler
        self.n_rep = n_rep
        self.n_split = n_split


    def update_state(self, fixation_map, saliency_map, n_rep = 10, n_split = 21, rand_sampler=None, **kwargs):
        """
        Parameters
        ----------
        fixation_map : Tensor ( Batch, H, W)
            DESCRIPTION.
        saliency_map : Tensor ( Batch, H, W)
            DESCRIPTION.
        n_rep : integer
            Number of times sampling of negative examples. The default is 10.
        n_split : integer
            Number of thresholds. The default is 21.
        rand_sampler : TYPE, optional
            Distribution to sample the negative exmapels. The default is None.
        **kwargs : TYPE
            DESCRIPTION.

        Returns
        -------
        None

        """
    	# If there are no fixation to predict, return NaN
        # if not tf.experimental.numpy.any(fixation_map):
        #     print('no fixation to predict')
        #     return  np.nan
        # if fixation_map.shape[0] == None:
        #     print('Place holder tensor.')
        #     return np.nan

    	# Make the saliency_map the size of the fixation_map
        if saliency_map.shape != fixation_map.shape:
            saliency_map = tf.image.resize(saliency_map, (fixation_map.shape[1], fixation_map.shape[2]), method='nearest')

    	# Normalize saliency map to have values between [0,1]
        saliency_map = (saliency_map - tf.reduce_min(saliency_map))/(tf.reduce_max(saliency_map)-tf.reduce_min(saliency_map))

        Batch_Size = fixation_map.shape[0]


        self.mean_auc_borji = self.add_weight(name="mean_auc_borji", initializer="zeros")

        for b in range(Batch_Size):
            S = tf.cast(tf.reshape(saliency_map[b], [-1]), dtype =tf.float32)
            F = tf.cast(tf.reshape(fixation_map[b], [-1]), dtype = tf.float32)
            n_fix = tf.cast(tf.shape(F)[0], tf.int64)
            n_pixels = tf.cast(tf.shape(S)[0], tf.int64)
        	# Calculate AUC per random split (set of random locations)
            # self.auc = tf.Variable(tf.zeros(n_rep))
            if rand_sampler is None:
                rand_F = tf.experimental.numpy.random.randint(0, high = n_pixels, size = [n_fix, n_rep])

            else:
                rand_F = rand_sampler(S, F, n_rep, n_fix)

            for rep in range(n_rep):

                thresholds = tf.cast(tf.linspace(0, 1, n_split)[::-1], S.dtype)

                for k in tf.range(len(thresholds)):
                    thresh = thresholds[k]
                    S_thres = tf.sign(S -thresh) >= 0
                    S_fix = tf.boolean_mask(F, S_thres) # Fixation points corrected predicted by saliency map
                    self.tp[k].assign(tf.reduce_sum(tf.cast(S_fix, tf.float32)) / float(n_fix))
                    S_rand = tf.boolean_mask(rand_F[:,rep], S_thres)
                    self.fp[k].assign(tf.reduce_sum(tf.cast(S_rand, tf.float32)) / float(n_fix))

                area = 0.0
                extra_area = 0.0
                for i in range(n_split-1):
                    area = area  + 0.5*(self.tp[i+1] + self.tp[i])*(tf.math.abs(self.fp[i]-self.fp[i+1]))

                # If the right end of x axis of AUC is not 1,
                # then the extra unsampled area under the curve must also be taken into account.
                if self.fp[0] < 1 :
                    extra_area = self.tp[0]*(1 - self.fp[0])

                auc = area+extra_area

                self.auc[rep].assign(auc)
            self.auc_borji[b].assign(tf.math.reduce_mean(self.auc)) # Average across random splits

        self.mean_auc_borji.assign(tf.math.reduce_mean(self.auc_borji))
        return self.mean_auc_borji

    def result(self):
        return self.auc_borji


    # def result(self):

    def reset_state(self):
        self.auc_borji.assign(0.0)
        self.auc.assign(tf.zeros(self.n_rep))
        self.tp.assign(tf.zeros(self.n_split))
        self.fp.assign(tf.zeros(self.n_split))


class AUC_Borji_v2(tf.keras.metrics.Metric):
    """
    Parallelized calculation for efficiency.

    """

    def __init__(self, name = 'AUC_Borji',  n_rep= 10, n_split= 21, replica_in_sync = 1,  **kwargs):
        super(AUC_Borji_v2, self ).__init__(name=name,**kwargs)
        self.auc_borji = self.add_weight(name="auc_borji", initializer="zeros")
        self.n_rep = n_rep
        self.n_split = n_split
        self.replica_in_sync = replica_in_sync


    def random_choice(self, n_rep, n_pixels, n_fix, replacement = False):
            # Sampling without replacement using Gumble matrix trick.
            # Sampling with replacement is tf.ranodm.categorical
            # Params:
            #    n_rep:  size of N-1 dimension
            #    n_pixels : maxvalue
            #    n_fix: size of sampling in the last dimension
            #Return:
            #    Tensor : shape [n_rep, n_fix]
            # 1. produce p, which can be nonnormalized distribution.
            p = tf.divide(tf.ones([n_rep, n_pixels]), tf.cast(n_pixels, tf.float32)) # [n_pixels]
            # 2. Gumble matrix
            z = -tf.math.log(-tf.math.log(tf.random.uniform(tf.shape(p),0,1)))
            _, indices = tf.nn.top_k(tf.math.log(p) + z, n_fix)
            return indices

    def update_state(self, fixation_map, saliency_map, n_rep = 10, n_split = 21,**kwargs):
        """
        Parameters
        ----------
        fixation_map : Tensor ( Batch, H, W)
            DESCRIPTION.
        saliency_map : Tensor ( Batch, H, W)
            DESCRIPTION.
        n_rep : integer
            Number of times sampling of negative examples. The default is 10.
        n_split : integer
            Number of thresholds. The default is 21.
        rand_sampler : TYPE, optional
            Distribution to sample the negative exmapels. The default is None.
        **kwargs : TYPE
            DESCRIPTION.

        Returns
        -------
        None

        """
    	# If there are no fixation to predict, return NaN
        # if not tf.experimental.numpy.any(fixation_map):
        #     print('no fixation to predict')
        #     return  np.nan
        # if fixation_map.shape[0] == None:
        #     print('Place holder tensor.')
        #     return np.nan

    	# Make the saliency_map the size of the fixation_map
        if saliency_map.shape != fixation_map.shape:
            saliency_map = tf.image.resize(saliency_map, (fixation_map.shape[1], fixation_map.shape[2]), method='nearest')

    	# Normalize saliency map to have values between [0,1]
        # saliency_map = (saliency_map - tf.reduce_min(saliency_map))/(tf.reduce_max(saliency_map)-tf.reduce_min(saliency_map))
        Batch_Size = fixation_map.shape[0]

        S = tf.cast(tf.reshape(saliency_map, [Batch_Size, -1]), dtype =tf.float32)
        S = tf.expand_dims(S, axis =1)
        S = tf.repeat(S, n_rep, axis = 1)
        S = tf.expand_dims(S, axis = 2)
        S = tf.repeat(S, n_split, axis = 2) # [Batch, n_rep, n_split, n_pixels]

        F = tf.cast(tf.reshape(fixation_map, [Batch_Size, -1]), dtype = tf.float32)
        n_fix = tf.cast(tf.reduce_sum( F, axis = -1), tf.int32)#[Batch]
        F = tf.expand_dims(F, axis = 1)
        F = tf.repeat(F, n_rep, axis = 1)
        F = tf.expand_dims(F, axis = 2)
        F = tf.repeat(F, n_split, axis = 2) # [Batch, n_rep, n_split, n_pixels]

        n_pixels = tf.constant(tf.shape(S)[3]).numpy() #[n_pixels]

        rand_F = tf.Variable(tf.zeros([Batch_Size, n_rep, n_pixels, 1]))

        for i in range(Batch_Size):
            # Create negative fixation map for each batch
            # Consider vectorize it for efficiency?

            # rand_F = tf.experimental.numpy.random.randint(0, high = n_pixels, size = [n_rep, n_fix]) # indices
            rand_ = self.random_choice(n_rep, n_pixels, n_fix[i]) # indices
            base = tf.reshape(tf.repeat(tf.range(n_rep),n_fix[i]),(n_rep, n_fix[i]))
            indices = tf.stack([base, rand_], axis = -1)
            updates = tf.ones([n_rep, n_fix[i], 1])
            shape = tf.constant([n_rep, n_pixels, 1])
            rand_f = tf.scatter_nd(indices, updates, shape) # insert ones to the shape
            rand_F[i,:].assign(rand_f)

        rand_F =tf.expand_dims(rand_F, axis = 2)
        rand_F = tf.squeeze(tf.repeat(rand_F, n_split, axis = 2)) #[Batch, n_rep, n_split, n_pixels]

        thresholds = tf.cast(tf.linspace(0, 1, n_split)[::-1], S.dtype)
        thresholds = tf.expand_dims(thresholds, 0)
        thresholds = tf.repeat(thresholds, Batch_Size, axis = 0 )
        thresholds = tf.expand_dims(thresholds, 1)
        thresholds = tf.repeat(thresholds, n_rep, axis = 1)
        thresholds = tf.expand_dims(thresholds, -1)
        thresholds = tf.repeat(thresholds,  n_pixels, axis = -1 )# [Batch, n_rep, n_split, n_pixels]

        # n_fix is not broadcastable in tf.math.divide()
        N_fix = tf.expand_dims(n_fix, axis = -1)
        N_fix = tf.repeat(N_fix, n_rep, axis = -1)
        N_fix = tf.expand_dims(N_fix, axis = -1)
        N_fix = tf.repeat(N_fix, n_split, axis = -1)

        S_thres = tf.cast( tf.math.greater_equal(tf.sign( tf.math.subtract(S,thresholds)), 0), tf.float32)
        S_fix = tf.math.multiply(F, S_thres) # Fixation points corrected predicted by saliency map
        tp =  tf.math.divide_no_nan(tf.reduce_sum(tf.cast(S_fix, tf.float32), axis = -1), tf.cast(N_fix, tf.float32))
        S_rand = tf.math.multiply(rand_F, S_thres)
        fp = tf.math.divide_no_nan(tf.reduce_sum(tf.cast(S_rand, tf.float32),axis = -1),tf.cast(N_fix, tf.float32))

        area = tf.math.abs(tfp.math.trapz(tp, fp, axis =-1))#[batch, n_rep]

        auc = tf.reduce_mean(area, axis =-1) # [batch]

        self.auc_borji.assign(tf.math.reduce_mean(auc)/ self.replica_in_sync) # Average across random splits

        return self.auc_borji

    def result(self):
        return self.auc_borji


    # def result(self):

    def reset_state(self):
        self.auc_borji.assign(0.0)