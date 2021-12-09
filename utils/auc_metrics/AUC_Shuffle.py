#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  8 13:04:07 2021

@author: pohsuanh
"""
import tensorflow as tf
import tensorflow_probability as tfp

class AUC_Shuffle(tf.keras.metrics.Metric):
    """
    Parallelized calculation for efficiency.


    This measures how well the saliency map of an image predicts the ground truth human fixations on the image.
	ROC curve created by sweeping through threshold values at fixed step size
	until the maximum saliency map value.
	True positive (tp) rate correspond to the ratio of saliency map values above threshold
	at fixation locations to the total number of fixation locations.
	False positive (fp) rate correspond to the ratio of saliency map values above threshold
	at random locations to the total number of random locations
	(as many random locations as fixations, sampled uniformly from fixation_map ON OTHER IMAGES),
	averaging over n_rep number of selections of random locations.

    Fixation maps on other images are ALL the other fixation maps of the same batch.

    """

    def __init__(self, name = 'AUC_Shuffle',  n_rep= 10, n_split= 21, replica_in_sync = 1,  **kwargs):
        super(AUC_Shuffle, self ).__init__(name=name,**kwargs)
        self.auc_shuffle = self.add_weight(name="auc_borji", initializer="zeros")
        self.n_rep = n_rep
        self.n_split = n_split
        self.replica_in_sync = replica_in_sync


    def sample_other(self, other_fmap, S, F, n_rep, n_fix):
            fixated = tf.experimental.numpy.nonzero(other_fmap)[1]
            if len(fixated) < n_fix:
                n_fix = len(fixated)
            a =tf.expand_dims(tf.range(len(fixated)),0)
            b = tf.constant([n_rep, 1], tf.int32)
            c = tf.tile(a,b)
            indexer = tf.map_fn(lambda x: tf.random.shuffle(x)[:n_fix], c)
            mask = tf.transpose(indexer)
            r = tf.cast(tf.gather(fixated, mask), tf.int32) # indices
            r = tf.transpose(r)
            return r

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
        if fixation_map == None or saliency_map ==None :
            print('invalid input. y_true: {}, y_pred: {}'.format(fixation_map, saliency_map))

        if  tf.math.reduce_sum(fixation_map) == 0:
            print('no fixation to predict. \n')
            return 0.0

        if fixation_map.shape.as_list()[0] == None:
            print('Place holder of batch size None.')
            return 0.0

        if saliency_map.shape != fixation_map.shape: # shrink to the smaller size of the two when size mismatch.
            if saliency_map.shape[1]*saliency_map.shape[2] < fixation_map.shape[1]*fixation_map.shape[2]:
                fixation_map = tf.transpose(fixation_map, [1,2,0])
                fixation_map = tf.image.resize(fixation_map, ( saliency_map.shape[1], saliency_map.shape[2]), method='nearest')
                fixation_map = tf.transpose(fixation_map, [2,0,1])
            else:
                saliency_map = tf.transpose(saliency_map, [1,2,0])
                saliency_map = tf.image.resize(saliency_map, ( fixation_map.shape[1], fixation_map.shape[2]), method='nearest')
                saliency_map = tf.transpose(saliency_map, [2,0,1])

        Batch_Size = tf.shape(fixation_map)[0]

        other_fmap = tf.cast(fixation_map, saliency_map.dtype)
        other_fmap = tf.reshape(other_fmap, [Batch_Size, -1])

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
        F = tf.repeat(F, n_split, axis = 2) # [Batch, n_rep, n_split,]

        n_pixels = tf.constant(tf.shape(S)[3]).numpy() #[n_pixels]

        rand_F = tf.Variable(tf.zeros([Batch_Size, n_rep, n_pixels, 1]))

        for i in range(Batch_Size):
            # Other fixation maps are the rest of the batch
            other_fmap = tf.roll(other_fmap, shift = -1, axis = 0) # permute the ground truths to create other maps.

            sample_indices = self.sample_other(other_fmap[:-1], S, F, n_rep, n_fix[i])
            n_fixation = sample_indices.shape[1]
            axis_0 = tf.reshape(tf.repeat(tf.range(n_rep), n_fixation),(n_rep, n_fixation)) # first coordinate of the indices
            indices = tf.stack([axis_0, sample_indices], axis = -1)
            updates = tf.ones([n_rep, n_fixation, 1])
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
        # S_thres = tf.cast( tf.math.sigmoid(tf.math.multiply(5,tf.math.subtract(S,thresholds))), tf.float32)

        S_fix = tf.math.multiply(F, S_thres) # Fixation points corrected predicted by saliency map
        tp =  tf.math.divide_no_nan(tf.reduce_sum(tf.cast(S_fix, tf.float32), axis = -1), tf.cast(N_fix, tf.float32))
        S_rand = tf.math.multiply(rand_F, S_thres)
        fp = tf.math.divide_no_nan(tf.reduce_sum(tf.cast(S_rand, tf.float32),axis = -1),tf.cast(N_fix, tf.float32))

        area = tf.math.abs(tfp.math.trapz(tp, fp, axis =-1))#[batch, n_rep]

        auc = tf.reduce_mean(area, axis =-1) # [batch]

        self.auc_shuffle.assign(tf.math.reduce_mean(auc)/ self.replica_in_sync) # Average across random splits

        return self.auc_shuffle

    def result(self):
        return self.auc_shuffle

    # def result(self):

    def reset_state(self):
        self.auc_shuffle.assign(0.0)