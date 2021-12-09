#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 12 16:38:06 2021

@author: pohsuanh

Custom loss function


Training with AUC loss

Abstract :

We used to train the model indirectly with MSE, KLD, BCE to predict
saliency maps.However, these metrics is not the final metric that we use to evaluate the goodness
of fit. We use Shuffle_AUC, Judd_AUC, and Borji_AUC to evaluate the sliency prediction
because the predicting fixation map is a binary classification problem.
Therefore, predicting 'ground truth' slaiency maps may introduce artifacts.
(What are the other obvious reasons using AUC loss aside from more direct training ?
L2 loss is obvious FAST. Maybe pretrain with L2 loss, and fine tune with AUC loss?
 )



Shuffle AUC Loss:

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


READ BEFORE USE.

    The function differs from the original evaluation metrics where each predicted
    fixatiom maps is compaired to a set of real fixation maps derived from other images.
    The reason is in the evaluation metrics everything is computed in Numpy. In
    Numpy, it is easy to add other fixation maps as the thrid arguent.

    During training, we can't add the third argument to the Tensor Graph because
    we have other loss functions that doesn't need the third argument.
    That is, we can't dynamically change the tensor graph during training.
    Tensorflow is intensionally designed this way to optimize training and deployment.

    In tensorflow 2.0, eager execution in enabled but customized training loop
    will break the high level Keras API that I am using for easy readibility.

    Instead of feeding other fixation maps tensors by modifying Tensor Graph,
    I use the other fixation maps from the same training batch. That is, the
    Shuffle AUC acccuracies is not very high because of small batch size.
    Fortunately, it doesn't have to be very high as stochastic gradient decent with
    minibatch. Another benifit of sacrificing the accuracy is to fasciitate the
    calculation of the loss function. You can say it is a featrue, not a bug.

"""
import tensorflow as tf
import tensorflow_probability as tfp

def shuffled_auc_loss_fn(y_true, y_pred):

        if  tf.math.reduce_sum(y_true) == 0:
            print('no fixation to predict. \n')
            return 0.0

        if y_true.shape.as_list()[0] == None:
            print('Place holder of batch size None.')
            return 0.0

        if y_pred.shape != y_true.shape: # shrink to the smaller size of the two when size mismatch.
            if y_pred.shape[1]*y_pred.shape[2] < y_true.shape[1]*y_true.shape[2]:
                y_true = tf.transpose(y_true, [1,2,0])
                y_true = tf.image.resize(y_true, ( y_pred.shape[1], y_pred.shape[2]), method='nearest')
                y_true = tf.transpose(y_true, [2,0,1])
            else:
                y_pred = tf.transpose(y_pred, [1,2,0])
                y_pred = tf.image.resize(y_pred, ( y_true.shape[1], y_true.shape[2]), method='nearest')
                y_pred = tf.transpose(y_pred, [2,0,1])

        n_split = 21
        n_rep = 10
        Batch_Size = tf.shape(y_true)[0]

        saliency_map = y_pred
        fixation_map = tf.cast(y_true, y_pred.dtype)

        other_fmap = tf.cast(y_true, y_pred.dtype)
        other_fmap = tf.reshape(other_fmap, [Batch_Size, -1])



        if y_true == None or y_pred ==None :
            print('invalid input. y_true: {}, y_pred: {}'.format(y_true, y_pred))

        if  tf.math.reduce_sum(fixation_map) == 0:
            print('no fixation to predict. \n')
            return 0.0

        if fixation_map.shape.as_list()[0] == None:
            print('Place holder of batch size None.')
            return 0.0

        # For each fixation, sample n_rep values (from fixated locations on other_map) on the saliency map
        def sample_other(other_fmap, S, F, n_rep, n_fix):
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

        # Normalize saliency map to have values between [0,1]
        # saliency_map = (saliency_map - tf.reduce_min(saliency_map))/(tf.reduce_max(saliency_map)-tf.reduce_min(saliency_map))

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

            sample_indices = sample_other(other_fmap[:-1], S, F, n_rep, n_fix[i])
            n_fixation = sample_indices.shape[1]
            axis_0 = tf.reshape(tf.repeat(tf.range(n_rep),n_fixation),(n_rep, n_fixation)) # first coordinate of the indices
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

        # Can't directly use greater equal threshold as in evaluation
        # otherwise the graudient vanishes.
        # S_thres = tf.cast( tf.math.greater_equal(tf.sign( tf.math.subtract(S,thresholds)), 0), tf.float32)

        S_thres = tf.cast( tf.math.sigmoid(tf.math.multiply(5,tf.math.subtract(S,thresholds))), tf.float32)

        S_fix = tf.math.multiply(F, S_thres) # Fixation points corrected predicted by saliency map


        tp =  tf.math.divide_no_nan(tf.reduce_sum(tf.cast(S_fix, tf.float32), axis = -1), tf.cast(N_fix, tf.float32))
        S_rand = tf.math.multiply(rand_F, S_thres)
        fp = tf.math.divide_no_nan(tf.reduce_sum(tf.cast(S_rand, tf.float32),axis = -1),tf.cast(N_fix, tf.float32))

        area = tf.math.abs(tfp.math.trapz(tp, fp, axis =-1))#[batch, n_rep]

        auc = tf.reduce_mean(area, axis =-1) # [batch]

        return -tf.math.reduce_mean(auc)