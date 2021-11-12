#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 12 16:38:06 2021

@author: pohsuanh

Custom loss function

We used to train the model indirectly with MSE, KLD, BCE to predict
saliency maps.

However, these metrics is not the final metric that we use to evaluate the goodness

of fit. We use Shuffle_AUC, Judd_AUC, and Borji_AUC to evaluate the sliency prediction

because the predicting fixation map is a binary classification problem.

Therefore, predicting 'ground truth' slaiency maps may introduce artifacts.

(What are the other obvious reasons using AUC loss aside from more direct training ?
L2 loss is obvious FAST. Maybe pretrain with L2 loss, and fine tune with AUC loss?
 )



"""
import tensorflow as tf
import tensorflow_probability as tfp

class Borji_auc_loss(tf.keras.losses.Loss):

    # def __init__(self):
    #     super().__init__()
    #     self.b_auc = []

    def call(self, y_true, y_pred):
        """
        y_true:
            GROUND TRUTHS

        y_pred:
            PREDICTION

        n_splits :
            NUMBER OF THRESHOLDS


        """
        center_bias=None; n_splits=100; n_rep =100

        y_true = tf.cast(y_true, y_pred.dtype)

        saliency_map = y_pred

        fixation_map = y_true # binary mask

        # if not tf.experimental.numpy.any(fixation_map):
        #     print('no fixation to predict. \n')
        #     return

        # elif tf.experimental.numpy.any(fixation_map):
        #     print('fixation exists.\n')

        # if fixation_map.shape.as_list()[0] == None:
        #     print('Place holder of batch size None.')
        #     return

        # if saliency_map.shape != fixation_map.shape:
        #     w = tf.cast(fixation_map.shape[1], tf.int32)
        #     h = tf.cast(fixation_map.shape[2], tf.int32)
        #     saliency_map = tf.image.resize(saliency_map, (w,h))

        # if center_bias.shape != fixation_map.shape:
        #     w = tf.cast(fixation_map.shape[1], tf.int32)
        #     h = tf.cast(fixation_map.shape[2], tf.int32)
        #     center_bias = tf.image.resize(center_bias, (w,h))

        # N_rep = tf.reduce_sum(center_bias) # number of fix from centerbias
        print('shape:',fixation_map.shape.as_list())
        F = tf.reshape(fixation_map, [8, -1]) #F
        S = tf.reshape(saliency_map, [8, -1]) #Y
        S_fix = tf.boolean_mask(saliency_map,fixation_map) # Saliency map values at fixation locations
        n_fix = tf.cast(tf.shape(S_fix)[0], tf.int64)
        n_pixels = tf.cast(tf.shape(S)[0], tf.int64)
        s_neg = tf.experimental.numpy.random.randint(0, n_pixels, size = [n_fix, n_rep]) # negative samples
        S_rand = tf.gather(tf.reshape(saliency_map,(8,-1)),s_neg) # Saliency map values at random locations (including fixated locations!? underestimated)

        for rep in range(n_rep):
            s_max = tf.math.reduce_max(tf.concat([S_fix, S_rand[:,rep]],axis=0),axis=0)
            thresholds = tf.cast(tf.linspace(0, s_max, n_splits)[::-1],tf.float32)
            S_neg = tf.SparseTensor(indices =s_neg[:,rep], values = [1]*n_fix, dense_shape = [n_fix]) # CB. from index to binary mask
            S_neg = tf.sparse.to_dense(S_neg) # from sparse tensor to tensor

            trapz = 0
            for k in tf.range(len(thresholds)-1):
                thres0 = tf.ones([n_fix])*thresholds[k]
                thres1 = tf.ones([n_fix])*thresholds[k+1]
                FPR0 = tf.math.reduce_sum(tf.math.sigmoid(S_neg*S-thres0))/n_fix
                FPR1 = tf.math.reduce_sum(tf.math.sigmoid(S_neg*S-thres1))/n_fix
                TPR0 = tf.math.reduce_sum(tf.math.sigmoid(F*S-thres0))/n_fix
                TPR1 = tf.math.reduce_sum(tf.math.sigmoid(F*S-thres1))/n_fix
                ### The formula on paper here #
                trapz = trapz + (FPR1-FPR0)*(TPR1+TPR0)/2

            self.b_auc.append(trapz)

        return -tf.math.reduce_mean(self.b_auc)

def borji_auc_loss_fn_wrong(y_true, y_pred):
        print('==============')
        print('y_true', y_true.shape.as_list())
        print('y_pred', y_pred.shape.as_list())
        print('==============')
        b_auc = []
        center_bias = None
        n_splits = 100
        n_rep = 100

        y_true = tf.cast(y_true, y_pred.dtype)

        saliency_map = y_pred

        fixation_map = y_true # binary mask

        if  tf.math.reduce_sum(fixation_map) == 0:
            print('no fixation to predict. \n')
            return 0.0

        if fixation_map.shape.as_list()[0] == None:
            print('Place holder of batch size None.')
            return 0.0

        if saliency_map.shape != fixation_map.shape:
            w = tf.cast(fixation_map.shape[1], tf.int32)
            h = tf.cast(fixation_map.shape[2], tf.int32)
            saliency_map = tf.image.resize(saliency_map, (w,h))

        if center_bias.shape != fixation_map.shape:
            w = tf.cast(fixation_map.shape[1], tf.int32)
            h = tf.cast(fixation_map.shape[2], tf.int32)
            center_bias = tf.image.resize(center_bias, (w,h))



        F = tf.reshape(fixation_map, [-1, 480*640]) #F
        S = tf.reshape(saliency_map, [-1, 480*640]) #Y
        print('=================')
        print('F', F.shape.as_list())
        print('S', S.shape.as_list())
        print('=================')

        S_fix = tf.boolean_mask( saliency_map, fixation_map) # Saliency map values at fixation locations
        n_fix = tf.cast(tf.shape(S_fix)[0], tf.int64)
        print('=================')
        print('S_fix', S_fix.shape.as_list())
        print('n_fix', n_fix.shape.as_list())
        print('=================')

        n_pixels = tf.cast(tf.shape(S)[0], tf.int64)
        s_neg = tf.experimental.numpy.random.randint(0, n_pixels, size = [n_fix, n_rep]) # negative samples
        S_rand = tf.gather(tf.reshape(saliency_map,(8,-1)),s_neg) # Saliency map values at random locations (including fixated locations!? underestimated)

        for rep in range(n_rep):
            s_max = tf.math.reduce_max(tf.concat([S_fix, S_rand[:,rep]],axis=0),axis=0)
            thresholds = tf.cast(tf.linspace(0, s_max, n_splits)[::-1],tf.float32)
            S_neg = tf.SparseTensor(indices =s_neg[:,rep], values = [1]*n_fix, dense_shape = [n_fix]) # CB. from index to binary mask
            S_neg = tf.sparse.to_dense(S_neg) # from sparse tensor to tensor

            trapz = 0
            for k in tf.range(len(thresholds)-1):
                thres0 = tf.ones([n_fix])*thresholds[k]
                thres1 = tf.ones([n_fix])*thresholds[k+1]
                FPR0 = tf.math.reduce_sum(tf.math.sigmoid(S_neg*S-thres0))/n_fix
                FPR1 = tf.math.reduce_sum(tf.math.sigmoid(S_neg*S-thres1))/n_fix
                TPR0 = tf.math.reduce_sum(tf.math.sigmoid(F*S-thres0))/n_fix
                TPR1 = tf.math.reduce_sum(tf.math.sigmoid(F*S-thres1))/n_fix
                ### The formula on paper here #
                trapz = trapz + tf.math.abs((FPR1-FPR0))*(TPR1+TPR0)/2

            b_auc.append(1-trapz)

        return -1*tf.math.reduce_mean(b_auc)

def random_choice( n_rep, n_pixels, n_fix, replacement = False):
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

def borji_auc_loss_fn(y_true, y_pred):
        # print('==============')
        # print('y_true', y_true.shape.as_list())
        # print('y_pred', y_pred.shape.as_list())
        # print('==============')
        b_auc = []
        n_split = 21
        n_rep = 10

        y_true = tf.cast(y_true, y_pred.dtype)

        saliency_map = y_pred

        fixation_map = y_true # binary mask

        if  tf.math.reduce_sum(fixation_map) == 0:
            print('no fixation to predict. \n')
            return 0.0

        if fixation_map.shape.as_list()[0] == None:
            print('Place holder of batch size None.')
            return 0.0

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
            rand_ = random_choice(n_rep, n_pixels, n_fix[i]) # indices
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

        # S_thres = tf.cast( tf.math.greater_equal(tf.sign( tf.math.subtract(S,thresholds)), 0), tf.float32)
        S_thres = tf.cast( tf.math.sigmoid(tf.math.multiply(5,tf.math.subtract(S,thresholds))), tf.float32)

        S_fix = tf.math.multiply(F, S_thres) # Fixation points corrected predicted by saliency map
        tp =  tf.math.divide_no_nan(tf.reduce_sum(tf.cast(S_fix, tf.float32), axis = -1), tf.cast(N_fix, tf.float32))
        S_rand = tf.math.multiply(rand_F, S_thres)
        fp = tf.math.divide_no_nan(tf.reduce_sum(tf.cast(S_rand, tf.float32),axis = -1),tf.cast(N_fix, tf.float32))

        area = tf.math.abs(tfp.math.trapz(tp, fp, axis =-1))#[batch, n_rep]

        auc = tf.reduce_mean(area, axis =-1) # [batch]

        return tf.math.reduce_mean(auc)