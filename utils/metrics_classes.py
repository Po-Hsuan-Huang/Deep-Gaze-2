#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 29 00:28:33 2021

@author: pohsuanh

metrics classes

tensorflow keras custom metrics
"""
import numpy as np
import tensorflow as tf
from skimage.transform import resize
from utils.evaluation.tools import normalize, match_hist

class AUC_Borji(tf.keras.metrics.Metric):

    def __init__(self, name = 'AUC_Borji', **kwargs):
        super(AUC_Borji, self).__init__(name=name, **kwargs)

    def update_state(self, fixation_map, saliency_map, n_rep=100, step_size=0.1, rand_sampler=None, **kwargs):
        fixation_map = tf.math.greater(fixation_map, 0.5)
    	# If there are no fixation to predict, return NaN
        if not tf.experimental.numpy.any(fixation_map):
            print('no fixation to predict')
            return np.nan
    	# Make the saliency_map the size of the fixation_map
        if saliency_map.shape != fixation_map.shape:
            saliency_map = tf.image.resize(saliency_map, fixation_map.shape, method='nearest')
    	# Normalize saliency map to have values between [0,1]
        saliency_map = normalize(saliency_map, method='range')

        S = tf.reshape(saliency_map, [-1])
        F = tf.reshape(fixation_map, [-1])
        S_fix = tf.boolean_mask(S,F) # Saliency map values at fixation locations
        n_fix = S_fix.shape[0]
        n_pixels = S.shape[0]
    	# For each fixation, sample n_rep values from anywhere on the saliency map
        if rand_sampler is None:
            r = np.random.randint(0, n_pixels, [n_fix, n_rep])
            S_rand = tf.boolean_mask(S,r) # Saliency map values at random locations (including fixated locations!? underestimated)
        else:
            S_rand = rand_sampler(S, F, n_rep, n_fix)
    	# Calculate AUC per random split (set of random locations)
        auc = np.zeros(n_rep) * np.nan
        for rep in range(n_rep):
            thresholds = np.r_[0:np.max(np.r_[S_fix, S_rand[:,rep]]):step_size][::-1]
            tp = np.zeros(len(thresholds)+2)
            fp = np.zeros(len(thresholds)+2)
            tp[0] = 0; tp[-1] = 1
            fp[0] = 0; fp[-1] = 1
            for k, thresh in enumerate(thresholds):
                tp[k+1] = np.sum(S_fix >= thresh) / float(n_fix)
                fp[k+1] = np.sum(S_rand[:,rep] >= thresh) / float(n_fix)
            auc[rep] = np.trapz(tp, fp)
        self.auc_borji = np.mean(auc) # Average across random splits

    def result(self):
        return self.auc_borji

    def reset_states(self):
        self.auc_borji.assign(0.0)