#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  9 16:37:41 2021

@author: pohsuanh

Image Data Smoothing Module

"""

import numpy as np

def Gaussian_kernel(l=5, sig =1):
    """\
    creates gaussian kernel with side length l and a sigma of sig
    https://stackoverflow.com/questions/29731726/how-to-calculate-a-gaussian-kernel-matrix-efficiently-in-numpy
    """

    ax = np.linspace(-(l - 1) / 2., (l - 1) / 2., l, dtype = np.float32)
    xx, yy = np.meshgrid(ax, ax)

    kernel = np.exp(-0.5 * (np.square(xx) + np.square(yy)) / np.square(sig))

    return kernel / np.sum(kernel)



