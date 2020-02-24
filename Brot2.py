# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 13:04:13 2020

@author: Peter
"""

import mpmath as mp
import numpy as np
import matplotlib.pyplot as plt

res = [1920, 1080]
base_factor = 2.

def brot(c, depth=200):
    z = complex(0)

    for i in range(depth):
        z = pow(z, 2) + c

        if abs(z) > 2:
            return i

    return -1

def brot_gen(re_lim, im_lim):
    re_span = np.linspace(re_lim[0], re_lim[1], res[0])
    im_span = np.linspace(im_lim[0], im_lim[1], res[1])
    
    mset = np.zeros([res[1], res[0]])
    
    for re in range(len(re_span)):
        for im in range(len(im_span)):
            mset[im][re] = brot(complex(re_span[re], im_span[im]))
    
    return mset

def handler(ax):
    def action(event):
        if event.button == 2:
            cur_re_lim = ax.get_xlim()
            cur_im_lim = ax.get_ylim()
        
            mset = brot_gen(cur_re_lim, cur_im_lim)
            
            ax.cla()
            ax.imshow(mset, extent=[cur_re_lim[0], cur_re_lim[1], cur_im_lim[0], cur_im_lim[1]], origin="lower", vmin=0, vmax=200, interpolation="bilinear")
            
            plt.draw()
    
    fig = ax.get_figure()
    fig.canvas.mpl_connect('button_release_event', action)
    
    return action


re_lim = np.array([-2.5, 2.5])
im_lim = res[1]/res[0] * re_lim

plt.imshow(brot_gen(re_lim, im_lim), extent=[re_lim[0], re_lim[1], im_lim[0], im_lim[1]], origin="lower", vmin=0, vmax=200, interpolation="bilinear")

ax = plt.gca()

f = handler(ax)

plt.show()