# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 13:04:13 2020

@author: Peter
"""

import mpmath
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import multiprocessing as mp
import time

res = [640, 480]
base_factor = 2.
cpuNum = mp.cpu_count() - 1
colmap = mpl.cm.get_cmap('rainbow')
colmap.set_under('k')
mpmath.mp.dps = 4

def close(event):
    exit()

def brot(c, depth=200, eps=0.001):
    z = c
    dz = 1
    epsSq = eps * eps

    for i in range(depth):
        dz = dz * z * 2
        z = mpmath.power(z, 2) + c

        if mpmath.power(mpmath.fabs(z), 2) > 4:
            return i
        if mpmath.power(mpmath.fabs(dz), 2) < epsSq:
            return -1

    return -1

def brot_gen(span, depth):
    re_span = span[0]
    im_span = span[1]
    
    mset = np.zeros([len(im_span), len(re_span)])
    
    for re in range(len(re_span)):
        for im in range(len(im_span)):
            mset[im][re] = brot(mpmath.mpc(re_span[re], im_span[im]), depth)
    
    return mset

def brot_gen_parallel(re_lim, im_lim, depth):
    re_span = mpmath.linspace(re_lim[0], re_lim[1], res[0])
    im_span = mpmath.linspace(im_lim[0], im_lim[1], res[1])
    
    start = time.time()
    split_re_span = np.array_split(re_span, cpuNum)
    
    packages = [(sec, im_span) for sec in split_re_span]
    print("Generating set between", re_lim, "and", im_lim, "at depth", depth, "with", cpuNum, "processes...")
    
    pool = mp.Pool(cpuNum)
    
    results = pool.starmap(brot_gen, [(package, depth) for package in packages])
    
    pool.close()

    mset = np.concatenate(list(results), axis=1)
    
    print("Set generated in", time.time() - start)
    
    return mset

class Generator:
    def __init__(self, ax, re_lim, im_lim):
         self.axes = ax
         self.re_lim = re_lim
         self.im_lim = im_lim
         self.canvas = ax.get_figure().canvas
         self.cid = self.canvas.mpl_connect('button_press_event', self)
         self.cidScroll = self.canvas.mpl_connect('scroll_event', self)
            
    def __call__(self, event):
        if event.button == 2:
            new_x_lim = self.axes.get_xlim()
            re_span = self.re_lim[1] - self.re_lim[0]
            new_re_lim = (new_x_lim[0] / res[0] * re_span + self.re_lim[0], 
                          new_x_lim[1] / res[0] * re_span + self.re_lim[0])
            
            new_y_lim = self.axes.get_ylim()
            im_span = self.im_lim[1] - self.im_lim[0]
            new_im_lim = (new_y_lim[0] / res[1] * im_span + self.im_lim[0],
                          new_y_lim[1] / res[1] * im_span + self.im_lim[0])
            
            new_re_span = new_re_lim[1] - new_re_lim[0]
            new_im_span = new_im_lim[1] - new_im_lim[0]
            
            print(int(mpmath.log10(1/new_re_span))+1, int(mpmath.log10(1/new_im_span))+1)
            
            dps = max([int(mpmath.log10(1/new_re_span))+1, int(mpmath.log10(1/new_im_span))+1]) + int(mpmath.log10(max(res))) + 1
            mpmath.mp.dps = dps
            print(mpmath.mp)
            
            mset = brot_gen_parallel(new_re_lim, new_im_lim, 1000000)
            
            self.axes.cla()
            self.axes.imshow(mset, origin="lower", vmin=-0.1, cmap=colmap, interpolation="bilinear")
            
            self.canvas.draw()
            
            self.re_lim = new_re_lim
            self.im_lim = new_im_lim
        else:
            if event.button == 'up':
                scale_factor = 1 / base_factor
                self.depth *= base_factor
            elif event.button == 'down':
                scale_factor = base_factor
                self.depth /= base_factor
            else:
                return
            
            cur_xlim = ax.get_xlim()
            cur_ylim = ax.get_ylim()
            xdata = event.xdata # get event x location
            ydata = event.ydata
            xlim = [xdata - (xdata - cur_xlim[0])*scale_factor,
                    xdata + (cur_xlim[1] - xdata)*scale_factor]
            ylim = [ydata - (ydata - cur_ylim[0])*scale_factor,
                    ydata + (cur_ylim[1] - ydata)*scale_factor]
            self.axes.set_xlim(xlim[0], xlim[1])
            self.axes.set_ylim(ylim[0], ylim[1])
            self.canvas.draw()
            
if __name__ == "__main__":
    re_lim = np.array([mpmath.mpmathify(-2.5), mpmath.mpmathify(2.5)])
    im_lim = res[1]/res[0] * re_lim

    mset = brot_gen_parallel(re_lim, im_lim, 1000000)
    
    fig = plt.figure()
    fig.canvas.mpl_connect('close_event', close)
    
    ax = plt.gca()
    ax.imshow(mset, origin="lower", vmin=-0.1, cmap=colmap, interpolation="bilinear")
    
    generator = Generator(ax, re_lim, im_lim)
    
    plt.axis("off")
    plt.show(block=True)