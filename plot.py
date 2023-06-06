#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Routines for plotting prediction curves.

For usage information, call without any parameters.

Author: Jan Schlüter
"""
import sys
import os
if sys.version_info[0] > 2:
    from io import BytesIO as StringIO
else:
    try:
        from cStringIO import StringIO
    except:
        from StringIO import StringIO

import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

def pred_curve(data, ymax=1.0, format='png', outfile=None):
    """
    Plots a prediction curve, either saved as a file or returned as a string.
    """
    dpi = 96.
    width = len(data)
    height = 52.5
    pad = 30.
    plt.figure(figsize=((width + pad) / dpi, (height + pad) / dpi), dpi=dpi)
    plt.axes((pad/(width+pad), .5*pad/(height+pad), width/(width+pad), height/(height+pad)))
    plt.plot(data, color='#468CFF', lw=3)
    plt.fill_between(np.arange(len(data)), 0, data.ravel(), color='#C9DEFF')
    plt.xlim(0, len(data))
    plt.ylim(0, ymax or data.max())
    plt.xticks([])
    if ymax is None:
        pass
    elif ymax==1:
        plt.yticks([0, .5, 1])
    elif ymax<.3:
        plt.yticks([0, .1, .2])
    elif ymax<.5:
        plt.yticks([0, .2, .4])
    f = outfile if outfile is not None else StringIO()
    plt.savefig(f, format=format, transparent=True, dpi=dpi)
    if outfile is None:
        return f.getvalue()

def print_usage():
    print('Plots a prediction curve.')
    print('Usage: %s INFILE OUTFILE' % sys.argv[0])
    print('  INFILE: .npy input file or a just number giving the length')
    print('  OUTFILE: graphics file to write')

def main():
    if len(sys.argv) < 3:
        print_usage()
        return
    
    # 'parse' command line
    infile, outfile = sys.argv[1:]

    # read/generate input
    if infile.endswith('.npy'):
        data = np.load(infile)
    else:
        data = 0.02 * np.ones(int(infile))

    # write output
    pred_curve(data, format=None, outfile=outfile)

if __name__=="__main__":
    main()

