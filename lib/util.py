import re

import matplotlib.pyplot as plt
from matplotlib import colors

import numpy as np

def map_score(perf):
    """ associate a performance midi with a kern score based on filename conventions """
    regex = re.compile('(\d\d\d)_bwv(\d\d\d)(f|p)')
    info = regex.search(perf)
    num, bwv, part = info.group(1,2,3)
    bwv = int(bwv)
    book = 1 + int(bwv > 869)
    score = 'wtc{}{}{:02d}'.format(book,part,bwv - 845 - (book-1)*24)

    return score

def plot_events(ax, events, stride=512, num_windows=2000):
    timings = np.cumsum(events[:,-1])
    x = np.zeros([num_windows,128])
    for i in range(num_windows):
        time = (stride*i)/44100.
        k = np.argmin(time>=timings)
        x[i] = events[k,:128]

    ax.imshow(x.T[::-1][30:90], interpolation='none', cmap='Greys', aspect=num_windows/250)

def colorplot(ax, x, y, aspect=4):
    cmap = colors.ListedColormap(['white','red','orange','black'])
    bounds = [0,1,2,3,4]
    norm = colors.BoundaryNorm(bounds, cmap.N)
    ax[0].imshow(x.T, interpolation='none', cmap='Greys', aspect=aspect)
    ax[1].imshow(y.T, interpolation='none', cmap='Greys', aspect=aspect)
    ax[2].imshow(x.T*2 + y.T, interpolation='none', cmap=cmap, aspect=aspect, norm=norm)

def pianoroll(events, fs=44100, stride=512):
    notes = events[:,:-1]
    timing = np.cumsum(events[:,-1])
    num_windows = int(timing[-1]*(44100./stride))+1
    
    x = np.zeros([num_windows,128])
    for i in range(num_windows):
        t = (i*stride)/fs
        x[i] = notes[np.argmin(t>timing)]
        
    return x
        
def pscore(score, alignment, stride=512, start=False):
    epsilon = 1e-4
    notes = score[:,:-1]
    score_time, perf_time = zip(*alignment)
    num_windows = int(alignment[-1][1]*(44100./stride))+1
    
    x = np.zeros([num_windows,128])
    for i in range(num_windows):
        t = (i*stride)/44100.                           # time (in seconds) in the performance
        if start:                                       # if start time is given
            if t < perf_time[0]: continue

        j = np.argmin(t>np.array(perf_time))            # index of the first event in performance that ends after time t
        s = score_time[j]                               # time (in beats) in the score
        if s > np.sum(score[:,-1]): continue
        k = np.argmin(s>np.cumsum(score[:,-1])+epsilon) # index of the first event in score that ends after time s
        x[i] = notes[k]
    
    return x

