"""
# static dataset test
"""
import os
import time
import numpy as np
import pandas as pd
import torch as pt
import matplotlib.pyplot as pl
#from bhmtorch_cpu import BHM2D_PYTORCH

def getPartitions(cell_max_min, nPartx1, nPartx2):
    """
    :param cell_max_min: The size of the entire area
    :param nPartx1: How many partitions along the longitude
    :param nPartx2: How many partitions along the latitude
    :return: a list of all partitions
    """
    width = cell_max_min[1] - cell_max_min[0]
    height = cell_max_min[3] - cell_max_min[2]
    cell_max_min_segs = []
    for x in range(nPartx1):
        for y in range(nPartx2):
            seg_i = (cell_max_min[0] + width / nPartx1 * x, cell_max_min[0] + width / nPartx1 * (x + 1), \
                     cell_max_min[2] + height / nPartx2 * y, cell_max_min[2] + height / nPartx2 * (y + 1))
            cell_max_min_segs.append(seg_i)

    return cell_max_min_segs

def load_parameters(case):
    parameters = \
        {filename: \
             ( os.path.abspath('../../Datasets/simulated/'+filename),
              (5, 5), #hinge point resolution
              (-100, 100, 0, 100), #area [x1_min, x1_max, x2_min, x2_max] #NOTE: chnaged the range
              None,
              None,
              0.08, #gamma
              ),

         }

    return parameters[case]

# Settings
dtype = pt.float32
device = pt.device("cpu")
# device = pt.device("cuda:0") # Uncomment this to run on GPU

# Get the filename to read data points from
filename = 'static_test1' #input("Input file: ")

# Read the file
fn_train, cell_resolution, cell_max_min, _, _, gamma = load_parameters(filename)

# Partition the environment into to 4 areas
# TODO: We can parallelize this
cell_max_min_segments = getPartitions(cell_max_min, 1, 1) #NOTE: this is implemented to segment the environment and do mapping speperately. For now, let's consider that there's only one segment.

# Read data
print('\nReading '+fn_train+'.csv...')
g = pd.read_csv(fn_train+'.csv', delimiter=', ').values[:, :]

# Filter data
layer = g[:,0] <= 100 #NOTE: let's consider time frames < 100
g = pt.tensor(g[layer, :], dtype=pt.float32)
X = g[:, 1:3] #NOTE: Previously we've read incorrect columns
y = g[:, 3].reshape(-1, 1)
# if pt.cuda.is_available():
#     X = X.cuda()
#     y = y.cuda()

toPlot = []
totalTime = 0
for segi in range(len(cell_max_min_segments)):
    print(' Mapping segment {} of {}...'.format(segi+1,len(cell_max_min_segments)))
    cell_max_min = cell_max_min_segments[segi]

    bhm_mdl = BHM2D_PYTORCH(gamma=gamma, grid=None, cell_resolution=cell_resolution, cell_max_min=cell_max_min, X=X, nIter=1)

    t1 = time.time()
    bhm_mdl.fit(X, y)
    t2 = time.time()
    totalTime += (t2-t1)

    # query the model
    q_resolution = 1
    xx, yy= np.meshgrid(np.arange(cell_max_min[0], cell_max_min[1] - 1, q_resolution),
                         np.arange(cell_max_min[2], cell_max_min[3] - 1, q_resolution))
    grid = np.hstack((xx.ravel()[:, np.newaxis], yy.ravel()[:, np.newaxis]))
    Xq = pt.tensor(grid, dtype=pt.float32)
    yq = bhm_mdl.predict(Xq)
    toPlot.append((Xq,yq))
print(' Total training time={} s'.format(np.round(totalTime, 2)))

# Plot occupancy map
pl.close('all')
pl.figure(figsize=(10,5))
pl.subplot(121)
# Scatter plot raw data
pl.rcParams['figure.facecolor'] = 'white'
pl.scatter(X[:,0], X[:,1], c=y, s=5, cmap='jet')
pl.subplot(122)
for segi in range(len(cell_max_min_segments)):
    ploti = toPlot[segi]
    Xq, yq = ploti[0], ploti[1]
    pl.scatter(Xq[:, 0], Xq[:, 1], c=yq, cmap='jet', s=5, vmin=0, vmax=1)
pl.colorbar()
#pl.xlim([0,200]); pl.ylim([-100,150])
pl.title(filename)
#pl.savefig(os.path.abspath('../../Outputs/'+filename+'.png'))
pl.show()
