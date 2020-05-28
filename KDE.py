
import csv
import numpy as np
import numpy.matlib
import pandas as pd
import scipy.sparse.linalg as ll
from scipy.spatial import distance
import matplotlib.pyplot as plt
from sklearn import preprocessing
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D

import os


cur_data_dir = os.getcwd()

path = cur_data_dir+'\n90pol.csv'

df = pd.read_csv(path,header=0)
data=df.to_numpy()

data.shape
y = data[:,2]
set(y)
data = data[:,:2]


pdata = data[:,:2]
pdata = preprocessing.scale(pdata)
m, n = pdata.shape


## PART A

# Histogram

# for 2 dimensional data
           
bin1=[5,10,15,20,25,30]
for i in bin1:
    min_data = pdata.min(0)
    max_data = pdata.max(0)
    nbin = i      
    fig = plt.figure()
    fig.set_size_inches(15, 15)
    ax = fig.add_subplot(111, projection='3d')
    hist, xedges, yedges = np.histogram2d(pdata[:,0], pdata[:,1], bins=nbin)
    xpos, ypos = np.meshgrid(xedges[:-1]+xedges[1:], yedges[:-1]+yedges[1:])
    xpos = xpos.flatten()/2.
    ypos = ypos.flatten()/2.
    zpos = np.zeros_like (xpos)
    dx = xedges [1] - xedges [0]
    dy = yedges [1] - yedges [0]
    dz = hist.flatten()
    max_height = np.max(dz)   
    min_height = np.min(dz)
    cmap = plt.cm.get_cmap('jet')
    rgba = [cmap((k-min_height)/max_height) for k in dz] 
    ax.bar3d(xpos, ypos, zpos, dx, dy, dz,color=rgba )
    plt.title('Using {} bins'.format(nbin))
    ax.set_ylabel('acc')
    ax.set_xlabel('amygdala')

plt.show()


## PART B

#kernel density estimator

#from scipy import stats
gridno=90
min_data = pdata.min(0)
max_data = pdata.max(0)
inc1 = (max_data[0]-min_data[0])/gridno
inc2 = (max_data[1]-min_data[1])/gridno
gridx, gridy = np.meshgrid(np.linspace(min_data[0], max_data[0],gridno), np.linspace(min_data[1], max_data[1],gridno) )
X, Y = gridx, gridy
#gridx, gridy = np.meshgrid(np.linspace(min_data[0]-10*inc1, max_data[0]+10*inc1,gridno), np.linspace(min_data[1]-10*inc2, max_data[1]+10*inc1,gridno) )


positions = np.vstack([X.ravel(), Y.ravel()])
values = np.vstack([pdata[:,0],pdata[:,1]])

#Defihning gassian KDE for the data


def KDE_Algo(pdata,var_data,m,h):
    
    #diff=np.linalg.norm(pdata-var_data)/h
    diff = np.power((pdata- var_data)/h, 2).sum(axis=1)
    total=(1/np.sqrt(2*np.pi))*np.exp(-diff/(2))
    p=total.sum()/(m*h)
    return p


h1=[0.1,0.2,0.3,0.43,0.48,0.5]


for b in h1:
    
    h=b
    Z=[KDE_Algo(pdata,i,m,h) for i in positions.T]
    Z=np.asarray(Z).reshape(X.shape)
    fig = plt.figure()
    fig.set_size_inches(10, 10)
    ax=fig.add_subplot(111)

    asd=ax.contour(X,Y,Z,cmap=plt.cm.gist_earth_r)
    ax.clabel(asd,inline=4, fontsize=10)
    ax.plot(pdata[:,0],pdata[:,1], 'k.', markersize=5)
    ax.set_xlim([min_data[0], max_data[0]])

    ax.set_ylim([min_data[1], max_data[1]])
    ax.set_ylabel('acc')
    ax.set_xlabel('amygdala')
    plt.title("Bandwidth used is {}".format(h))
plt.show()





## Part C


## Plotting conditinal distribution for ACC 

a4_dims = (10, 10)
fig, ax = plt.subplots(figsize=a4_dims)
for orientation1 in [2,3,4,5] :
    pdatac=df.loc[df['orientation'] == orientation1,['amygdala']].to_numpy()
    
    # Draw the density plot\
    
    sns.distplot(pdatac, hist = False, kde = True,
                 kde_kws = {'shade': True, 'linewidth': 3,'kernel':'gau'}, 
                  label = 'Orientation {}'.format(orientation1),ax=ax)
# Plot formatting
plt.legend(prop={'size': 16}, title = 'Amydala')
plt.title('AMYGDALA conditional density Plot with Multiple Orientations')
plt.xlabel('X')
plt.ylabel('Density')



## Plotting conditinal distribution for ACC 

a4_dims = (10, 10)
fig, ax = plt.subplots(figsize=a4_dims)
for orientation1 in [2,3,4,5] :
    pdatac=df.loc[df['orientation'] == orientation1,['acc']].to_numpy()
    
    # Draw the density plot
    sns.distplot(pdatac, hist = False, kde = True,
                 kde_kws = {'shade': True, 'linewidth': 3,'kernel':'gau'}, 
                  label = 'Orientation {}'.format(orientation1),ax=ax)
# Plot formatting
plt.legend(prop={'size': 16}, title = 'Acc')
plt.title('ACC condistional density Plot with Multiple Orientations')
plt.xlabel('X')
plt.ylabel('Density')






