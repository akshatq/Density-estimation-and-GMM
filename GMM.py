######################################################################
# The codes are based on Python3.7.6

# @version 1.0
# @author Akshat Chauhan
######################################################################
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.linalg import eigh
import pandas as pd


print("Current Working Directory " , os.getcwd())


matFile1 = sio.loadmat('data.mat')
data = matFile1['data']
data=data.T
d=data.shape[1]

## Set Parameters K as number of components and r for rank approximatio
k=2
r=50

### visualizing digit 2
img2=data[456,:]

fig = plt.figure()
fig.set_size_inches(5, 5)
#ax=fig.plot()
ax = fig.add_subplot()
ax.set_title('Sample Image of digit 2')

ax.imshow(img2.reshape(28,28).T, aspect='auto', cmap=plt.cm.gray, interpolation='nearest',zorder=100000)


### visualizing digit 6

img6=data[1450,:]

fig = plt.figure()
fig.set_size_inches(5, 5)
#ax=fig.plot()
ax = fig.add_subplot()
ax.set_title('Sample Image of digit 6')

ax.imshow(img6.reshape(28,28).T, aspect='auto', cmap=plt.cm.gray, interpolation='nearest',zorder=100000)


def initialize(k):
    muk=[]
    pik=[]
    covk=[]
    for i in range(k):
        covk.append(np.identity(d))
        pik.append(1/k)
        muk.append(np.random.normal(loc=0,scale=1,size=(1,d)))
    muk=np.array([muk]).reshape(k,d)
    covk=np.array([covk]).reshape(k,d,d)
    pik=np.array([pik]).reshape(k,1)
    return pik,muk,covk

def rank_approx(x,s,mu,r):   
    S,U=eigh(s)
    qw=np.round(S,3) # convert negative or really small values to zero
    if np.count_nonzero(qw)<r: ## check for zero values and reassign r
        r=np.count_nonzero(qw)  
    U=U[:,::-1]
    S=S[::-1]
    Sr=S[0:r]
    Ur=U[:,0:r]
    Xr=x@Ur
    muR=mu@Ur
    return Xr,Sr,muR.reshape(1,r)


#v=np.array([[.00000023,.000023,.124,2,3,4.5566,.00000000067]])
#np.round(v,5)

def density_f(Xr,Sr,muR):
  
   Xr=Xr.reshape(1,len(Xr))
   Sr=np.diag(Sr)
   sigmaI=np.linalg.inv(Sr)
   mk=(Xr-muR)@sigmaI@(Xr-muR).T
   det=np.linalg.det(Sr)
   d=1/(det)**0.5
   calc=d*np.exp(-mk/2)
   return calc[0][0]
   
  
def expectation_alg(pik,x,muk,covk,r):
    tauk=[]
    k=len(covk)
    for i in range(k):
        Xr,Sr,muR = rank_approx(x,covk[i],muk[i],r)
        t=[pik[i][0]*density_f(Xr[j],Sr,muR) for j in range(len(x))]
        tauk.append(t)
    tauk=np.array(tauk).T 
    mle=np.log(np.sum(tauk,axis=1))
    mle_sum=np.sum(mle,axis=0)
    tauk=tauk/((np.sum(tauk,axis=1).reshape(x.shape[0],1)))
    return tauk,mle_sum

    


def maximization_alg(x,tauk):
    k=tauk.shape[1]
    mk=[]
    ck=[]
    pk=[]
    m=len(tauk)
    for i in range(k):
        pk.append(np.sum(tauk[:,i])/m)
        mk.append(np.sum(tauk[:,i].reshape(x.shape[0],1)*x,axis=0)/np.sum(tauk[:,i]))
        mk[i]=mk[i].reshape(1,x.shape[1])

        ck.append(((x-mk[i]).T@(tauk[:,i].reshape(x.shape[0],1)*(x-mk[i])))/(np.sum(tauk[:,i])))
    mk=np.array([mk]).reshape(k,x.shape[1])
    ck=np.array([ck]).reshape(k,x.shape[1],x.shape[1])
    pk=np.array([pk]).reshape(k,1)
    return pk,mk,ck
        

### Running Algo
    
pik,muk,covk=initialize(k)
p,z=expectation_alg(pik,data,muk,covk,r)
old_logl=abs(z)
new_logl=old_logl-20
log_l=[]
for i in range(60):
#for i in range(20):     
    p,mle=expectation_alg(pik,data,muk,covk,r)
    old_logl=abs(new_logl)
    new_logl=abs(mle)
    
    log_l.append(mle)
    print("Iteration {} log likelihood is .....{}".format(i+1,mle))
    pik,muk,covk=maximization_alg(data,p)

fig = plt.figure()
fig.set_size_inches(10, 10)
ax = fig.add_subplot()
ax.plot(log_l)
ax.set_xlabel('iterations')
ax.set_ylabel('Log likelihood')



######Visualizing new centers

fig = plt.figure()
fig.set_size_inches(4, 4)
#ax=fig.plot()
ax = fig.add_subplot()
ax.set_title(' Image of First Component')

ax.imshow(muk[0,:].reshape(28,28).T, aspect='auto', cmap=plt.cm.gray, interpolation='nearest',zorder=100000)

fig = plt.figure()
fig.set_size_inches(4, 4)
#ax=fig.plot()
ax = fig.add_subplot()
ax.set_title(' Image of Second Component')

ax.imshow(muk[1,:].reshape(28,28).T, aspect='auto', cmap=plt.cm.gray, interpolation='nearest',zorder=100000)


##########Calculating acucuracy

label_matFile1 = sio.loadmat('label.mat')
label_data = label_matFile1['trueLabel']

label_data
### Note it is required to look at the image of gassuian centers to identify which component refers to digit 2 or 6
####Please modify the function accordingly. 

def getlabel(x):
    if x[0]>x[1]:
        return 2 ## first component..please modify accordingly
    else:
        return 6 ## second component...please modify accordingly
    
    
ob=[getlabel(i) for i in p]
ob=np.array(ob)
s=[]
[s.append(1) for i,j in zip(ob,label_data[0]) if i==j]

print("miss Classification rate using GMM is ...{}%".format
      (round(((len(data)-len(s))/len(data))*100,2)))

#### using k-means

from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=2, random_state=0).fit(data)

### assuming first image is 2 and assigning labels to classes accordingly

def getlabel_kmeans(x):
    if x==1:
        return 2 ## first label..please modify accordingly
    else:
        return 6 ## second label...please modify accordingly

ob1=[getlabel_kmeans(i) for i in kmeans.labels_]
ob1=np.array(ob1) 
s1=[]
[s1.append(1) for i,j in zip(ob1,label_data[0]) if i==j]

print("miss Classification rate using k-means is ...{}%".format
      (round(((len(data)-len(s1))/len(data))*100,2)))


df=pd.DataFrame();
df['Actual-Labels']=label_data[0]
df['GMM-Labels']=ob
df['K-means-Lables']=ob1

df_2=df[df['Actual-Labels']==2]
df_6=df[df['Actual-Labels']==6]


accuracy=len(df_2[df_2['Actual-Labels']==df_2['GMM-Labels']])/len(df_2)
print("miss Classifiation rate for digit 2 using GMM is {}%".format((1-accuracy)*100))

accuracy=len(df_2[df_2['Actual-Labels']==df_2['K-means-Lables']])/len(df_2)
print("miss Classifiation rate for digit 2 using K-means is {}%".format((1-accuracy)*100))

accuracy=len(df_6[df_6['Actual-Labels']==df_6['GMM-Labels']])/len(df_6)
print("miss Classifiation rate for digit 6 using GMM is {}%".format((1-accuracy)*100))

accuracy=len(df_6[df_6['Actual-Labels']==df_6['K-means-Lables']])/len(df_6)
print("miss Classifiation rate for digit 6 using K-means is {}%".format((1-accuracy)*100))