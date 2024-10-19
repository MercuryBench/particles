# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 21:08:35 2024

@author: phkwa
"""

import collections

import numpy as np
from scipy.linalg import solve

import numpy as np
import matplotlib.pyplot as plt
from numpy import exp, log, pi, random, sqrt, cos, sin
from scipy.stats import norm



import sys
# Insert the path of the modified version at the front of sys.path
sys.path.insert(0, "C:\\Users\\phkwa\\particles_github")
import particles
from particles import distributions as dists
from particles import state_space_models as ssm
from particles import kalman as km

lingauss = km.LinearGauss()
print(lingauss.PX0().rvs(5))

#%%
## Static case 

F = np.eye(2)
covX = 0.01*np.eye(2)
G = np.array([[0,1]])
covY = 2.5*np.eye(1)

mu0 = np.array([0,0])
cov0 = 3*np.eye(2)

mvlingauss = km.MVLinearGauss(theta0=None, F=F, G=G, covX=covX, covY=covY, mu0=mu0, cov0=cov0)

x0 = mvlingauss.PX0().rvs(1)
N = 100
xs = np.zeros((N,2))
ys = np.zeros((N-1,1))
xs[0] = x0
#ys[0] = mvlingauss.PY(0, None, x0).rvs(1)
for n in range(N-1):
  xs[n+1] = mvlingauss.PX(0, xs[n]).rvs(1)
  ys[n] = mvlingauss.PY(0, None, xs[n+1]).rvs(1)



plt.figure()
plt.plot(xs[:,0], xs[:,1])

plt.figure()
plt.plot(np.sum(xs,axis=1))
plt.plot(ys)

k = km.Kalman(mvlingauss, ys)
k.filter()


means = np.array([k.filt[n].mean for n in range(len(k.filt))])
covs =  [k.filt[n].cov for n in range(len(k.filt))]
covx1s = np.array([covs[n][0,0] for n in range(len(k.filt))])
covx2s = np.array([covs[n][1,1] for n in range(len(k.filt))])

plt.figure()
plt.subplot(211)
plt.plot(xs[1:,0], color="tab:orange")
plt.plot(means[:,0], color="tab:blue")
plt.fill_between(np.arange(1,N), means[:,0]+2*covx1s, means[:,0]-2*covx1s, alpha=0.2, color="tab:blue")
plt.subplot(212)
plt.plot(xs[1:,1], color="tab:orange")
plt.plot(means[:,1], color="tab:blue")
plt.fill_between(np.arange(1,N), means[:,1]+2*covx2s, means[:,1]-2*covx2s, alpha=0.2, color="tab:blue")


#%% Parameter-dependent

F = lambda t: np.eye(2)
covX = lambda t: 0.05*np.eye(2)
G = lambda t: np.array([[np.cos(t), np.sin(t)]])
covY = lambda t: 0.5*np.eye(1)

mu0 = lambda t: np.array([0,0])
cov0 = lambda t: 1*np.eye(2)

mvlingauss = km.MVLinearGauss(theta0=0, F=F, G=G, covX=covX, covY=covY, mu0=mu0, cov0=cov0)
x0 = mvlingauss.PX0(0).rvs(1)
N = 100
ts = np.linspace(0,2*pi,N)
xs = np.zeros((N,2))
ys = np.zeros((N-1,1))
xs[0] = x0
#ys[0] = mvlingauss.PY(0, None, x0).rvs(1)
for n in range(N-1):
  xs[n+1] = mvlingauss.PX(0, xs[n], theta=ts[n+1]).rvs(1)
  ys[n] = mvlingauss.PY(0, None, xs[n+1], theta=ts[n+1]).rvs(1)



plt.figure()
plt.plot(xs[:,0], xs[:,1])

plt.figure()
plt.plot(ts[1:], ys)


k = km.Kalman(mvlingauss, ys, thetas=ts)
k.filter()


means = np.array([k.filt[n].mean for n in range(len(k.filt))])
covs =  [k.filt[n].cov for n in range(len(k.filt))]
covx1s = np.array([covs[n][0,0] for n in range(len(k.filt))])
covx2s = np.array([covs[n][1,1] for n in range(len(k.filt))])

plt.figure()
plt.subplot(211)
plt.plot(ts[1:], xs[1:,0], color="tab:orange", label="x_true_1")
plt.plot(ts[1:], means[:,0], color="tab:blue", label="mean_1")
plt.fill_between(ts[1:], means[:,0]+2*covx1s, means[:,0]-2*covx1s, alpha=0.2, color="tab:blue")
plt.legend()
plt.subplot(212)
plt.plot(ts[1:], xs[1:,1], color="tab:orange", label="x_true_2")
plt.plot(ts[1:], means[:,1], color="tab:blue", label="mean_2")
plt.fill_between(ts[1:], means[:,1]+2*covx2s, means[:,1]-2*covx2s, alpha=0.2, color="tab:blue")
plt.legend()