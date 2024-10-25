# -*- coding: utf-8 -*-
"""
Created on Sat Oct 19 22:22:31 2024

@author: phkwa
"""

import sys
sys.path.insert(0, "P:\\Philipp\\Christchurch\\parameter_upanshu\\particles_github")

import numpy as np
import matplotlib.pyplot as plt
from math import pi, sqrt
from particles import distributions as dists
from particles import state_space_models as ssm
from particles import mcmc
from particles import smc_samplers as smp
from particles import kalman
from particles import collectors
from particles import resampling 

import particles 
from scipy.special import logsumexp

np.random.seed(1)
# D_obs = 0.01
h = 0.02

# def A(theta):
#     AA = np.eye(3)
#     AA[0,2] = h*np.cos(theta['phi']) 
#     AA[1,2] = h*np.sin(theta['phi'])
#     return AA

class ABM(ssm.StateSpaceModel):
  default_params = {'D0': 0.1, 'Drot': 1.0, 'Dobs': 0.01, 'v': 15., 
                    'm0xy': 0, 'sd0xy': 5}
  
  def __init__(self, **kwargs):
      self.__dict__.update(self.default_params)  
      self.__dict__.update(kwargs) # allow explicit choices to overwrite default 
      
  def PX0(self):
    return dists.IndepProd(dists.Normal(self.m0xy, self.sd0xy), # prior on x
                           dists.Normal(self.m0xy, self.sd0xy), # prior on y
                           dists.Uniform(a=0., b=2*pi))  # prior on angle
  
  def PX(self, t, xp):
    newx_mean = xp[:,0] + self.v*h*np.cos(xp[:,2]) 
    newy_mean = xp[:,1] + self.v*h*np.sin(xp[:,2])
    #covXY = 2*self.D0*h*np.eye(2)
    Xdistr = dists.Normal(newx_mean, sqrt(2*self.D0*h)) # watch out for h = self.h in FK_ABM_RB!
    Ydistr = dists.Normal(newy_mean, sqrt(2*self.D0*h))
    phi_distr = dists.Normal(xp[:,2], sqrt(self.Drot*h))
    return dists.IndepProd(Xdistr,Ydistr, phi_distr)
  
  def PY(self, t, xp, x):
    return dists.MvNormal(loc = x[:,0:2], cov = np.array([[self.Dobs,0],[0, self.Dobs]]))


class FK_ABM_RB_Bootstrap(particles.FeynmanKac):
    
  default_params = {"D0": 0.1, "Drot": 1.0,"Dobs": 0.01,  
                    'm0': np.array([0,0,20]), 'cov0': np.diag([5,5,10])}
  def __init__(self, yt=None, **kwargs):
    self.__dict__.update(self.default_params)  
    self.__dict__.update(kwargs)
    self.yt = yt
    self.T = 0 if yt is None else len(yt)
  
  def M0(self, N):
    prior_dict = {'phi': dists.Uniform(0, 2*pi)}
    my_prior = dists.StructDist(prior_dict)
    thetas = my_prior.rvs(size=N)
    MC0 = smp.FancyList([kalman.MeanAndCov(mean=self.m0, cov=self.cov0) for j in range(N)])
    
    
    # do all Kalman predictions
    J = N
    # MC_pred = [None for j in range(J)]
    MC_filter = [None for j in range(J)]
    logpyts = np.zeros(J)#np.array([0.0 for j in range(J)])
    for j in range(J):
      #theta_old = part['theta']
      # theta_old = thetas_old_array[j,:]
      # theta_pred = thetas_new_arr[j,:]
      # MC_old = part['MC']
      # Amatrix = A(theta_old)
      # Gammamatrix = Gamma(theta_old)
      # MC_pred[j] = kalman.predict_step(Amatrix, np.diag([(2*self.h*self.D0),(2*self.h*self.D0),0]), MC_old)
      Hmatrix = np.array([[1,0,0],[0,1,0]])
      # Sigmamatrix = Sigma(theta_pred)
      Sigmamatrix = np.eye(2)*np.abs(self.Dobs)
      MC_filter[j], logpyts[j] = kalman.filter_step(Hmatrix, Sigmamatrix, MC0[j], self.yt[0])
    return smp.ThetaParticles(theta=thetas, MC=smp.FancyList(MC_filter), logGs=logpyts)
  
  def M(self, t, xp):
    thetas_old = xp.theta
    # logGs_old = xp.logGs
    thetas_old_array = smp.view_2d_array(thetas_old)
    thetas_new_arr = np.mod(thetas_old_array + np.random.normal(0, sqrt(h*self.Drot)), 2*pi)
    out_dtype = thetas_old.dtype
    out_shape = thetas_old.shape
    thetas_new = np.zeros(out_shape, dtype=out_dtype)
    thetas_new['phi'] = thetas_new_arr[:,0]
    
    
    # do all Kalman predictions
    J = xp.N
    MC_pred = [None for j in range(J)]
    MC_filter = [None for j in range(J)]
    logpyts = np.zeros(J)
    for j, part in enumerate(xp):
      #theta_old = part['theta']
      theta_old = thetas_old_array[j,:]
      theta_pred = thetas_new_arr[j,:]
      MC_old = part['MC']
      #Amatrix = A(theta_old)
      Amatrix = np.eye(3)
      Amatrix[0,2] = h*np.cos(theta_old) 
      Amatrix[1,2] = h*np.sin(theta_old)
      # Gammamatrix = Gamma(theta_old)
      # print(theta_old*360/(2*pi))
      # print(MC_old)
      MC_pred[j] = kalman.predict_step(Amatrix, np.diag([(2*h*self.D0),(2*h*self.D0),0]), MC_old)
      # print(MC_pred[j])
      Hmatrix = np.array([[1,0,0],[0,1,0]])
      # Sigmamatrix = Sigma(theta_pred)
      Sigmamatrix = np.eye(2)*np.abs(self.Dobs)
      MC_filter[j], logpyts[j] = kalman.filter_step(Hmatrix, Sigmamatrix, MC_pred[j], self.yt[t])
      # print(MC_filter[j])
      # print(logpyts[j])
      
      # logweights_new = logGs_old + logpyts
      # logweights_new -= logsumexp(logweights_new)
    return smp.ThetaParticles(theta=thetas_new, MC=smp.FancyList(MC_filter), logGs=logpyts)  

  def logG(self, t, xp, x):
    # all these computations have been done in a previous M step, so in principle just load them
    return x.logGs
  
class FK_ABM_RB_GuidedPF(particles.FeynmanKac):
    
  default_params = {"D0": 0.1, "Drot": 1.0,"Dobs": 0.01,  
                    'm0': np.array([0,0,20]), 'cov0': np.diag([5,5,10]), 'guide_angle_sd': pi/2}
  def __init__(self, yt=None, **kwargs):
    self.__dict__.update(self.default_params)  
    self.__dict__.update(kwargs)
    self.yt = yt
    self.T = 0 if yt is None else len(yt)
  
  def M0(self, N):
    prior_dict = {'phi': dists.Uniform(0, 2*pi)}
    my_prior = dists.StructDist(prior_dict)
    thetas = my_prior.rvs(size=N)
    MC0 = smp.FancyList([kalman.MeanAndCov(mean=self.m0, cov=self.cov0) for j in range(N)])
    
    
    # do all Kalman predictions
    J = N
    # MC_pred = [None for j in range(J)]
    MC_filter = [None for j in range(J)]
    logpyts = np.zeros(J)#np.array([0.0 for j in range(J)])
    for j in range(J):
      #theta_old = part['theta']
      # theta_old = thetas_old_array[j,:]
      # theta_pred = thetas_new_arr[j,:]
      # MC_old = part['MC']
      # Amatrix = A(theta_old)
      # Gammamatrix = Gamma(theta_old)
      # MC_pred[j] = kalman.predict_step(Amatrix, np.diag([(2*self.h*self.D0),(2*self.h*self.D0),0]), MC_old)
      Hmatrix = np.array([[1,0,0],[0,1,0]])
      # Sigmamatrix = Sigma(theta_pred)
      Sigmamatrix = np.eye(2)*np.abs(self.Dobs)
      MC_filter[j], logpyts[j] = kalman.filter_step(Hmatrix, Sigmamatrix, MC0[j], self.yt[0])
    return smp.ThetaParticles(theta=thetas, MC=smp.FancyList(MC_filter), logGs=logpyts)
  
  def M(self, t, xp):
    thetas_old = xp.theta
    # logGs_old = xp.logGs
    thetas_old_array = smp.view_2d_array(thetas_old)
    thetas_new_arr = np.mod(thetas_old_array + np.random.normal(0, sqrt(h*self.Drot)), 2*pi)
    out_dtype = thetas_old.dtype
    out_shape = thetas_old.shape
    thetas_new = np.zeros(out_shape, dtype=out_dtype)
    thetas_new['phi'] = thetas_new_arr[:,0]
    
    
    # guess angle for guided PF
    if t  >  0:
        obs_now = self.yt[t]
        obs_prev = self.yt[t-1]
        angle_guessed = np.arccos((obs_now[0] - obs_prev[0])/np.sqrt(((obs_now[0] - obs_prev[0]))**2 + (obs_now[1] - obs_prev[1])**2))
        
    
    # do all Kalman predictions
    J = xp.N
    MC_pred = [None for j in range(J)]
    MC_filter = [None for j in range(J)]
    logpyts = np.zeros(J)
    for j, part in enumerate(xp):
      #theta_old = part['theta']
      theta_old = thetas_old_array[j,:]
      theta_pred = thetas_new_arr[j,:]
      MC_old = part['MC']
      #Amatrix = A(theta_old)
      Amatrix = np.eye(3)
      Amatrix[0,2] = h*np.cos(theta_old) 
      Amatrix[1,2] = h*np.sin(theta_old)
      # Gammamatrix = Gamma(theta_old)
      # print(theta_old*360/(2*pi))
      # print(MC_old)
      MC_pred[j] = kalman.predict_step(Amatrix, np.diag([(2*h*self.D0),(2*h*self.D0),0]), MC_old)
      # print(MC_pred[j])
      Hmatrix = np.array([[1,0,0],[0,1,0]])
      # Sigmamatrix = Sigma(theta_pred)
      Sigmamatrix = np.eye(2)*np.abs(self.Dobs)
      MC_filter[j], logpyts[j] = kalman.filter_step(Hmatrix, Sigmamatrix, MC_pred[j], self.yt[t])
      # print(MC_filter[j])
      # print(logpyts[j])
      
      # logweights_new = logGs_old + logpyts
      # logweights_new -= logsumexp(logweights_new)
    return smp.ThetaParticles(theta=thetas_new, MC=smp.FancyList(MC_filter), logGs=logpyts)  

  def logG(self, t, xp, x):
    # all these computations have been done in a previous M step, so in principle just load them
    return x.logGs

  

if __name__ == "__main__":
    np.random.seed(1)
    
    
    def plot_thetaPart(tp, wgts, k=None, N_smp=50):
        angles = tp.theta['phi']
        ms = [MC.mean for MC in tp.MC]
        Cs = [MC.cov for MC in tp.MC]
        
        plt.figure(figsize=(3,6))
        plt.subplot(311)
        plt.hist(angles,50,weights=wgts.W)
        if k is not None:
            plt.axvline(angles[k], color='k')
        plt.xlim([-0.5,2*pi+0.5])
        plt.title("phi")
        
        indices = resampling.stratified(wgts.W)
        
        samples = np.vstack([np.random.multivariate_normal(ms[0].flatten(), Cs[0], N_smp) for k in indices])
        
        plt.subplot(312)
        # plt.plot(samples[:,0], samples[:, 1], '.', alpha=0.01)
        plt.hist2d(samples[:,0], samples[:, 1], 20)
        if k is not None:
            plt.plot(x_true[k][0,0], x_true[k][0,1], 'kx')
            plt.plot(data[k][0,0], data[k][0,1], 'rx')
        plt.title("x,y")
        
        plt.subplot(3,1,3)
        plt.hist(samples[:,2], 20)
        if k is not None:
            plt.axvline(myABM.v, color='k')
        plt.title("v")
        plt.xlim([-30,30])
        plt.tight_layout()
        
    Nt = 20
    myABM = ABM()
    x_true, data = myABM.simulate(Nt) 
    
    
    plt.figure()
    plt.plot(x_true[0][0,0], x_true[0][0,1], 'kx')
    plt.plot([x[:,0] for x in x_true], [x[:,1] for x in x_true], '.-')
    plt.plot([d[:,0] for d in data], [d[:,1] for d in data], '.-')
    plt.axis("equal")
    
    plt.figure()
    plt.plot([x[:,2] for x in x_true])
    plt.title("angle")
    
    my_FKRB = FK_ABM_RB_Bootstrap(yt = data)
    pf = particles.SMC(fk=my_FKRB, N=1000, store_history=True)
    # plot_thetaPart(pf.X, pf.wgts, 0)  
    pf.run()
    
    # 
    # 
    # print(pf.logLt)
    
    # for k in range(Nt):
    #     plot_thetaPart(pf.hist.X[k], pf.hist.wgts[k], k)
        
    # my_FKRB_mod = FK_ABM_RB_Bootstrap(yt = data, Drot=4.0, D0=100.0)
    # pf = particles.SMC(fk=my_FKRB_mod, N=10000, store_history=True)
    # pf.run()
    # print(pf.logLt)
    # # plt.figure()
    # # phi_history = np.stack([pf.hist.X[k].theta['phi'] for k in range(Nt)])
    # # plt.violinplot(phi_history.T)
    # # plt.plot([x[0,2] for x in x_true])

    # # plt.figure()
    # # plt.hist(pf.hist.X[-1].theta['phi'], 20)
    
    
    
    for k in range(Nt):
        plot_thetaPart(pf.hist.X[k], pf.hist.wgts[k], k)
