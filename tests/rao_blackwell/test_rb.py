import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict
from particles import distributions as dists
from particles import state_space_models as ssm
from particles import smc_samplers as smp
from particles import kalman
from particles import resampling
from particles import mcmc
from particles import rao_blackwell
import particles

case = 3


if case == 2:
    # 2nd example
    from math import sqrt, cos, sin, pi
    D0 = 1.4
    Drot = 0.5
    Dobs = 0.1
    dt = 0.1
    # A = lambda w: np.array([[1., 0., dt*cos(w)],[0., 1., dt*sin(w)],[0.,0.,1.]])
    # H = lambda w: np.array([[1,0,0],[0,1,0]])
    # Gamma = lambda w: np.diag([(2*dt*D0),(2*dt*D0),0.00001])
    # Sigma = lambda w: np.eye(2)*Dobs
    # m0 = lambda w: np.array([0.,0.,20.])
    # cov0 = lambda w: np.diag([5.,5.,10.])
    # PW0 = dists.Uniform(0, 2*pi)
    # def PW(t, wp):
    #     return dists.Normal(loc=wp, scale=sqrt(Drot))
    # myRB_nd = RBMVLinearGauss(PW0, PW, A, H, Gamma, Sigma, m0, cov0)
    
    class ABM(rao_blackwell.RBMVLinearGauss):
        def __init__(self, **kwargs):
            default_parameters = {'D0': D0, 'Drot': Drot, 'Dobs': Dobs}
            self.__dict__.update(default_parameters)
            self.__dict__.update(kwargs)
            A = lambda w: np.array([[1., 0., dt*cos(w)],[0., 1., dt*sin(w)],[0.,0.,1.]])
            H = lambda w: np.array([[1,0,0],[0,1,0]])
            Gamma = lambda w: np.diag([(2*dt*self.D0),(2*dt*self.D0),0.00001])
            Sigma = lambda w: np.eye(2)*self.Dobs
            m0 = lambda w: np.array([0.,0.,20.])
            cov0 = lambda w: np.diag([5.,5.,10.])
            PW0 = dists.Uniform(0, 2*pi)
            def PW(t, wp):
                return dists.Normal(loc=wp, scale=sqrt(self.Drot))
            RBMVLinearGauss.__init__(self,PW0, PW, A, H, Gamma, Sigma, m0, cov0)
            
    
    myRB_nd = ABM()
    xs, ys = myRB_nd.simulate(10)
    xs = np.vstack(xs).flatten()
    ys = np.vstack(ys)
    
    plt.figure()
    plt.plot(xs['z'][:,0], xs['z'][:,1], '.-')
    plt.plot(ys[:,0], ys[:,1], '.-')
    plt.axis("equal")
    import seaborn as sb
    #myBootRB = Bootstrap_RaoBlackwell(data = ys, A=A, Gamma=Gamma, H=H, Sigma=Sigma, PW0=PW0, PW=PW, m0=m0, cov0=cov0)
    
    
    # myBootRB = FK_ABM()
    myBootRB = rao_blackwell.Bootstrap_RaoBlackwell(myRB_nd, ys)
    results = particles.multiSMC(fk=myBootRB, N=500, nruns=30)
    plt.figure()
    plt.boxplot([r['output'].logLt for r in results]);
    pf = particles.SMC(fk=myBootRB, N=1000, store_history=True)
    pf.run()
    
    print(pf.logLt)
    
    def plot_thetaPart(tp, wgts, k=None, N_smp=50):
        angles = tp.w
        ms = [MC.mean for MC in tp.MC]
        Cs = [MC.cov for MC in tp.MC]
        
        plt.figure(figsize=(3,6))
        plt.subplot(311)
        plt.hist(np.mod(angles, 2*pi),50,weights=wgts.W)
        if k is not None:
            plt.axvline(np.mod(xs['w'][k], 2*pi), color='k')
            #plt.axvline(angles[k], color='k')
        plt.xlim([-0.5,2*pi+0.5])
        plt.title("phi")
        
        indices = resampling.stratified(wgts.W)
        
        samples = np.vstack([np.random.multivariate_normal(ms[0].flatten(), Cs[0], N_smp) for k in indices])
        
        plt.subplot(312)
        # plt.plot(samples[:,0], samples[:, 1], '.', alpha=0.01)
        plt.hist2d(samples[:,0], samples[:, 1], 20)
        if k is not None:
            plt.plot(xs['z'][k][0], xs['z'][k][1], 'kx')
            plt.plot(ys[k][0], ys[k][1], 'rx')
        plt.title("x,y")
        
        plt.subplot(3,1,3)
        plt.hist(samples[:,2], 20)
        if k is not None:
            plt.axvline(xs['z'][0,2], color='k')
        plt.title("v")
        plt.xlim([-30,30])
        plt.tight_layout()
    
    plot_thetaPart(pf.X, pf.wgts, k=len(ys)-1)
    
    case = "Drot_D0"
    
    if case == "Drot_D0_Dobs":
        prior_dict = {'D0': dists.Gamma(),
              'Drot': dists.Gamma(),
              'Dobs':dists.Gamma(a=1., b=5)}
        true_vals = [D0, Drot, Dobs]
    else:
        prior_dict = {'D0': dists.Gamma(),
              'Drot': dists.Gamma()}
        true_vals = [D0, Drot]
    my_prior = dists.StructDist(prior_dict)
    
    my_pmmh = mcmc.PMMH(ssm_cls=ABM, fk_cls=rao_blackwell.Bootstrap_RaoBlackwell, prior=my_prior, data=ys, Nx=500,  niter=1000, verbose=100)
    my_pmmh.run()
    for mm, p in enumerate(prior_dict.keys()):  # loop over D0, Drot, Dobs
        plotrange = np.linspace(prior_dict[p].ppf(0.05), prior_dict[p].ppf(0.85), 100)
        plt.figure()
        plt.subplot(211)
        plt.plot(my_pmmh.chain.theta[p], label="samples")
        plt.axhline(true_vals[mm], color="tab:orange", label="true")
        plt.xlabel('iter')
        plt.ylabel(p)
        plt.title("samples over time")
        plt.subplot(212)
        plt.hist(my_pmmh.chain.theta[p], 150, range=(plotrange[0],plotrange[-1]), density=True, label="samples")
        plt.axvline(true_vals[mm], color="tab:orange", label="true")
        plt.plot(plotrange, prior_dict[p].pdf(plotrange), '--', color="tab:green", label="prior density")
        plt.xlabel(p)
        plt.title("histogram of samples")
        plt.legend()
        #plt.xlim()
        plt.tight_layout()
    
    array_samples = np.stack([my_pmmh.chain.theta[p] for p in prior_dict.keys()])
    import corner
    corner.corner(array_samples.T, truths=true_vals, labels=[p for p in prior_dict.keys()])
    # plt.figure()
    # plt.subplot(2,1,1)
    # plt.hist(np.mod(pf.X.w, 2*pi), 50, weights=pf.wgts.W)
    # plt.axvline(xs['w'][-1], color="tab:orange")
    # plt.subplot(2,1,2)
else:
    def plot_thetaPart(tp, wgts, k=None, N_smp=50):
        diffs = tp.w
        ms = [MC.mean for MC in tp.MC]
        Cs = [MC.cov for MC in tp.MC]
        
        plt.figure(figsize=(3,6))
        plt.subplot(211)
        plt.hist(diffs,50,weights=wgts.W)
        if k is not None:
            plt.axvline(xs['w'][k], color='k')
            #plt.axvline(angles[k], color='k')
        plt.title("phi")
        
        indices = resampling.stratified(wgts.W)
        
        samples = np.vstack([np.random.multivariate_normal(ms[0].flatten(), Cs[0], N_smp) for k in indices])
        
        plt.subplot(212)
        # plt.plot(samples[:,0], samples[:, 1], '.', alpha=0.01)
        plt.hist2d(samples[:,0], samples[:, 1], 20)
        if k is not None:
            plt.plot(xs['z'][k][0], xs['z'][k][1], 'kx')
            plt.plot(ys[k][0], ys[k][1], 'rx')
        plt.title("x,y")
        
        plt.tight_layout()
    dt = 0.05
    m_param = 3.0
    tau_param = 0.5
    sigma_param= 1.0
    Dobs = 0.01
    from math import sqrt
    class DiffDiff(rao_blackwell.RBMVLinearGauss):
        def __init__(self, **kwargs):
            default_parameters = {'m_param': m_param, 'tau_param': tau_param, 'sigma_param': sigma_param}
            self.__dict__.update(default_parameters)
            self.__dict__.update(kwargs)
            A = lambda w: np.eye(2)
            H = lambda w: np.eye(2)
            Gamma = lambda w: np.eye(2)*2*dt*w
            Sigma = lambda w: np.eye(2)*2*dt*sqrt(Dobs)
            m0 = lambda w: np.zeros(2)
            cov0 = lambda w: np.diag([5.,5.])
            PW0 = dists.Gamma()
            def PW(t, wp):
                mean_D = wp +1/self.tau_param*(self.m_param - wp)*dt
                scale_D = self.sigma_param*np.sqrt(2*wp*dt)
                return dists.TruncNormal(mu=mean_D, sigma=scale_D, a=0.0, b=1000.0)
            rao_blackwell.RBMVLinearGauss.__init__(self,PW0, PW, A, H, Gamma, Sigma, m0, cov0)
    myRB_nd = DiffDiff()
    N_sim = 200
    
    ts = np.arange(0,dt*N_sim,dt)
    
    xs, ys = myRB_nd.simulate(N_sim)
    xs = np.vstack(xs).flatten()
    ys = np.vstack(ys)
    
    plt.figure()
    plt.plot(xs['z'][:,0], xs['z'][:,1], '.-')
    plt.plot(ys[:,0], ys[:,1], '.-')
    plt.axis("equal")
    
    plt.figure()
    plt.subplot(211)
    plt.plot(ts, xs['w'], label='diff')
    plt.legend()
    plt.subplot(212)
    plt.plot(ts, xs['z'][:,0],label='x')
    plt.plot(ts, xs['z'][:,1],label='y')
    plt.legend()
    
    myBootRB = rao_blackwell.Bootstrap_RaoBlackwell(myRB_nd, ys)#FK_DiffDiff(myRB_nd, ys)
    # results = particles.multiSMC(fk=myBootRB, N=500, nruns=30)
    # plt.figure()
    # plt.boxplot([r['output'].logLt for r in results]);
    pf = particles.SMC(fk=myBootRB, N=1000, store_history=True)
    pf.run()
    
    # for nn in range(N_sim):
    #     plot_thetaPart(pf.hist.X[nn], pf.hist.wgts[nn], k=nn)
    
    prior_dict = {'m_param': dists.Uniform(0.0, 5.0),
          'tau_param': dists.Uniform(0.0, 5.0),
          'sigma_param':dists.Uniform(0.0, 5.0)}
    true_vals = [m_param, tau_param, sigma_param]
    my_prior = dists.StructDist(prior_dict)
    
    my_pmmh = mcmc.PMMH(ssm_cls=DiffDiff, fk_cls=rao_blackwell.Bootstrap_RaoBlackwell, prior=my_prior, data=ys, Nx=250,  niter=4000, verbose=1000)
    my_pmmh.run()
    
    for mm, p in enumerate(prior_dict.keys()):  # loop over D0, Drot, Dobs
        plotrange = np.linspace(prior_dict[p].ppf(0.001), prior_dict[p].ppf(0.999), 100)
        plt.figure()
        plt.subplot(211)
        plt.plot(my_pmmh.chain.theta[p], label="samples")
        plt.axhline(true_vals[mm], color="tab:orange", label="true")
        plt.xlabel('iter')
        plt.ylabel(p)
        plt.title("samples over time")
        plt.subplot(212)
        plt.hist(my_pmmh.chain.theta[p], 150, range=(plotrange[0],plotrange[-1]), density=True, label="samples")
        plt.axvline(true_vals[mm], color="tab:orange", label="true")
        plt.plot(plotrange, prior_dict[p].pdf(plotrange), '--', color="tab:green", label="prior density")
        plt.xlabel(p)
        plt.title("histogram of samples")
        plt.legend()
        #plt.xlim()
        plt.tight_layout()
    
    array_samples = np.stack([my_pmmh.chain.theta[p] for p in prior_dict.keys()])
    import corner
    corner.corner(array_samples.T, truths=true_vals, labels=[p for p in prior_dict.keys()])