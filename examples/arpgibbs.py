# Python program to compute posterior results of an AR(p) model
#
#  y_t = beta_0 + beta_1 y_{t-1} + .... + beta_{p} y_{t-p} + e_t
#
# where e_t ~ N(0,sigma^2) for t=1,...,T.

# The prior for beta is flat, that is, p(beta) propto 1
# The prior for sigma^2 is p(sigma^2) propto sigma^{-2}.

# The program reports posterior means, posterior standard deviations,
# posterior median, 95% closed HPD intervals of the parameters,
# the unconditional mean beta_0/(1-beta_1-...-beta_{p+1}) and the
# 1-to-vsp step ahead forecast distributions.

# Posterior results are computed using the Gibbs sampler.

import numpy as np
import matplotlib.pyplot as  plt
from statsmodels.graphics.tsaplots import plot_acf
from scipy.stats import norm
from tabulate import tabulate
from proc import chpdi,gewtest,perc


lagp = 2               # order of autoregression 
nos = 50000             # number of valid simulations 
nob = 1000             # number of burn-in 
nod = 1                # consider every nod-th draw (thin value) 
vsp = 2                # number of prediction periods (>0) 

data =  np.loadtxt('d:\\home\\ondw\\bayes2122\\week3\\pcapinc.dat')
data=100*(data[1:]-data[:-1])   # take first differences of the data 

y=data[lagp:]       # create y vector
x=np.ones((len(y),1))   # create vector of ones

for i in range(1,lagp+1):
   x=np.column_stack((x,data[lagp-i:-i]))  # add lagged time series
  


drawpar = np.zeros((nod*nos+nob,lagp+2))        # matrix to store parameter draws
drawvsp = np.zeros((nod*nos+nob,vsp))              # matrix to store forecasts 
   
   
beta= np.linalg.inv(x.T@x)@x.T@y  # starting value Gibbs sampler
np.random.seed(0) # set random seed


for i in range((nos*nod)+nob):

     if i%1000==0: print(i)
     
     res=y-x@beta                          # residuals
     u = np.random.normal(size=len(y));    # random draws
     sigma2 = res.T@res/(u.T@u);           # simulate sigma^2
          
     bcov = np.linalg.inv(x.T@x)         # (X'X)^{-1}
     bhat = bcov@x.T@y                   # OLS estimate
     beta = (np.linalg.cholesky(sigma2*bcov))@np.random.normal(size=lagp+1)+bhat;    # simulate beta
     
     drawpar[i,-1]=sigma2       # store draws sigma2
     drawpar[i,0:lagp+1]=beta   # store draws beta
     
     # forecasting
     xvsp=x[-lagp+1]
     ylag=y[-1]
     for j in range(vsp):
        xvsp=np.insert(xvsp,1,ylag)
        drawvsp[i,j] = xvsp[0:lagp+1]@beta+np.sqrt(sigma2)*np.random.normal(size=1)
        ylag=drawvsp[i,j]
             
         

# remove bur-in draws and apply thinning
drawpar = drawpar[nob:]
drawpar = drawpar[range(0,nos*nod,nod)]
drawvsp = drawvsp[nob:]
drawvsp = drawvsp[range(0,nos*nod,nod)]


# create parameter names
varnames=[]
for i in range(lagp+1):
   varnames.append('beta ' + str(i))
varnames.append('sigma^2')   
vspnames=[]
for i in range(vsp):
   vspnames.append(str(i+1)+' step-ahead forecast')


print('Posterior Results');
print('Total number of draws',(nos*nod)+nob);
print('Number of burn-indraws',nob);
print('Thin value',nod)
print('Number of valid draws',nos)

# Plot autocorrelation in the chain (variance parameter)
plot_acf(drawpar[:,-1],lags=10)
plt.show()

# Geweke test
table=np.column_stack((varnames,gewtest(drawpar)))
headers=["parameter","mean 1","mean 2","st dev","test statistic","p-value"]
table = tabulate(table, headers, tablefmt="fancy_grid")
print("Geweke convergence test (non-HAC variances)")
print(table)

uncmean=np.zeros((nos,1))
uncmean[:,0]=drawpar[:,0]/(1-np.sum(drawpar[:,1:-1],axis=1)) # compute draws unconditional mean


# compute posterior mean, posterior standard deviation and hpd intervals 
postmean=(np.mean(drawpar, axis=0))
postsd=np.sqrt(np.var(drawpar, axis=0))
hpdi=chpdi(drawpar,0.95)

# parameters
table=np.column_stack((varnames,postmean,postsd,perc(drawpar,0.50),hpdi))
headers=["parameter","mean","stand dev","median","lower bound","upper bound"]
print("Posterior results parameters (95% HPDI)")
table = tabulate(table, headers, tablefmt="fancy_grid")
print(table)

# unconditional mean
table=np.column_stack((['unconditional mean'],np.mean(uncmean,axis=0),np.sqrt(np.var(uncmean,axis=0)),perc(uncmean,0.50),chpdi(uncmean,0.95)))
headers=["parameter","mean","stand dev","median","lower bound","upper bound"]
print("Posterior results unconditional mean")
table = tabulate(table, headers, tablefmt="fancy_grid")
print(table)

# forecasts
table=np.column_stack((vspnames,np.mean(drawvsp,axis=0),np.sqrt(np.var(drawvsp,axis=0)),perc(drawvsp,0.50),chpdi(drawvsp,0.95)))
headers=["horizon","mean","stand dev","median","lower bound","upper bound"]
print('Posterior results forecasts')
table = tabulate(table, headers, tablefmt="fancy_grid")
print(table)

  
# plot histogram of draws parameters
for i in range(lagp+2):
    plt.title(varnames[i]) 
    plt.ylabel('Frequency')
    plt.hist(x=drawpar[:,i], bins=25)
    plt.show()
        
plt.title('unconditional mean') 
plt.ylabel('Frequency')
plt.hist(x=uncmean, bins=25)
plt.show()
        

# plot histogram of draws forecasts
for i in range(vsp):
    plt.title(vspnames[i]) 
    plt.ylabel('Frequency')
    plt.hist(x=drawvsp[:,i], bins=25)
    plt.show()


