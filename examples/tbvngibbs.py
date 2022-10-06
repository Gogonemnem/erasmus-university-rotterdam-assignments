# Python program to simulate (theta_1,theta_2) given theta_1 < theta_2 from a bivariate normal distribution using the Gibbs sampler 
# Input is the mean and covariance matrix of the normal distribution. 
# The output of the program are trace plots of the draws.

import numpy as np
import matplotlib.pyplot as  plt
from statsmodels.graphics.tsaplots import plot_acf
from scipy.stats import norm

nos = 10000        # number of simulations 
nob = 1000;        # number of burn-in simulations 

mu = [1,2]              # mean parameters of (theta_1,theta_2) 
sigma = np.array([[1,0.5],[0.5,2]])   # covariance matrix of (theta_1,theta_2) 

theta = [0,1]      # starting values  of Gibbs sampler

drawpar = np.zeros((nos+nob,2))          # matrix to store Gibbs draws 

np.random.seed(0) # set random seed

# Gibbs sampler
for i in range(nob+nos):
    
    if i%1000==0: print(i)
  

    mean = (mu[0]+sigma[0,1]*(theta[1]-mu[1])/sigma[1,1])         # conditional mean theta_1
    var = sigma[0,0]-sigma[0,1]*sigma[0,1]/sigma[1,1]             # conditional variance  theta_1
    lb = 0                                                        # lower bound theta_1
    ub = norm.cdf((theta[1]-mean)/np.sqrt(var))                   # upper bound theta_1
    theta[0] = mean+np.sqrt(var)*norm.ppf(lb+(ub-lb)*np.random.uniform())     # simulate theta_1 given theta_2  
        
    mean = (mu[1]+sigma[0,1]*(theta[0]-mu[0])/sigma[0,0])         # conditional mean theta_2
    var = sigma[1,1]-sigma[0,1]*sigma[0,1]/sigma[0,0]             # conditional variance theta_2
    lb = norm.cdf((theta[0]-mean)/np.sqrt(var))                   # lower bound theta_2
    ub = 1;                                                       # upper bound theta_2
    theta[1] = mean+np.sqrt(var)*norm.ppf(lb+(ub-lb)*np.random.uniform())      # simulate theta_2 given theta_1 

    drawpar[i] = theta                                            # store Gibbs draws 


# trace plots of first 100 draws 
fig, (ax1, ax2) = plt.subplots(2, 1)
ax1.set_title('Trace plot of first 100 draws of theta_1')
ax2.set_title('Trace plot of first 100 draws of theta_2')
ax1.plot(drawpar[0:99,0])
ax2.plot(drawpar[0:99,1])
fig.tight_layout()
plt.show()

# plot of first 100 draws in 2 dimensional graph
plt.xlabel('theta_1')
plt.ylabel('theta_2')
plt.title('First 100 draws in 2-dimensional graph')
plt.plot(drawpar[0:99,0],drawpar[0:99,1])
plt.show()


drawpar = drawpar[nob:]  # remove burn-in draws

# trace plots of draws after removing burnin
fig, (ax1, ax2) = plt.subplots(2, 1)
ax1.set_title('Trace plot of theta_1 draws after burn-in period')
ax2.set_title('Trace plot of theta_2 draws after burn-in period')
ax1.plot(drawpar[:,0])
ax2.plot(drawpar[:,1])
fig.tight_layout()
plt.show()

plt.xlabel('theta_1')
plt.ylabel('theta_2')
plt.title('Draws in 2-dimensional graph')
plt.plot(drawpar[:,0],drawpar[:,1])
plt.show()

# autocorrelation of theta_1 draws
plot_acf(drawpar[:,0],lags=10)
plt.show()

