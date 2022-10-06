import numpy as np
from scipy.stats import norm

def chpdi(y,p):  
    draws=y+0
    nrows=len(draws) 
    ncols=len(draws[0])
    nod = round((1-p)*nrows);   # number of draws outside interval
    if ((p<1) and (p>0) and (nod < nrows-2)):
        draws=np.sort(draws,axis=0)
        lb = draws[1:nod]                    # determine potential lower bounds
        ub = draws[nrows-nod+1:nrows]        # determine potential upper` bounds
        isize = ub-lb      # compute interval sizes
        isize = np.argmin(isize,axis=0)   # choose minimum interval   
        hpdi=np.zeros((ncols,2))
        for i in range(ncols):
            hpdi[i,0]= lb[isize[i],i]
            hpdi[i,1]= ub[isize[i],i]
    else:
       hpdi=np.zeros((ncols,2))
       hpdi[:]=np.nan 
       
    return(hpdi)
    


    
def gewtest(y):
    draws=y+0
    nrows=len(draws) 
    mean1=np.mean(draws[0:round(0.1*nrows)],axis=0)
    mean2=np.mean(draws[round(0.5*nrows):],axis=0)
    var1=np.var(draws[0:round(0.1*nrows)],axis=0)/round(0.1*nrows)
    var2=np.var(draws[round(0.5*nrows):],axis=0)/round(0.5*nrows)
    test=(mean1-mean2)/np.sqrt(var1+var2)
    pval=norm.cdf(-abs(test))
    return(np.column_stack((mean1,mean2,np.sqrt(var1+var2),test,pval)))
    
def perc(y,p):
    draws=y+0
    if (p<1) and (p>0):
        draws=np.sort(draws,axis=0)
    else:       
        draws[:]=np.nan
    return(draws[round(p*len(draws))-1])    
    
    
    