import numpy as np
import scipy.signal as sig

def diag_dist(x, cp):
    err = np.zeros(len(x))
    for i1, i2 in zip(cp[:-1],cp[1:]):
        sl = (x[i2]-x[i1])/(i2-i1)
        err[i1:i2] = sl*(np.arange(i1,i2)-i1)+x[i1]-x[i1:i2]
    return(err)

def poly_dist(x, cp):
    err = np.zeros(len(x))
    for i1, i2 in zip(cp[:-1],cp[1:]):
        ii = np.arange(i1,i2+1)
        pp = np.polyfit(ii-i1,x[ii],1)
        err[ii] = np.polyval(pp,ii-i1)-x[ii]
    return(err)


def max_diag_dist(x,cp,mindist=0):
    err = diag_dist(x,cp)
    if mindist>0:
        for pt in cp:
            err[max(pt-mindist,0):min(pt+mindist,len(err))]=0
    return np.argmax(err**2)

def changepoints(x, n, details=False, mindist=0):
    cp = np.array([0,len(x)-1])

    dists=[(0,np.sqrt(np.sum(diag_dist(x,cp)**2)))]

    for ii in range(n):
        newcp = max_diag_dist(x, cp,mindist=mindist)
        idx = np.flatnonzero(cp<=newcp)[-1]+1
        cp=np.insert(cp,idx,newcp)
        dists.append((newcp,np.sqrt(np.sum(diag_dist(x,cp)**2))))
        
    ret = cp
    if details:
        ret = (cp,dists)
    return ret

    
