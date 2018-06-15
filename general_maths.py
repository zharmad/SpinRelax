import numpy as np
import sys

def gather(a, b):
    return np.sum(np.multiply(a, b))

def calc_central_moments(x, y, bSymm ):
    ctot = np.sum(y)
    mu=np.zeros(4, dtype=type(ctot))
    if bSymm:
        mu[1] = gather( y, np.power(x,2) ) / ctot
        mu[3] = gather( y, np.power(x,4) ) / ctot
        return mu
    else:
        EX1 = gather( y, x ) / ctot
        EX2 = gather( y, np.power(x,2) ) / ctot
        EX3 = gather( y, np.power(x,3) ) / ctot
        EX4 = gather( y, np.power(x,4) ) / ctot
        mu[0] = EX1
        mu[1] = EX2 - np.power(EX1,2)
        mu[2] = EX3 - 3*EX1*EX2 + 2*np.power(EX1,3)
        mu[3] = EX4 - 4*EX1*EX3 + 6*np.power(EX1,2)*EX2 - 3*np.power(EX1,4)
        return mu

def anova_total_mean_square(Ns, means, sigmas):
    """
    This function performs an average over multiple sets of observations, each with its own standard deviation.
    For example: 5 simulations compute RMSDs+-Sigma of the same protein. What is the aggregate?
    See: http://www.burtonsys.com/climate/composite_standard_deviations.html
    Here, Ns = the number of observations in each copy, means +- sigma are the values of each.
    """
    copies=len(Ns)
    grand_total = np.sum( Ns )
    grand_mean = np.multiply(Ns, means) / grand_total
    GSS = np.sum( [ Ns[i]*(means[i] - grand_mean)**2.0 for i in range(copies) ] )
    ESS = np.sum( [ (Ns[i]-1)*sigmas[i]**2.0 for i in range(copies) ] )
    return (GSS+ESS)/(grand_total-1)

def simple_total_mean_square(means, sigmas):
    """
    This version of the above assumes that each sample has numerous observations, and that
    alls samples have the same number of observations each.
    """
    copies=len(means)
    grand_mean = np.mean( means )
    GSS = np.sum( [ (means[i] - grand_mean)**2.0 for i in range(copies) ] )
    ESS = np.sum( [ sigmas[i]**2.0 for i in range(copies) ] )
    return (GSS+ESS)/copies

def weighted_average_stdev(values, weights, axis=-1):
    """
    Computes the weighted average and stdev when values are weighted.
    values  in N   dimensions
    weights in N-1 dimensions
    """
    #print type(values), values.shape
    #print type(weights), weights.shape
    avg   = np.average(values, axis=axis, weights=weights)
    stdev = np.sqrt( np.average( (values-avg)**2.0, axis=axis, weights=weights) )
    return avg, stdev


def _perturb_tuple(t,mod,axis):
    l = list(t)
    l[axis]+=mod
    return tuple(l)

def xyz_to_rtp(uv, vaxis=-1, bUnit=False ):
    """
    Converts a vector or a set of vectors from the X/Y/Z to R/Phi/Theta.
    Noting that 0 ~ Theta ~ pi from positive Z.
    vaxis denotes the dimension in which the X/Y/Z value resides.
    This is the first (0) or last (-1) dimension of the array.
    """
    sh = uv.shape
    dims= len(sh)
    if bUnit:
        if dims == 1:
            rtp = np.zeros(2, dtype=uv.dtype)
            rtp[0] = np.arctan2(uv[1], uv[0])
            rtp[1] = np.arccos(uv[2]/rtp[0])
        elif vaxis==-1:
            rtp = np.zeros(_perturb_tuple(sh,mod=-1,axis=-1), dtype=uv.dtype)
            rtp[...,0] = np.arctan2(uv[...,1], uv[...,0])
            rtp[...,1] = np.arccos(uv[...,2]/rtp[...,0])
        elif vaxis==0:
            rtp = np.zeros(_perturb_tuple(sh,mod=-1,axis=0), dtype=uv.dtype)
            rtp[0,...] = np.arctan2(uv[1,...], uv[0,...])
            rtp[1,...] = np.arccos(uv[2,...]/rtp[0,...])
        else:
            print >> sys.stderr, "= = ERROR encountered in vec-to-rtp in general_maths.py, vaxis only accepts arguments of -1 or 0 for now."
    else:
        rtp = np.zeros_like(uv)
        if dims == 1:
            rtp[0] = np.linalg.norm(uv)
            rtp[1] = np.arctan2(uv[1], uv[0])
            rtp[2] = np.arccos(uv[2]/rtp[0])
        elif vaxis==-1:
            rtp[...,0] = np.linalg.norm(uv,axis=-1)
            rtp[...,1] = np.arctan2(uv[...,1], uv[...,0])
            rtp[...,2] = np.arccos(uv[...,2]/rtp[...,0])
        elif vaxis==0:
            rtp[0,...] = np.linalg.norm(uv,axis=0)
            rtp[1,...] = np.arctan2(uv[1,...], uv[0,...])
            rtp[2,...] = np.arccos(uv[2,...]/rtp[0,...])
        else:
            print >> sys.stderr, "= = ERROR encountered in vec-to-rtp in general_maths.py, vaxis only accepts arguments of -1 or 0 for now."
    return rtp

def rtp_to_xyz(rtp, vaxis=-1 , bUnit=False ):
    """
    Converts a vector or a set of vectors from R/Phi/Theta to X/Y/Z.
    Noting that 0 ~ Theta ~ pi from positive Z.
    vaxis denotes the dimension in which the X/Y/Z value resides.
    This is the first (0) or last (-1) dimension of the array.
    If bUnit, expect only Phi/Theta instead of R/Phi/THeta
    """
    sh = rtp.shape
    dims = len(sh)
    if bUnit:
        if dims == 1:
            uv = np.zeros(3, dtype=rtp.dtype)
            uv[0]=np.cos(rtp[0])*np.sin(rtp[1])
            uv[1]=np.sin(rtp[0])*np.sin(rtp[1])
            uv[2]=np.cos(rtp[1])
        elif vaxis == -1:
            uv = np.zeros( _perturb_tuple(sh,mod=1,axis=-1) , dtype=rtp.dtype)
            uv[...,0]=np.cos(rtp[...,0])*np.sin(rtp[...,1])
            uv[...,1]=np.sin(rtp[...,0])*np.sin(rtp[...,1])
            uv[...,2]=np.cos(rtp[...,1])
        elif vaxis == 0:
            uv = np.zeros( _perturb_tuple(sh,mod=1,axis=0) , dtype=rtp.dtype)
            uv[0,...]=np.cos(rtp[0,...])*np.sin(rtp[1,...])
            uv[1,...]=np.sin(rtp[0,...])*np.sin(rtp[1,...])
            uv[2,...]=np.cos(rtp[1,...])
        else:
            print >> sys.stderr, "= = ERROR encountered in rtp-to-vec in general_maths.py, vaxis only accepts arguments of -1 or 0 for now."
    else:
        uv = np.zeros_like( rtp )
        if dims == 1:
            uv[0]=rtp[0]*np.cos(rtp[1])*np.sin(rtp[2])
            uv[1]=rtp[0]*np.sin(rtp[1])*np.sin(rtp[2])
            uv[2]=rtp[0]*np.cos(rtp[2])
        elif vaxis == -1:
            uv[...,0]=rtp[0]*np.cos(rtp[...,1])*np.sin(rtp[...,2])
            uv[...,1]=rtp[0]*np.sin(rtp[...,1])*np.sin(rtp[...,2])
            uv[...,2]=rtp[0]*np.cos(rtp[...,2])
        elif vaxis == 0:
            uv[0,...]=rtp[0]*np.cos(rtp[1,...])*np.sin(rtp[2,...])
            uv[1,...]=rtp[0]*np.sin(rtp[1,...])*np.sin(rtp[2,...])
            uv[2,...]=rtp[0]*np.cos(rtp[2,...])
        else:
            print >> sys.stderr, "= = ERROR encountered in rtp-to-vec in general_maths.py, vaxis only accepts arguments of -1 or 0 for now."

    return uv
