#!/usr/bin/python

from math import *
import sys, os, argparse, time
import numpy as np
#from scipy.optimize import curve_fit
from scipy.optimize import fmin_powell
import transforms3d.quaternions as qops
import transforms3d_supplement as qs
import plumedcolvario as pl
import xvgio
import dxio

import scipy.version as scipyVersion
from distutils.version import LooseVersion


def LegendreP1_quat(v_q):
    return 1 - 2*np.sum(np.square(v_q))

def LegendreP2_quat(v_q):
    return 1.5*(LegendreP1_quat(v_q))**2-0.5

def P1x(x):
    return 1-2*x**2

def P2x(x):
    return 1.5*(P1x(x))**2-0.5

def calculate_expectation_value_weighted(func, x, w):
    totn=len(x)
    totw=np.sum(w)
    val=np.zeros(totn)
    for i in range(totn):
        val[i]=func(x[i])
    avg=np.sum(np.multiply(val,w))/totw
    sigsq=np.sum(np.multiply(np.square(val),w))/totw - avg**2
    if ( sigsq >= 0 ):
        sig=sqrt( np.sum(np.multiply(np.square(val),w))/totw - avg**2 )
    else:
        sig=0.0
        print "= = WARNING: In calculate-expectation value, a negative standard deviation is found: %g +- sqrt(%g)" % (avg, sigsq)
        print "= = NOTES: totn: %i totw: %g" % (totn, totw)
    #print totw, avg, sig
    return avg, sig

def calculate_aniso_nosort(D):
    iso=(D[0]+D[1]+D[2])/3.0
    aniL  = 2*D[2]/(D[1]+D[0])
    rhomL = 1.5*(D[1]-D[0])/(D[2]-0.5*(D[1]+D[0]))
    aniS  = 2*D[0]/(D[1]+D[2])
    rhomS = 1.5*(D[1]-D[2])/(D[0]-0.5*(D[1]+D[2]))
    return (iso, aniL, rhomL, aniS, rhomS)

def calculate_anisotropies( D, chunkD=[]):
    if len(chunkD)==0:
        iso=(D[0]+D[1]+D[2])/3.0
        D_sorted = np.sort(D)
        aniL  = 2*D_sorted[2]/(D_sorted[1]+D_sorted[0])
        rhomL = 1.5*(D_sorted[1]-D_sorted[0])/(D_sorted[2]-0.5*(D_sorted[1]+D_sorted[0]))
        aniS  = 2*D_sorted[0]/(D_sorted[1]+D_sorted[2])
        rhomS = 1.5*(D_sorted[1]-D_sorted[2])/(D_sorted[0]-0.5*(D_sorted[1]+D_sorted[2]))
        return (iso, aniL, rhomL, aniS, rhomS)
    else:
        # Do some set magic
        block=np.concatenate( (np.reshape(D,(1,3)),chunkD) ).T
        block=block[block[:,0].argsort()]
        # calculate values from sub-chunks
        chunkdat=np.array( [ calculate_aniso_nosort( block[:,i]) for i in range(len(chunkD)+1)] )
        out=[ ( chunkdat[0,i],np.std(chunkdat[1:,i]) ) for i in range(5) ]
        # Output format [ (iso, isoErr), (aniL, aniLErr), (rhomL,ehomLErr), ... ]
        return out

def obtain_v_dqs(ndat, delta, q):
    out=np.zeros((ndat,3))
    for i in range(ndat):
        j=i+delta
        #Since everything is squared does quat_reduce matter? ...no, but do so anyway.
        dq=qs.quat_reduce(qops.qmult( qops.qinverse(q[i]), q[j]))
        out[i]=dq[1:4]
    return out

def obtain_self_dq(q, delta):
    """
    Vectorise { q^{-1}(t) q(t+delta) } over the range of q.
    Assumes that there are only two input dimensions q(N, 4) and returns dq(N-delta,4) with imaging to reduce rotations
    """
    #print q.shape, delta
    #return qs.vecnorm_NDarray( qs.quat_reduce_simd( qs.quat_mult_simd( qs.quat_invert(q[:-delta]), q[delta:]) ) )
    return qs.quat_reduce_simd( qs.quat_mult_simd( qs.quat_invert(q[:-delta]), q[delta:]) )

def average_LegendreP1quat(ndat, vq):
    out=0.0
    for i in range(ndat):
        out+=LegendreP1_quat(vq[i])
    return out/ndat

def average_anisotropic_tensor_old(vq, qframe=(1,0,0,0)):
    ndat=vq.shape[0]
    out=np.zeros((3,3))
    if qops.nearly_equivalent(qframe,(1,0,0,0)):
        for i in range(ndat):
            out+=np.outer(vq[i],vq[i])
    else:
        for i in range(ndat):
            vrot=qops.rotate_vector(vq[i],qframe)
            out+=np.outer(vrot,vrot)
    return out/ndat

def average_anisotropic_tensor(vq, qframe=(1,0,0,0)):
    """
    Given the list of N components of q_v in te shape of (N,3)
    Calculates the average tensor < q_i q_j >, i,j={x,y,z}
    """
    ndat=vq.shape[0]
    #print "= = Debug: vq shape", vq.shape
    # Check if there is no rotation.
    if not qops.nearly_equivalent(qframe,(1,0,0,0)):
        vq=qs.rotate_vector_simd(vq, qframe)
    return np.einsum('ij,ik->jk', vq, vq) / ndat

def average_LegendreP1quat_chunk(ndat, vq, nchunk):
    nblock=int(ceil(1.0*ndat/nchunk))
    out=np.zeros(nchunk)
    for i in range(nchunk):
        jmin=nblock*i
        jmax=min(ndat, nblock*(i+1))
        out[i]=average_LegendreP1quat(jmax-jmin, vq[jmin:jmax])
    return out

def average_anisotropic_tensor_chunk(vq, nchunk, qframe=(1,0,0,0)):
    ndat=vq.shape[0]
    nblock=int(ceil(1.0*ndat/nchunk))
    out=np.zeros((nchunk,3,3))
    for i in range(nchunk):
        jmin=nblock*i
        jmax=min(ndat, nblock*(i+1))
        out[i]=average_anisotropic_tensor(jmax-jmin, vq[jmin:jmax], qframe)
    return out

def isotropic_decay(x, a):
    return 1.5*np.exp(-x/a)-0.5

def anisotropic_decay_noc(x, a):
    return 0.5*np.exp(-x/a)+0.5

def powell_expdecay(pos, *args):
    """
    Fits y = C0 exp (x/A) + C1
    """
    x=args[0]
    y=args[1]
    C0=args[2]
    C1=args[3]
    A=pos
    chi2=0.0
    nval=len(x)
    for i in range(nval):
        ymodel=C0*exp(-x[i]/A)+C1
        chi2+=(ymodel-y[i])**2
        #print x[i], y[i], ymodel, A, C0, C1
    return chi2/nval

def obtain_guess_isotropic(vlist):
    #Try to guess by looking at the first two values and fitting the function to them.
    x0=vlist[0,0]; x1=vlist[1,0];
    y0=vlist[0,1]; y1=vlist[1,1];
    # When F(x) = Ae^(Bx)+C
    # log( (y1 - C)/(y0-C) ) = B (x0-x1)
    return (x0-x1)/log((y1+0.5)/(y0+0.5))

def obtain_guess_anisotropic(vlist):
    out=[]
    xdiff=vlist[0,0]-vlist[1,0]
    out.append( xdiff/log((vlist[1,1]-0.5)/(vlist[0,1]-0.5)) )
    out.append( xdiff/log((vlist[1,2]-0.5)/(vlist[0,2]-0.5)) )
    out.append( xdiff/log((vlist[1,3]-0.5)/(vlist[0,3]-0.5)) )
    return out

def build_model(func, args, xvals):
    out=[]
    for i in range(len(xvals)):
        out.append( func(xvals[i], args) )
    return out

def obtain_exponential_guess(x, y, C1):
    return (x[0]-x[1])/log((y[1]-C1)/(y[0]-C1))

# F(x) = C0 e ^ (-x/A) + C1.
def conduct_exponential_fit(xlist, ylist, C0, C1):
    print '= = Begin exponential fit.'
    xguess=[xlist[0],xlist[1]]
    yguess=[ylist[0],ylist[1]]
    guess=obtain_exponential_guess(xguess, yguess, C1)
    print '= = = guessed initial tau: ', guess
    fitOut = fmin_powell(powell_expdecay, guess, args=(xlist, ylist, C0, C1))
    print '= = = = Tau obtained: ', fitOut
    return fitOut

def get_flex_bounds(x, samples, nsig=1):
    """
    Here, we wish to report the distribution of the subchunks 'sample'
    along with the value of the full sample 'x'
    So this function will return x, x_lower_bound, x_upper_bound,
    where the range of the lower and upper bound expresses
    the standard deviation of the sample distribution, the mean
    of which is often not aligned with x.
    """
    mean=np.mean(samples); sig=np.std(samples)
    return [x, nsig*sig+x-mean, nsig*sig+mean-x]

def format_header(style_str, tau, taus=[]):
    l=[]
    if style_str=='iso':
        l.append('# model fit, tau = %e [ps]' % (tau) )
        Dval = 0.5e12/tau
        l.append("# Converted D_iso = %e [s^-1]" % (Dval))
        l.append("# t cos(th) P2[cos(th)] cos(th/2) th")
    elif style_str=='iso_err':
        # Print overall stats
        bound=get_flex_bounds(tau, taus)
        l.append('# model fit, tau = %e +- %e %e [ps]'  % (bound[0], bound[1], bound[2]))
        Dval = 0.5e12/tau ; Dvals= [0.5e12/taus[i] for i in range(len(taus)) ]
        bound=get_flex_bounds(Dval, Dvals)
        l.append('# Converted D_iso = %e +- %e %e [s^-1]' % (bound[0], bound[1], bound[2]))
        # Print chunk data:
        for i in range(len(taus)):
            l.append('# Chunk_%d D_iso = %e [s^-1]' % (i, Dvals[i]))
        l.append("# t cos(th) P2[cos(th)] cos(th/2) th")
    elif style_str=='aniso':
        Dval = 0.5e12/tau
        for i in range(3):
            l.append("# model fit, e_%i tau = %e [ps]" % (i, tau[i]))
            l.append("# Converted D_%i = %e [s^-1]" % (i, Dval[i]))
        anis=calculate_anisotropies( Dval )
        l.append("# Converted Diso = %e [s^-1]" % (anis[0]))
        l.append("# Converted Dani_L = %f" % (anis[1]))
        l.append("# Converted Drho_L = %f" % (anis[2]))
        l.append("# Converted Dani_S = %f" % (anis[3]))
        l.append("# Converted Drho_S = %f" % (anis[4]))
        l.append("# t <1-2x^2> <1-2y^2> <1-2z^2>")
    elif style_str=='aniso_err':
        # Print overall stats.
        Dval=0.5e12/tau
        Dvals=0.5e12/taus
        #Dvals=[ [0.5e12/taus[i][j] for j in range(len(taus[i]))] for i in range(len(taus)) ]
        print tau
        print taus
        print Dval
        print Dvals
        for i in range(3):
            bound=get_flex_bounds(tau[i], taus[:,i])
            l.append('# model fit, e_%i tau = %e +- %e %e [ps]'  % (i, bound[0], bound[1], bound[2]))
            bound=get_flex_bounds(Dval[i], Dvals[:,i])
            l.append('# Converted D_%i = %e +- %e %e [s^-1]' % (i, bound[0], bound[1], bound[2]))
        anis=calculate_anisotropies(Dval, Dvals)
        l.append("# Converted Diso = %e +- %e [s^-1]" % anis[0])
        l.append("# Converted Dani_L = %f +- %f" % anis[1])
        l.append("# Converted Drho_L = %f +- %f" % anis[2])
        l.append("# Converted Dani_S = %f +- %f" % anis[3])
        l.append("# Converted Drho_S = %f +- %f" % anis[4])
        # Print chunk data
        for j in range(len(taus)):
            for i in range(3):
                l.append('# Chunk_%d D_%d = %e [s^-1]' % (j, i, Dvals[j,i]))
        l.append("# t <1-2x^2> <1-2y^2> <1-2z^2>")
    return l

def format_header_quat(q):
    return '# Quaternion orientation frame: %f %f %f %f' % (q[0], q[1], q[2], q[3])

def print_model_fits_gen(fname, ydims, str_header, xlist, ylist):
    """
    Generic XVG  printout. Uses the dimension of y1list to determine
    how to print them. THe number of entries in each plot must be in the last dimension.
    - 1D ylist is a single plot
    - 2D ylist will have multiple plots on the samge graph
    - 3D ylist will have multiple graphs (first axis), on which multiple plots will exist(second axis).
    """
    g=0; s=0
    ndat=len(xlist)
    fp = open(fname, 'w')
    for line in str_header:
        print >> fp, "%s" % line
    if ydims==1:
        for i in range(ndat):
            print >> fp, "%g %g" % (xlist[i], ylist[i])
    elif ydims==2:
        dim1=len(ylist)
        for j in range(dim1):
            print >> fp, "@target g%d.s%d" % (g,s)
            for i in range(ndat):
                print >> fp, "%g %g" % (xlist[i], ylist[j][i])
            print >> fp, "&"
            s+=1
    elif ydims==3:
        dim1=len(ylist)
        print "dim1: ", dim1
        for k in range(dim1):
            print >> fp, "@g%d on" % g
            g+=1
        g=0
        for k in range(dim1):
            dim2=len(ylist[k])
            print "dim2: ", dim2
            for j in range(dim2):
                print >> fp, "@target g%d.s%d" % (g,s)
                for i in range(ndat):
                    print >> fp, "%g %g" % (xlist[i], ylist[k][j][i])
                print >> fp, "&"
                s+=1
            g+=1 ; s=0
    else:
        print "= = = Critical ERROR: invalid dimension specifier in print_model_fits_gen!"
        sys.exit(1)
    fp.close()

def print_model_fits_iso(fname, xlist, ylist, model, tau):
    ndat=len(xlist)
    fp = open(fname, 'w')
    print >> fp, "# model fit, tau = %g " % (tau)
    Dval = 0.5e12/tau
    #Derr = 0.0
    #Derr = Dval*(tau_error/tau)
    print >> fp, "# Converted D_iso = %g [s^-1]" % (Dval)
    print >> fp, "# t cos(th) P2[cos(th)] cos(th/2) th"
    for i in range(ndat):
        print >> fp, "%g %g" % (xlist[i], ylist[i])
    print >> fp, "&"
    for i in range(len(out_isolist)):
        print >> fp, "%g %g" % (xlist[i], model[i])
    print >> fp, "&"
    fp.close()

def print_model_fits_aniso(fname, xlist, ylist, model, tau):
    ndat=len(xlist)
    fp = open(fname, 'w')
    for i in range(3):
        print >> fp, "# model fit, e_%i tau = %g [ps]" % (i, tau[i])
        Dval = 0.5e12/tau[i]
        print >> fp, "# Converted D_%i = %g [s^-1]" % (i, Dval)
    print >> fp, "# t <1-2x^2> <1-2y^2> <1-2z^2>"
    for i in range(ndat):
        print >> fp, "%g %g %g %g" % (xlist[i],ylist[0][i],ylist[1][i],ylist[2][i])
    print >> fp, "&"
    for i in range(ndat):
        print >> fp, "%g %g %g %g" % (xlist[i], model[0][i], model[1][i], model[2][i])
    print >> fp, "&"
    fp.close()

def write_to_gnuplot_hist(in_fname, hist, edges):
    fp = open(in_fname, 'w')
    dim=len(edges)
    nbin_vec=[ len(edges[i])-1 for i in range(dim) ]
    # For Gnuplot, we should plot the average between the two edges.
    # ndemunerate method iterates across all specified dimensions
    # First gather and print some data as comments to aid interpretation and future scripting.
    # Use final histogram output as authoritative source!
    print >> fp, '# DIMENSIONS: %i' % dim
    binwidth=np.zeros(dim)
    print >> fp, '# BINWIDTH: ',
    for i in range(dim):
        binwidth[i]=(edges[i][-1]-edges[i][0])/nbin_vec[i]
        print >> fp, '%g ' % binwidth[i],
    print >> fp, ''
    print >> fp, '# N_BINS: ',
    for i in range(dim):
        print >> fp, '%g ' % nbin_vec[i],
    print >> fp, ''
    for index, val in np.ndenumerate(hist):
        for i in range(dim):
            x=(edges[i][index[i]]+edges[i][index[i]+1])/2.0
            print >> fp, '%g ' % x ,
        print >> fp, '%g' % val
        if index[-1] == nbin_vec[-1]-1:
            print >> fp, ''
    fp.close()

def print_list(fname, list):
    fp = open(fname, 'w')
    #nfield=len(list[0])
    #fmtstr="%g "*nfield
    for i in range(len(list)):
        for j in range(len(list[i])):
            print >> fp, "%10f " % float(list[i][j]),
            #print >> fp, fmtstr[:-1] % list[i]
        print >> fp , ""
    print >> fp, "&"
    fp.close()

def print_xylist(fname, xlist, ylist):
    fp = open(fname, 'w')
    for i in range(len(xlist)):
        print >> fp, "%10f " % float(xlist[i]),
        for j in range(len(ylist[i])):
            print >> fp, "%10f " % float(ylist[i][j]),
            #print >> fp, fmtstr[:-1] % ylist[i]
        print >> fp , ""
    print >> fp, "&"
    fp.close()

def print_axes_as_xyz(fname, mat):
    fp = open(fname, 'w')
    for i in range(len(mat)):
        print >> fp, "3"
        print >> fp, "AXES"
        print >> fp, "X %g %g %g" % (out_moilist[i][0,0], out_moilist[i][0,1], out_moilist[i][0,2])
        print >> fp, "Y %g %g %g" % (out_moilist[i][1,0], out_moilist[i][1,1], out_moilist[i][1,2])
        print >> fp, "Z %g %g %g" % (out_moilist[i][2,0], out_moilist[i][2,1], out_moilist[i][2,2])
    fp.close()

def debug_orthogonality(axes):
    print "= = = Orthogonality check: "
    print "%f, %f, %f\n" % \
        (np.dot(axes[0],axes[1]),
         np.dot(axes[0],axes[2]),
         np.dot(axes[1],axes[2]))

if __name__ == '__main__':

    #= = = Test for whether scipy accepts bounds. This is a dvelopment in 0.17.0
    print "= = scipy version: ", scipyVersion.version
    # Irrelevant, now that we are using the more robust fmin_powell
    #if LooseVersion(scipyVersion.version) < LooseVersion("0.17.0"):
    #    print "= = = WARNING: SciPy version does not contain bounded curve_fit! Requires >= 0.17.0."
    #    print "= = =          ...will run unbounded version of curve fitting..."
    #else:
    #    print "= = = Note: SciPy version contains bounded curve_fit."

    scriptname=os.path.basename(__file__)

    parser = argparse.ArgumentParser(description='Calculates the difference quaternions from PLUMED output: '
                                     'a time-series of quaternion representation of orientations '
                                     'then manipulate them in various ways',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-f', '--infn', type=str, dest='infn', default='colvar-q',
                        help='Input file in PLUMED form, '
                             'containing a header lines of format "#! FIELD <time> <quaternion>, '
                             'and then space-delimited data columns. Comment chars "#".'
                             'Assumes that dt is identical between every frame! (be careful).')
    parser.add_argument('-o', '--outpref', type=str, dest='out_pref', default='out',
                        help='Output file prefix for the histograms and/or quaternion-decay curves.')
    parser.add_argument('--hist', dest='bDoHist', action='store_true', default=False,
                         help='Record the 3D-histogram of rotation-quaternions dq at each delay time, dt.')
    parser.add_argument('-o2', '--outtype', type=str, dest='out_suff', default='dat',
                        help='File format to store the histograms in, options are dx (VMD), and dat (GNUPLOT).')
    parser.add_argument('--iso', dest='bDoIso', action='store_true', default=False,
                         help='Record the isotropic decay of dq, evaluated as <cos(theta)> and <P2(cos(theta))>'
                              'where q = ( cos(theta/2), sin(theta/2)*v_q )')
    parser.add_argument('--aniso', dest='bDoAniso', action='store_true', default=False,
                         help='Record an estimate of the anisotropic decay of dq,'
                              'using the coordinate axes as a guide.')
    parser.add_argument('-n','--num_bins', type=int, dest='num_bins', default=101,
                         help='Number of bins in the histogram spanning [-1,1] in the elements of q-space.')
    parser.add_argument('--mindt','--min_dt', type=float, dest='min_dt', default=0.0,
                        help='Minimum interval delta_t to calculate in picoseconds [ps].'
                             'This will be calculated as the maximum between the given value '
                             'and the minimum interval found in the dataset.')
    parser.add_argument('--num_chunk', type=int, dest='num_chunk', default=0,
                        help='Do some uncertainty estimation by calculating values for subchunks of the given trajectory.'
                             'This reports the standard devation of those subchunks as well as their plots.')
    parser.add_argument('--maxdt','--max_dt', type=float, dest='max_dt', default=1000.0,
                        help='Maximum interval delta_t to calculate in picoseconds [ps].'
                             'This determines the maximum interval (t) <-> (t+dt) to extract from the data.')
    parser.add_argument('--skip','--skip_dt', type=float, dest='skip_dt', default=0.0,
                        help='The interval of time in which the calculation should be carried out. '
                             'This is in the same time units as above, which the script will convert to frames, '
                             'as close as possible. ')

    time_start=time.clock()
    args = parser.parse_args()

    bDoTest=False
    bDoSubchunk=False
    bFit=True
    in_fname=args.infn
    out_pref=args.out_pref
    out_suff=args.out_suff
    if out_suff != "dx" and out_suff != "dat" and out_suff != "none":
        print "= = ERROR in input: histogram output type must be either dx, or dat, or none."
        sys.exit()
    min_dt=args.min_dt
    max_dt=args.max_dt
    skip_dt=args.skip_dt

    bDoHist=args.bDoHist
    nbins=args.num_bins
    bDoIso=args.bDoIso
    bDoAniso=args.bDoAniso
    num_chunk=args.num_chunk
    if num_chunk>1:
        bDoSubchunk=True


    fact=180.0/np.pi

    #= = = Read inputs and prepare for matrix ops.
    fields, data = pl.read_from_plumedprint_multi(in_fname)
    #data = np.swapaxes(data,1,2)
    data_shape     = data.shape
    num_replicas   = data_shape[0]
    num_timepoints = data_shape[1]

    print data_shape
    print "= = Input data found to be as %i replicate trajectories, each with %i fields and %i entries. = =" \
            % (num_replicas, len(fields[0]) , num_timepoints)

    # = = Make sure there is sensible error analysis by making sure sub-chunk analysis does not cut across entries..?
    if num_chunk > 1 and num_replicas % num_chunk != 0:
        print >> sys.stderr, "= = ERROR: Uncertainty estimation is not appropriately handled: the number of sub_chunks must divide the total number of replicate trajectories! num_chunk: %d num_replicas: %d" % (num_chunk, num_replicas)
        sys.exit(1)


    print data[0,0,1:5]
    qprev=data[0,0,1:5]
    print "= = Initial quaternion read: ", str(qprev).strip('[]'), " = ="

    # = = Translate input dt to the frame deltas = =
    data_delta_t=data[0,1,0]-data[0,0,0]
    skip_int=max(1,int(skip_dt/data_delta_t))
    min_int=max(skip_int,int(min_dt/data_delta_t))
    max_int=int(max_dt/data_delta_t)
    num_int=int(np.floor((max_int-min_int)/skip_int)+1)
    report_interval=max(1,int(num_int*0.05))
    min_delta_t=min_int*data_delta_t
    max_delta_t=max_int*data_delta_t
    print "= = Will calculate statistics for %i intervals between %g - %g ps, every %g ps) = =" % (num_int, min_delta_t, max_delta_t, skip_dt)
    print "= = ...corresponding to %i - %i frames, every %i frames. = =" % (min_int, max_int, skip_int)
    #No longer check if the maximum correlation time is longer than half the trajectory...


    arange=[(-1,1), (-1,1), (-1,1)]
    nbin_vec=(nbins,nbins,nbins)
    zeroloc=(nbins-1)/2

    #Prepare for rotated Daniso measurements.
    q_frame=(1,0,0,0)
    bFirst=True
    bFirstSub=True
    # = = Main loop over frame intervals.
    nIntervals=(max_int-min_int)/skip_int+1
    out_dtlist=np.zeros(nIntervals)
    out_isolist=np.zeros(nIntervals)
    out_aniso1list=np.zeros((3,nIntervals))
    out_aniso2list=np.zeros((3,nIntervals))
    out_qlist=np.zeros((nIntervals,4))
    out_moilist=np.zeros((nIntervals,3,3))
    if bDoSubchunk:
        chunk_isolist=np.zeros((num_chunk,nIntervals))
        #chunk_aniso1list=np.zeros((num_chunk,3,nIntervals))
        chunk_aniso2list=np.zeros((num_chunk,3,nIntervals))
        #chunk_qlist=np.zeros((num_chunk,nIntervals,4))
        #chunk_moilist=np.zeros((num_chunk,nIntervals,3,3))

    # Start at the same location: go through each lag time delta_t and accumulat data.
    index=0
    for index, delta in enumerate( range(min_int,max_int+1,skip_int) ):
        time_delta=delta*data_delta_t

        #Gather the individual samples of angular displacements.
        #Iterate through each chunk
        num_one=num_timepoints-delta
        num_nd=num_one*num_replicas
        v_dq=np.zeros((num_nd,3), dtype=data.dtype)
        for ch in range(num_replicas):
            chunk = np.delete(np.take(data, indices=ch, axis=0), 0, axis=1)
            v_dq[num_one*ch:num_one*(ch+1)]=obtain_self_dq( chunk, delta )[...,1:4]

        #Isotropic diffusion, use theta_q
        iso_sum1=average_LegendreP1quat(num_nd, v_dq)
        #Anisotropic Diffusion. Need tensor of vq
        moi=average_anisotropic_tensor(v_dq)
        if not qops.nearly_equivalent(q_frame,(1,0,0,0)):
            moiR1=average_anisotropic_tensor(v_dq, q_frame)
        else:
            moiR1=moi
        print " = = %i of %i intervals summed." % ((delta-min_int)/skip_int+1, nIntervals)

        #Append to overall list in next outer loop.
        out_dtlist[index]=time_delta
        if bDoIso:
           out_isolist[index]=iso_sum1
        if bDoAniso:
            eigval, eigvec = np.linalg.eigh(moi)
            moi_axes=eigvec.T
            #Numpy returns the axes as column vectors.
            if False:
                debug_orthogonality(moi_axes)

            #rotations to change REF frame into MoI frame
            q_rot=qs.quat_frame_transform_min(moi_axes)
            #Store first quaternion for subsequent data points
            if bFirst:
                bFirst=False
                q_frame=q_rot
                # The answer will be the same as simply eigenvals in the matching frame,
                # but we'll calculate it again anyway as a debug.
                moiR1=average_anisotropic_tensor(v_dq, q_frame)
                print "= = Begin Debug."
                print "= = = PAF Axes in REF frame:"
                print moi_axes[0], moi_axes[1], moi_axes[2]
                print "= = = Eigenvalues in REF frame and PAF frame:"
                print eigval
                print "= = = FRAME rotation from REF frame to PAF frame."
                print q_rot, moi_axes[0], moi_axes[1], moi_axes[2]
                print "= = = deltaQ Moment of Inertia tensor in REF frame:"
                print moi
                print "= = = deltaQ Moment of Inertia tensor in MoI frame:"
                print moiR1
                print "= = Debug finished."

            # In the PAF frame, the eigenvalues *are* the cartesian components of <r^2>
            out_aniso1list[:,index]=[  1-2*eigval[0],  1-2*eigval[1],  1-2*eigval[2]]
            out_aniso2list[:,index]=[ 1-2*moiR1[0,0], 1-2*moiR1[1,1], 1-2*moiR1[2,2]]
            out_qlist[index]     = q_rot
            out_moilist[index]   = moi_axes


        # Do sub-chunking when errors are involved
        if bDoSubchunk:
            chunk_isolist[:,index]=average_LegendreP1quat_chunk(num_nd, v_dq, num_chunk)
            #Anisotropic Diffusion. Need tensor of vq
            #temp=average_anisotropic_tensor_chunk(num_nd, v_dq, num_chunk)
            #chunk_aniso1list[:,:,index]=[ temp[i][0,0], temp[i][1,1], temp[i][0,0]
            if not qops.nearly_equivalent(q_frame,(1,0,0,0)):
                temp2=average_anisotropic_tensor_chunk(v_dq, num_chunk, q_frame)
            else:
                temp2=average_anisotropic_tensor_chunk(v_dq, num_chunk)

            chunk_aniso2list[:,:,index]=[ [1-2*temp2[i][0,0], 1-2*temp2[i][1,1], 1-2*temp2[i][2,2]]
                                          for i in range(num_chunk) ]
            #print "= = Debug:"
            #print "chunk_isolist"
            #print chunk_isolist[:,index]
            #print "chunk_aniso2list"
            #print chunk_aniso2list[:,:,index]

        # Must write a histogram for each frame interval.
        if bDoHist:
            hist, edges = np.histogramdd(v_dq, range=arange, bins=nbin_vec, normed=True)
            out_file=out_pref+"-hist-"+str(delta*data_delta_t)+"ps."+out_suff
            #DX output
            if out_file.endswith("dx"):
                xmin=[ (edges[0][0]+edges[0][1])/2.0,
                       (edges[1][0]+edges[1][1])/2.0,
                       (edges[2][0]+edges[2][1])/2.0 ]
                abc=np.zeros((3,3))
                for i in range(3):
                    abc[i][i]=(edges[i][-1]-edges[i][0])/nbin_vec[i]
                dxio.write_to_dx(out_file, hist, nbin_vec, xmin, abc, 'nm')
            #Gnuplot output
            elif out_file.endswith("dat"):
                write_to_gnuplot_hist(out_file, hist, edges)

        #= = End of main for-loop governing a single delta t = =

    #= = End of main for-loop governing all delta t = =

    #Transpose the anisotropics so that x/y/z are now first indices, and time second.
    #out_aniso1list=out_aniso1list.T
    #out_aniso2list=out_aniso2list.T
    #Quaternions and full Matrices remain time first.
    time_chk1=time.clock()

    if bDoIso:
        tau=conduct_exponential_fit(out_dtlist, out_isolist, 1.5, -0.5)
        model=build_model(isotropic_decay,tau,out_dtlist)

        #Check if errors are also to be done.
        if bDoSubchunk:
            chtaus=[ conduct_exponential_fit(out_dtlist, chunk_isolist[i], 1.5, -0.5)
                   for i in range(num_chunk) ]
            chmodels=[ build_model(isotropic_decay,chtaus[i],out_dtlist)
                     for i in range(num_chunk)]
            printlist=[[out_isolist, model]]
            for i in range(num_chunk):
                printlist.append([chunk_isolist[i],chmodels[i]])
            file_header=format_header('iso_err', tau, chtaus)
            print_model_fits_gen(out_pref+"-iso.dat", 3,
                             file_header, out_dtlist, printlist)
        else:
            file_header=format_header('iso', tau)
            print_model_fits_gen(out_pref+"-iso.dat", 2,
                             file_header, out_dtlist, [out_isolist, model] )
            #print_model_fits_iso(out_pref+"-iso.dat",
            #                     out_dtlist, out_isolist, model, tau)

    if bDoAniso:
        # Model 1 where the frame is always rotates so as to diagonalise the distribution.
        #taus=[ conduct_exponential_fit(out_dtlist,out_aniso1list[i], 0.5, 0.5)
        #       for i in range(3) ]
        #models=[ build_model(anisotropic_decay_noc, taus[i], out_dtlist)
        #         for i in range(3) ]
        #print_model_fits_aniso(out_pref+"-aniso1.dat",
        #                 out_dtlist, out_aniso1list, models, taus)

        #Model 2 where the frame is rotated to match a particular PAF frame.
        taus=[ conduct_exponential_fit(out_dtlist,out_aniso2list[i], 0.5, 0.5)
               for i in range(3) ]
        taus=np.array(taus)
        models=[ build_model(anisotropic_decay_noc, taus[i], out_dtlist)
                 for i in range(3) ]
        #print_model_fits_aniso(out_pref+"-aniso2.dat",
        #                 out_dtlist, out_aniso2list, models, taus)
        if bDoSubchunk:
            chtaus=[[ conduct_exponential_fit(out_dtlist, chunk_aniso2list[i][j], 0.5, 0.5)
                   for j in range(3) ]
                   for i in range(num_chunk) ]
            chtaus=np.array(chtaus)
            chmodels=[[ build_model(anisotropic_decay_noc,chtaus[i][j],out_dtlist)
                     for j in range(3) ]
                     for i in range(num_chunk)]
            file_header=format_header('aniso_err', taus, chtaus)
            file_header.append( format_header_quat(q_frame) )
            printlist=[ np.concatenate((out_aniso2list, models)) ]
            for i in range(num_chunk):
                printlist.append( np.concatenate((chunk_aniso2list[i],chmodels[i])) )
            print_model_fits_gen(out_pref+"-aniso2.dat", 3,
                             file_header, out_dtlist, printlist)
        else:
            file_header=format_header('aniso', taus)
            file_header.append( format_header_quat(q_frame) )
            print_model_fits_gen(out_pref+"-aniso2.dat", 2,
                             file_header, out_dtlist, np.concatenate((out_aniso2list, models)) )

        #Print optimal fit quaternions
        print_xylist(out_pref+"-aniso_q.dat", out_dtlist, out_qlist)

        #Print axis vectors
        print_axes_as_xyz(out_pref+"-moi.xyz", out_moilist)

    time_stop=time.clock()
    #Report time
    print "= = Total seconds elapsed: %g" % (time_stop - time_start)
    print "= = Time of Read and fit halves: %g , %g" % ( time_chk1-time_start, time_stop-time_chk1 )
