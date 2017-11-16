
from math import *
import sys, os, argparse, time
import numpy as np
from scipy.optimize import curve_fit
import mdtraj as md
import general_scripts as gs
import general_maths as gm
import transforms3d.quaternions as q_ops
import transforms3d_supplement as qs

def print_selection_help():
    print "Notes: This python program uses MDTraj as its underlying engine to analyse trajectories and select atoms."
    print "It uses selection syntax such as 'chain A and resname GLY and name HA1 HA2', in a manner similar to GROMACS and VMD."

def P2(x):
    return 1.5*x**2.0-0.5

def normalise_vecs(v, ax):
    n=np.linalg.norm(v,axis=ax)
    shape=v.shape
    return np.array([ [v[i,j]/n[i,j] for j in range(shape[1])] for i in range(shape[0]) ])

def get_spherical_coords(uv):
    """
    Note: this does not assume unit vectors, and so should be safe.
    """
    sh = uv.shape
    rtp = np.zeros(sh)
    rtp[:,:,0] = np.linalg.norm(uv,axis=2)
    rtp[:,:,1] = np.arctan2(uv[:,:,1], uv[:,:,0])
    rtp[:,:,2] = np.arccos(uv[:,:,2]/rtp[:,:,0])
    return rtp

def assert_seltxt(mol, txt):
    ind = mol.topology.select(txt)
    if len(ind) == 0:
        print "= = = ERROR: selection text failed to find atoms! ", seltxt
        print "     ....debug: N(%s) = %i " % (txt, ind)
        return []
    else:
        return ind

# There may be minor mistakes in the selection text. Try to identify what is wrong.
def confirm_seltxt(ref, Hseltxt, Xseltxt):
    bError=False
    indH = assert_seltxt(ref, Hseltxt)
    indX = assert_seltxt(ref, Xseltxt)
    if len(indH) == 0:
        bError=True
        t1 = mol.topology.select('name H')
        t2 = mol.topology.select('name HN')
        t3 = mol.topology.select('name HA')
        print "    .... Note: The 'name H' selects %i atoms, 'name HN' selects %i atoms, and 'name HA' selects %i atoms." % (t1, t2, t3)
    if len(indX) == 0:
        bError=True
        t1 = mol.topology.select('name N')
        t2 = mol.topology.select('name NT')
        print "    .... Note: The 'name N' selects %i atoms, and 'name NT' selects %i atoms." % (t1, t2)
    if bError:
        sys.exit(1)
    return indH, indX

# Note: Return just the res number for easier plotting...
def obtain_XHres(traj, seltxt):
    indexH = traj.topology.select(seltxt)
    if len(indexH) == 0:
        print "= = = ERROR: selection text failed to find atoms! ", seltxt
        print "     ....debug: N(%s) = %i " % (Hseltxt, numH)
        sys.exit(1)
    #    indexH = traj.topology.select("name H and resSeq 3")
    resXH = [ traj.topology.atom(indexH[i]).residue.resSeq for i in range(len(indexH)) ]
    return resXH

def obtain_XHvecs(traj, Hseltxt, Xseltxt):
    print "= = = Obtaining XH-vectors from trajectory..."
    #nFrames= traj.n_frames
    indexX = traj.topology.select(Xseltxt)
    indexH = traj.topology.select(Hseltxt)
    numX = len(indexX) ; numH = len(indexH)
    if numX == 0 or numH == 0 :
        print "= = = ERROR: selection text failed to find atoms!"
        print "     ....debug: N(%s) = %i , N(%s) = %i" % (Xseltxt, numX, Hseltxt, numH)
        sys.exit(1)
    if len(indexH) != len(indexX):
        print "= = = ERROR: selection text found different number of atoms!"
        print "     ....debug: N(%s) = %i , N(%s) = %i" % (Xseltxt, numX, Hseltxt, numH)
        sys.exit(1)
    #Do dangerous trick to select nitrogens connexted to HN..
    #indexX = [ indexH[i]-1 for i in range(len(indexH))]

    # Extract submatrix of vector trajectory
    vecXH = np.take(traj.xyz, indexH, axis=1) - np.take(traj.xyz, indexX, axis=1)
    vecXH = normalise_vecs(vecXH, 2)

    return  vecXH

# 3 Sum_i,j <e_i * e_j >^2 - 1
def S2_by_outer(v):
    outer = np.mean([ np.outer(v[i],v[i]) for i in range(len(v))], axis=0)
    return 1.5*np.sum(outer**2.0)-0.5

def calculate_S2_by_outer(vecs, delta_t=-1, tau_memory=-1):
    """
    Calculates the general order parameter S2 by using the quantity:
    3*Sum_i,j <e_i * e_j >^2 - 1
    Returns a single vector S2 with dimension num_vecs when no memory time data is specified.
    Elsewise, return array (N_vec, 2) with average S2 and SEM.
    """
    num_frames = vecs.shape[0]
    num_bonds  = vecs.shape[1]
    if delta_t < 0 or tau_memory < 0:
        #Use no block-averaging
        S2=[]
        for i in range(num_bonds):
            S2.append( S2_by_outer(np.take(vecs,i, axis=1)) )
        return np.array(S2)
    else:
        #Use input time to determine block averaging.
        num_frames_block = int( tau_memory / delta_t )
        S2=[]
        for i in range(nBonds):
            avg=0.0
            av2=0.0
            c=0.0
            for j in range(0, num_frames, num_frames_block):
                if num_frames_block > num_frames - j:
                    break
                S2now = S2_by_outer( np.take(vecs, i, axis=1)[j:j+num_frames_block] )
                avg += S2now
                av2 += S2now*S2now
                c+=1.0
            avg/=c
            SEM=np.sqrt( (av2/c-avg**2.0)/c )
            S2.append( (avg, SEM) )
        return np.array(S2)

# iRED and wiRED implemented according to reading of
# Gu, Li, and Bruschweiler, JCTC, 2014
# http://dx.doi.org/10.1021/ct500181v
#
# 1. Implementation depends on an prior estimate of isotropic tumbling time, tau,
# that is used to determine averaging window size, and statistics.
# 2. Then we calculate the diagonalised matrix, spanning n by n of the cosine angle between each unit vector.
# 3. S2 is then the component that is covered by the internal motions,
#    excluding the first 5 motions governing global reorientation.
def calculate_S2_by_wiRED(vecs, dt, tau):
    print tau, dt
    # Todo.
    # Construct M for each chunk
    # Average over all frames
    # Diagonalise
    # Report on eigenvalues and eigenvectors

    #1. Determine chunks
    nFrames=len(vecs)
    nChunks=floor(nFrames*dt/(2*tau))
    nFrChunk=floor(2*tau/dt)

def calculate_S2_by_iRED(vecs, dt, tau):
    print tau, dt
    # Todo.
    # Construct M for each chunk
    # Average over all frames
    # Diagonalise
    # Report on eigenvalues and eigenvectors

    #1. Determine chunks
    nFrames=len(vecs)
    nChunks=floor(nFrames*dt/(5*tau))
    nFrChunk=floor(5*tau/dt)

def Ct_by_Palmer_old(vecs, dt, tau):
    """
    Vector auto-correlation function using Palmer's method, where only chunks of tau_memory
    are considered in each trajectory. Thus, maximum of C(t) will be tau/2.0
    """
    nFrames=len(vecs)
    nChunks=int(nFrames*dt/tau)
    nFrChunk=int(tau/dt)
    #np.correlate(a,a,mode='full')
    tot=np.zeros((nChunks,nFrChunk/2))
    #Obtain sample for each chunk spanning tau.
    for j in range(0,nChunks):
        off=j*nFrChunk
        # Average across frames of trajectory in a given chunk.
        Ct_sample=[ np.mean([ P2(np.dot(vecs[i+off],vecs[i+off+dFr])) for i in range(nFrChunk-dFr)]) for dFr in range(1,nFrChunk/2+1) ]
        tot[j]=Ct_sample
    #Average across samples.
    ret = tot.mean(axis=0)
    return ret

def Ct_Palmer_inner(vecs):
    """
    Treats 3D+ vectors of form ([chunks], frames, XYZ), calculating the autocorrelation C(t)
    only across each set of frames, then averaging across all chunks in the upper dimensions.
    Returns 1D array Ct of length frames/2 .
    """
    sh = vecs.shape
    nFr  = sh[-2]
    nPts = int(sh[-2]/2.0)
    Ct  = np.zeros( nPts )
    dCt = np.zeros( nPts )
    # Compute each data point across all samples, and average across all remaining dimensions.
    for dt in range(1,nPts+1):
        P2 = -0.5 + 1.5*np.square( np.einsum('...i,...i', vecs[...,0:nFr-dt,:], vecs[...,dt:nFr,:]) )
        Ct[dt-1]  = np.mean( P2 )
        dCt[dt-1] = np.std( P2 ) / np.sqrt( P2.size )
    return Ct, dCt

def calculate_Ct_Palmer(vecs):
    """
    This half of the proc handles the spliting of the vector array into chunks manageable by the subrouting.
    It will assumes that vecs coms in the form ( ..., bonds, XYZ).
    The function Ct_Palmer_inner handles all upper dimensions.
    """
    Ct=[] ; dCt = [] ;
    for i in range(nBonds):
        Ct_loc, dCt_loc = Ct_Palmer_inner( np.take(vecs, i, axis=-2) )
        Ct.append( Ct_loc ) ; dCt.append( dCt_loc )
        print "= = Bond %i Ct computed. Ct(%g) = %g , Ct(%g) = %g " % (i, dt[0], Ct_loc[0], dt[-1], Ct_loc[-1])

    return Ct, dCt

def calculate_dt(dt, tau):
    nPts = int(0.5*tau/dt)
    out = ( np.arange( nPts ) + 1.0) * dt
    return out

def reformat_vecs_by_tau(vecs, dt, tau):
    """
    This proc assumes that vecs list is N 3D-arrays in the form <Nfile>,(frames, bonds, XYZ).
    We take advantage of Palmer's iteration where the trajectory is divided into N chunks each of tau in length,
    to reformulate everything into fast 4D nparrays of form (nchunk, frames, bonds, XYZ) so as to
    take full advantage of broadcasting.
    where frames will be automatically divisible by the chunk length.
    """
    # Don't assume all files have the same number of frames.
    nFiles = len(vecs)
    nFramesPerChunk=int(tau/dt)
    print "    ...debug: Using %i frames per chunk based on tau/dt (%g/%g)." % (nFramesPerChunk, tau, dt)
    used_frames     = np.zeros(nFiles)
    remainders = np.zeros(nFiles)
    for i in range(nFiles):
        nFrames = vecs[i].shape[0]
        used_frames[i] = int(nFrames/nFramesPerChunk)*nFramesPerChunk
        remainders[i] = nFrames % nFramesPerChunk
        print "    ...Source %i divided into %i chunks. Usage rate: %g %%" % (i, used_frames[i]/nFramesPerChunk, 100.0*used_frames[i]/nFrames )

    nFramesTot = int( used_frames.sum() )
    out = np.zeros( ( nFramesTot, vecs[0].shape[1], vecs[0].shape[2] ) )
    start = 0
    for i in range(nFiles):
        end=int(start+used_frames[i])
        endv=int(used_frames[i])
        out[start:end,...] = vecs[i][0:endv,...]
        start=end
    sh = out.shape
    print "    ...Done. vecs reformatted into %i chunks." % ( nFramesTot/nFramesPerChunk )
    return out.reshape ( (nFramesTot/nFramesPerChunk, nFramesPerChunk, sh[-2], sh[-1]) )

def LS_one(x, S2, tau_c):
    if tau_c > 0.0:
        return (1-S2)*np.exp(-x/tau_c)+S2
    else:
        return S2

def fit_Ct(x, ylist):
    guess=(0.5,x[-1]/2.0)
    nFits=len(ylist)
    params=np.zeros((nFits,2))
    errors=np.zeros((nFits,2))
    for i in range(len(ylist)):
        popt, pcov = curve_fit(LS_one, x, ylist[i], p0=guess, method='dogbox', jac='3-point')
        perr = np.sqrt(np.diag(pcov))
        params[i]=popt
        errors[i]=perr
    return params.T, errors.T

def print_sy(fname, slist,ylist):
    fp = open(fname, 'w')
    for i in range(len(slist)):
        print >> fp, "%10s " % slist[i],
        print >> fp, "%10f" % float(ylist[i])
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

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Obtain the unit X-H vectors from one of more trajectories,'
                                     'and conduct calculations on it, such as S^2, C(t), and others analyses.'
                                     'N.B. Since we\'re following Palmer\'s formalisams by dividing data into chunks '
                                     'of length tau, ',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-s', type=str, dest='topfn', required=True, nargs='+',
                        help='Suitable topology PDB file for use in the MDTraj module.'
                             'This file is used as the reference frame for fits. If multiple are given, one refpdb will be loaded for each trajectory.')
    parser.add_argument('-f', '--infn', type=str, dest='infn', nargs='+',
                        help='One or more trajectories. Data from multiple trajectories will be analysed separately'
                             'in C(t)-calculations, but otherwise aggregated.' )
    parser.add_argument('-o', '--outpref', type=str, dest='out_pref', default='out',
                        help='Output file prefix.')
    parser.add_argument('-t','--tau', type=float, dest='tau', default=5000.0,
                        help='An estimate of the global tumbling time that is used to ignore internal motions'
                             'on timescales larger than can be measured by NMR relaxation. Same time units as trajectory, usually ps.')
    parser.add_argument('--prefact', type=float, dest='zeta', default=(1.02/1.04)**6,
                        help='MD-specific prefactor that accounts for librations of the XH-vector not seen in classical MD.'
                             'See, e.g. Trbovic et al., Proteins, 2008 -who references- Case, J. Biomol. NMR (1999)')
    parser.add_argument('--S2', dest='bDoS2', action='store_true', default=False,
                         help='Calculate order parameters S2, currenting using the simplest form with no tau-dependence.')
    parser.add_argument('--Ct', dest='bDoCt', action='store_true', default=False,
                         help='Calculate autocorrelation Ct, following Palmer\'s advice that'
                              'long MD data be split into short chunks to remove timescale not detectable by NMR.')
    parser.add_argument('--vecDist', dest='bDoVecDistrib', action='store_true', default=False,
                         help='Print the vectors distribution in spherical coordinates. In the frame of the PDB unless q_rot is given.')
    parser.add_argument('--vecHist', dest='bDoVecHist', action='store_true', default=False,
                         help='Print the 2D-histogram rather than just the collection of vecs.')
    parser.add_argument('--vecAvg', dest='bDoVecAverage', action='store_true', default=False,
                         help='Print the average unit XH-vector.')
    parser.add_argument('--vecRot', dest='vecRotQ', type=str, default='',
                         help='Rotation quaternion to be applied to the vector to transform it into PAF frame.')
    parser.add_argument('--Hsel', '--selection', type=str, dest='Hseltxt', default='name H',
                         help='Selection of the H-atoms to which the N-atoms are attached. E.g. "name H and resSeq 2 to 50 and chain A"')
    parser.add_argument('--Xsel', type=str, dest='Xseltxt', default='name N and not resname PRO',
                         help='Selection of the X-atoms to which the H-atoms are attached. E.g. "name N and resSeq 2 to 50 and chain A"')
    parser.add_argument('--fitsel', type=str, dest='fittxt', default='name CA',
                         help='Selection in which atoms will be fitted.')
    parser.add_argument('--help_sel', action='store_true', help='Display help for selection texts and exit.')

    args = parser.parse_args()
    time_start=time.clock()

    if args.help_sel:
        print_selection_help()
        sys.exit(0)

    # Read Parameters here
    tau_memory=args.tau
    bDoS2=args.bDoS2
    bDoCt=args.bDoCt
    bDoVecDistrib=args.bDoVecDistrib
    bDoVecHist=args.bDoVecHist
    if bDoVecHist and not bDoVecDistrib:
        bDoVecDistrib=True
    bDoVecAverage=args.bDoVecAverage
    if args.vecRotQ!='':
        bRotVec=True
        q_rot = [ float(v) for v in args.vecRotQ.split() ]
        if len(q_rot) != 4 or not q_ops.qisunit(q_rot):
            print "= = = ERROR: input rotation quaternion is malformed!", q_rot
            sys.exit(23)
    else:
        bRotVec=False
    zeta=args.zeta

    in_flist=args.infn
    in_reflist=args.topfn
    #top_fname=args.topfn
    out_pref=args.out_pref

    Hseltxt=args.Hseltxt
    Xseltxt=args.Xseltxt
    #seltxt='name H and resSeq 3 to 4'
    fittxt=args.fittxt

    #Determine the input format and construct 3D arrays of dimension (n_file, n_frame, XYZ)
    n_refs = len(in_reflist)
    n_trjs = len(in_flist)
    if n_refs == 1:
        bMultiRef=False
        top_filename=in_reflist[0]
        ref = md.load(top_filename)
        print "= = = Loaded single reference file: %s" % (top_filename)
    else:
        print "= = = Detected multiple reference file inputs."
        bMultiRef=True
        if n_refs != n_trjs:
            print >> sys.stderr, "= = ERROR: When giving multiple reference files, you must have one for each trajecfile file given!"
            sys.exit(1)

    # = = Load all trajectory data. Notes: Each file's resXH is 1D, vecXH is 3D in (frame, bond, XYZ)
    resXH = [] ; vecXH = [] ; vecXHfit = []
    deltaT  = np.nan ; nFrames = np.nan ; nBonds = np.nan
    bFirst=True
    for i in range(n_trjs):
        if bMultiRef:
            top_filename=in_reflist[i]
            ref = md.load(top_filename)
            print "= = = Loaded reference file %i: %s" % (i, top_filename)
        trj = md.load(in_flist[i], top=top_filename)
        print "= = = Loaded trajectory file %s - it has %i atoms and %i frames." % (in_flist[i], trj.n_atoms, trj.n_frames)
        # = = Run sanity check
        tmpH, tmpX = confirm_seltxt(trj, Hseltxt, Xseltxt)
        deltaT_loc = trj.timestep ; nFrames_loc = trj.n_frames
        resXH_loc  = obtain_XHres(trj, Hseltxt)
        vecXH_loc  = obtain_XHvecs(trj, Hseltxt, Xseltxt)
        trj.center_coordinates()
        trj.superpose(ref, frame=0, atom_indices=trj.topology.select(fittxt)  )
        print "= = = Molecule centered and fitted."
        #msds = md.rmsd(trj, ref, 0, precentered=True)
        vecXHfit_loc = obtain_XHvecs(trj, Hseltxt, Xseltxt)
        nBonds_loc  = vecXH_loc.shape[1]

        # = = Update overall variables
        if bFirst:
            resXH = resXH_loc
            deltaT = deltaT_loc ; nFrames = nFrames_loc ; nBonds = nBonds_loc
        else:
            if deltaT != deltaT_loc or nBonds != nBonds_loc or not np.equal(resXH, resXH_loc):
                print >> sys.stderr, "= = = ERROR: Differences in trajectories have been detected! Aborting."
                print >> sys.stderr, "      ...delta-t: %g vs.%g " % (deltaT, deltaT_loc)
                print >> sys.stderr, "      ...n-bonds: %g vs.%g " % (nBonds, nBonds_loc)
                print >> sys.stderr, "      ...Residue-XORs: %s " % ( set(resXH)^set(resXH_loc) )

        vecXH.append(vecXH_loc) ; vecXHfit.append(vecXHfit_loc)
        print "     ... XH-vector data added to memory."
#        print "= = = Loaded trajectory %s - Found %i XH-vectors %i frames." %  ( in_flist[i], nBonds_loc, vecXH_loc.shape[0] )

    vecXH_loc = [] ; vecXHfit_loc = []
    # = = =
    print "= = Loading finished."
    print "= = Reformatting all vecXH information into chunks of tau ( %g ) " % tau_memory

    vecXH = reformat_vecs_by_tau(vecXH, deltaT, tau_memory)
    vecXHfit = reformat_vecs_by_tau(vecXHfit, deltaT, tau_memory)

    # First analysis to be done is the C(t)-analysis, since that cannot be done after compressing the 4D to 3D.
    if bDoCt:
        print "= = = Conducting Ct_external using Palmer's approach."
        print "= = = timestep: ", deltaT, "ps"
        print "= = = tau_memory: ", tau_memory, "ps"
        if n_trjs > 1:
            print "= = = N.B.: For multiple files, 2D averaging is conducted at each datapoint."

        dt = calculate_dt(deltaT, tau_memory)
        Ct, dCt = calculate_Ct_Palmer(vecXH)
        gs.print_sxylist(out_pref+'_Ctext.dat', resXH, dt, np.stack( (Ct,dCt), axis=-1) )
        print "= = = Conducting Ct_internal using Palmer's approach."
        Ct, dCt = calculate_Ct_Palmer(vecXHfit)
        gs.print_sxylist(out_pref+'_Ctint.dat', resXH, dt, np.stack( (Ct,dCt), axis=-1) )

    # Compress 4D down to 3D for the rest of the calculations to simplify matters.
    sh = vecXH.shape
    vecXH    = vecXH.reshape( ( sh[0]*sh[1], sh[-2], sh[-1]) )
    vecXHfit = vecXHfit.reshape( ( sh[0]*sh[1], sh[-2], sh[-1]) )
    # = = = All sections below assume simple 3D arrays = = =

    if bRotVec:
        print "= = = Rotating all fitted vectors by the input quaternion into PAF."
        vecXHfit = qs.rotate_vector_simd(vecXHfit, q_rot)

    if bDoVecAverage:
        # Note: gs-script normalises along the last axis, after this mean operation
        vecXHfitavg = (gs.normalise_vector_array( np.mean(vecXHfit, axis=0) ))
        gs.print_xylist(out_pref+'_avgvec.dat', resXH, np.array(vecXHfitavg).T, True)

    if bDoVecDistrib:
        print "= = = Converting vectors into spherical coordinates."
        rtp = get_spherical_coords(vecXHfit)
        rtp = np.transpose(rtp,axes=(1,0,2)) ;# Convert from time first, to resid first.
        print rtp.shape
        if not bDoVecHist:
            gs.print_s3d(out_pref+'_PhiTheta.dat', resXH, rtp, (1,2))
        else:
            rtp = np.delete( rtp, 0, axis=2)
            # Should only bin on equal-area projections to conserve relative bin heights.
            print "= = = Histgrams will use Lambert Cylindrical projection by converting Theta spanning (0,pi) to cos(Theta) spanning (-1,1)"
            rtp[...,1]=np.cos(rtp[...,1])
            hist_list=[]
            for i in range(rtp.shape[0]):
                #hist, edges = np.histogramdd(rtp[i],bins=(36,36),range=((-np.pi,np.pi),(0,np.pi)),normed=True)
                hist, edges = np.histogramdd(rtp[i],bins=(72,36),range=((-np.pi,np.pi),(-1,1)),normed=True)
                ofile=out_pref+'_vecXH_'+str(resXH[i])+'.hist'
                gs.print_gplot_hist(ofile, hist, edges, header='# Lamber Cylindrical Histogram over phi,cos(theta).', bSphere=True)
                #gs.print_R_hist(ofile, hist, edges, header='# Lamber Cylindrical Histogram over phi,cos(theta).')
                print "= = = Written to output: ", ofile
    if bDoS2:
        print "= = = Conducting S2 analysis using memory time", tau_memory, "ps"
        S2 = calculate_S2_by_outer(vecXHfit, deltaT, tau_memory)
        #print_sy(out_pref+'_S2.dat', resXH, S2*zeta )
        gs.print_xylist(out_pref+'_S2.dat', resXH, (S2.T)*zeta, True )
        print "      ...complete."

    time_stop=time.clock()
    #Report time
    print "= = Finished. Total seconds elapsed: %g" % (time_stop - time_start)

sys.exit()
