from math import *
import sys, os, argparse, time, gc
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
    vecXH = qs.vecnorm_NDarray(vecXH, axis=2)

    return  vecXH

# 3 Sum_i,j <e_i * e_j >^2 - 1
def S2_by_outerProduct(v):
    """
    Two
    """
    outer = np.mean([ np.outer(v[i],v[i]) for i in range(len(v))], axis=0)
    return 1.5*np.sum(outer**2.0)-0.5

def calculate_S2_by_outerProduct(vecs, delta_t=-1, tau_memory=-1):
    """
    Calculates the general order parameter S2 by using the quantity 3*Sum_i,j <e_i * e_j >^2 - 1 , which is akin to P2( CosTheta )
    Expects vecs to be of dimensions (time, 3) or ( time, nResidues, 3 )

    This directly collapses all dimensions in two steps:
    - 1. calculate the outer product <v_i v_j >
    - 2. calculate Sum <v_i v_j>^2

    When both delta_t and tau_memory are given, then returns average and SEM of the S2 samples of dimensions ( nResidues, 2 )
    """
    sh=vecs.shape
    nDim=sh[-1]
    if len(sh)==2:
        nFrames=vecs.shape[0]
        if delta_t < 0 or tau_memory < 0:
            #Use no block-averaging
            tmp = np.einsum( 'ij,ik->jk', vecs,vecs) / nFrames
            return 1.5*np.einsum('ij,ij->',tmp,tmp)-0.5
        else:
            nFramesPerBlock = int( tau_memory / delta_t )
            nBlocks = int( nFrames / nFramesPerBlock )
            # Reshape while dumping extra frames
            vecs = vecs[:nBlocks*nFramesPerBlock].reshape( nBlocks, nFramesPerBlock, nDim )
            tmp = np.einsum( 'ijk,ijl->ikl', vecs,vecs) / nFramesPerBlock
            tmp = 1.5*np.einsum('ijk,ijk->i',tmp,tmp)-0.5
            S2 = np.mean( tmp )
            dS2 = np.std( tmp ) / ( np.sqrt(nBlocks) - 1.0 )
            return np.array( [S2,dS2])
    elif len(sh)==3:
        # = = = Expect dimensions (time, nResidues, 3)
        nFrames = vecs.shape[0]
        nResidues  = vecs.shape[1]
        if delta_t < 0 or tau_memory < 0:
            #Use no block-averaging
            tmp = np.einsum( 'ijk,ijl->jkl', vecs,vecs) / nFrames
            return 1.5*np.einsum('...ij,...ij->...',tmp,tmp)-0.5
        else:
            nFramesPerBlock = int( tau_memory / delta_t )
            nBlocks = int( nFrames / nFramesPerBlock )
            # Reshape while dumping extra frames
            vecs = vecs[:nBlocks*nFramesPerBlock].reshape( nBlocks, nFramesPerBlock, nResidues, nDim )
            tmp = np.einsum( 'ijkl,ijkm->iklm', vecs,vecs) / nFramesPerBlock
            tmp = 1.5*np.einsum('...ij,...ij->...',tmp,tmp)-0.5
            S2  = np.mean( tmp, axis=0 )
            dS2 = np.std ( tmp, axis=0 ) / ( np.sqrt(nBlocks) - 1.0 )
            return np.stack( (S2,dS2), axis=-1 )
    else:
        print >> sys.stderr, "= = = ERROR in calculate_S2_by_outerProduct: unsupported number of dimensions! vecs.shape: ", sh
        sys.exit(1)

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

#def Ct_Palmer_inner(vecs):
#    """
#    Treats 3D+ vectors of form ([chunks], frames, XYZ), calculating the autocorrelation C(t)
#    only across each set of frames, then averaging across all chunks in the upper dimensions.
#    Returns 1D array Ct of length frames/2 .
#    """
#    sh = vecs.shape
#    nFr  = sh[-2]
#    nPts = int(sh[-2]/2.0)
#    Ct  = np.zeros( nPts, dtype=vecs.dtype)
#    dCt = np.zeros( nPts, dtype=vecs.dtype)
#    # Compute each data point across all samples, and average across all remaining dimensions.
#    for dt in range(1,nPts+1):
#        P2 = -0.5 + 1.5*np.square( np.einsum('...i,...i', vecs[...,0:nFr-dt,:], vecs[...,dt:nFr,:]) )
#        Ct[dt-1]  = np.mean( P2 )
#        dCt[dt-1] = np.std( P2 ) / np.sqrt( P2.size )
#    return Ct, dCt

def calculate_Ct_Palmer(vecs):
    """
    Definition: < P2( v(t).v(t+dt) )  >
    (Rewritten) This proc assumes vecs to be of square dimensions ( nReplicates, nFrames, nResidues, 3).
    Operates a single einsum per delta-t timepoints to produce the P2(v(t).v(t+dt)) with dimensions ( nReplicates, nResidues )
    then produces the statistics from there according to Palmer's theory that trajectory be divide into N-replcates with a fixed memory time.
    Output Ct and dCt should take dimensions ( nResidues, nDeltas )
    """
    sh = vecs.shape
    print "= = = Debug of calculate_Ct_Palmer confirming the dimensions of vecs:", sh
    if sh[1]<50:
        print >> sys.stderr,"= = = WARNING: there are less than 50 frames per block of memory-time!"

    if len(sh)!=4:
        # Not in the right form...
        print >> sys.stderr, "= = = ERROR: The input vectors to calculate_Ct_Palmer is not of the expected 4-dimensional form! " % sh
        sys.exit(1)

    nReplicates=sh[0] ; nDeltas=sh[1]/2 ; nResidues=sh[2]
    Ct  = np.zeros( (nDeltas,nResidues), dtype=vecs.dtype )
    dCt = np.zeros( (nDeltas,nResidues), dtype=vecs.dtype )
    bFirst=True
    for delta in range(1,1+nDeltas):
        nVals=sh[1]-delta
        # = = Create < vi.v'i > with dimensions (nRep, nFr, nRes, 3) -> (nRep, nFr, nRes) -> ( nRep, nRes ), then average across replicates with SEM.
        tmp = -0.5 + 1.5 * np.square( np.einsum( 'ijkl,ijkl->ijk', vecs[:,:-delta,...] ,vecs[:,delta:,...] ) )
        tmp  = np.einsum( 'ijk->ik', tmp ) / nVals
        Ct[delta-1]  = np.mean( tmp,axis=0 )
        dCt[delta-1] = np.std( tmp,axis=0 ) / ( np.sqrt(nReplicates) - 1.0 )
        #if bFirst:
        #    bFirst=False
        #    print tmp.shape, P2.shape
        #    print tmp[0,0,0], P2[0,0]
        #Ct[delta-1]  = np.mean( tmp,axis=(0,1) )
        #dCt[delta-1] = np.std( tmp,axis=(0,1) ) / ( np.sqrt(nReplicates*nVals) - 1.0 )

    #print "= = Bond %i Ct computed. Ct(%g) = %g , Ct(%g) = %g " % (i, dt[0], Ct_loc[0], dt[-1], Ct_loc[-1])
    # Return with dimensions ( nDeltas, nResidues ) by default.
    return Ct, dCt

def calculate_dt(dt, tau):
    nPts = int(0.5*tau/dt)
    out = ( np.arange( nPts ) + 1.0) * dt
    return out

def reformat_vecs_by_tau(vecs, dt, tau):
    """
    This proc assumes that vecs list is N 3D-arrays in the form <Nfile>,(frames, bonds, XYZ).
    We take advantage of Palmer's iteration where the trajectory is divided into N chunks each of tau in length,
    to reformulate everything into fast 4D np.arrays of form (nchunk, frames, bonds, XYZ) so as to
    take full advantage of broadcasting.
    This will throw away additional frame data in each trajectory that does not fit into a single block of memory time tau.
    """
    # Don't assume all files have the same number of frames.
    nFiles = len(vecs)
    nFramesPerChunk=int(tau/dt)
    print "    ...debug: Using %i frames per chunk based on tau/dt (%g/%g)." % (nFramesPerChunk, tau, dt)
    used_frames     = np.zeros(nFiles, dtype=int)
    remainders = np.zeros(nFiles, dtype=int)
    for i in range(nFiles):
        nFrames = vecs[i].shape[0]
        used_frames[i] = int(nFrames/nFramesPerChunk)*nFramesPerChunk
        remainders[i] = nFrames % nFramesPerChunk
        print "    ...Source %i divided into %i chunks. Usage rate: %g %%" % (i, used_frames[i]/nFramesPerChunk, 100.0*used_frames[i]/nFrames )

    nFramesTot = int( used_frames.sum() )
    out = np.zeros( ( nFramesTot, vecs[0].shape[1], vecs[0].shape[2] ) , dtype=vecs[0].dtype)
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

def get_indices_mdtraj( seltxt, top, filename):
    """
    NB: A workaround for MDTraj is needed becusae the standard reader
    does not return topologies.
    """
    if seltxt == 'custom occupancy':
        pdb  = md.formats.pdb.pdbstructure.PdbStructure(open(filename))
        mask = [ atom.get_occupancy() for atom in pdb.iter_atoms() ]
        inds = top.select('all')
        return [ inds[i] for i in range(len(mask)) if mask[i] > 0.0 ]
    else:
        return top.select(seltxt)

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
    parser.add_argument('-t','--tau', type=float, dest='tau', required=True, default=None,
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
    parser.add_argument('--binary', action='store_true', default=False,
                         help='Change the vector storage format to numpy binary to save a bit of space and read speed for large files.')
    parser.add_argument('--vecHist', dest='bDoVecHist', action='store_true', default=False,
                         help='Print the 2D-histogram rather than just the collection of vecs.')
    parser.add_argument('--histBin', type=int, default=72,
                         help='Resolution of the spherical histogram as the number of bins along phi (-pi,pi), by default covering 5-degrees.'
                              'The number of bins along theta (0,pi) is automatically half this number.')
    parser.add_argument('--vecAvg', dest='bDoVecAverage', action='store_true', default=False,
                         help='Print the average unit XH-vector.')
    parser.add_argument('--vecRot', dest='vecRotQ', type=str, default='',
                         help='Rotation quaternion to be applied to the vector to transform it into PAF frame.')
    parser.add_argument('--Hsel', '--selection', type=str, dest='Hseltxt', default='name H',
                         help='Selection of the H-atoms to which the N-atoms are attached. E.g. "name H and resSeq 2 to 50 and chain A"')
    parser.add_argument('--Xsel', type=str, dest='Xseltxt', default='name N and not resname PRO',
                         help='Selection of the X-atoms to which the H-atoms are attached. E.g. "name N and resSeq 2 to 50 and chain A"')
    parser.add_argument('--fitsel', type=str, dest='fittxt', default='custom occupancy',
            help='Selection in which atoms will be fitted. Examples include: \'name CA\' and '
                 '\'resSeq 3 to 30\'. The default selection invokes a workaround to read the occupancy data and take positive entries.')
    parser.add_argument('--help_sel', action='store_true', help='Display help for selection texts and exit.')

    args = parser.parse_args()
    time_start=time.clock()

    if args.help_sel:
        print_selection_help()
        sys.exit(0)

    # = = = Read Parameters here = = =
    tau_memory=args.tau
    bDoS2=args.bDoS2
    bDoCt=args.bDoCt
    if bDoCt and tau_memory == None:
        print >> sys.stderr, "= = = Refusing to do C(t)-analysis without using a block averaging over memory_time tau!"
        sys.exit(1)
    bDoVecDistrib=args.bDoVecDistrib
    bDoVecHist=args.bDoVecHist
    histBinX=args.histBin
    histBinY=histBinX/2
    bBinary=args.binary
    if bDoVecHist and not bDoVecDistrib:
        bDoVecDistrib=True
    bDoVecAverage=args.bDoVecAverage
    if args.vecRotQ!='':
        bRotVec=True
        q_rot = np.array( [ float(v) for v in args.vecRotQ.split() ] )
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
        # Load the atom indices over which the atom fit will take place.
        fit_indices = get_indices_mdtraj(top=ref.topology, filename=top_filename, seltxt=fittxt)
        print "= = = Debug: fit_indices number: %i" % len(fit_indices)
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
            fit_indices = get_indices_mdtraj( top=ref.topology, filename=top_filename, seltxt=fittxt)
            print "= = = Debug: fit_indices number: %i" % len(fit_indices)
        trj = md.load(in_flist[i], top=top_filename)
        print "= = = Loaded trajectory file %s - it has %i atoms and %i frames." % (in_flist[i], trj.n_atoms, trj.n_frames)
        # = = Run sanity check
        tmpH, tmpX = confirm_seltxt(trj, Hseltxt, Xseltxt)
        deltaT_loc = trj.timestep ; nFrames_loc = trj.n_frames
        resXH_loc  = obtain_XHres(trj, Hseltxt)
        vecXH_loc  = obtain_XHvecs(trj, Hseltxt, Xseltxt)
        trj.center_coordinates()
        print "= = DEBUG: Fitted indices are ", fit_indices
        trj.superpose(ref, frame=0, atom_indices=fit_indices )
        print "= = = Molecule centered and fitted."
        #msds = md.rmsd(trj, ref, 0, precentered=True)
        vecXHfit_loc = obtain_XHvecs(trj, Hseltxt, Xseltxt)
        nBonds_loc  = vecXH_loc.shape[1]

        del trj

        if deltaT > 0.5*tau_memory:
            print >> sys.stderr, "= = = ERROR: delta-t form the trajectory is too small relative to tau! %g vs. %g" % (deltaT, tau_memory)
            sys.exit(1)

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

    # print type(vecXH[0,0,0,0])
    # First analysis to be done is the C(t)-analysis, since that cannot be done after compressing the 4D to 3D.
    if bDoCt:
        print "= = = Conducting Ct_external using Palmer's approach."
        print "= = = timestep: ", deltaT, "ps"
        print "= = = tau_memory: ", tau_memory, "ps"
        if n_trjs > 1:
            print "= = = N.B.: For multiple files, 2D averaging is conducted at each datapoint."

        dt = calculate_dt(deltaT, tau_memory)
        Ct, dCt = calculate_Ct_Palmer(vecXH)
        gs.print_sxylist(out_pref+'_Ctext.dat', resXH, dt, np.stack( (Ct.T,dCt.T), axis=-1) )
        print "= = = Conducting Ct_internal using Palmer's approach."
        Ct, dCt = calculate_Ct_Palmer(vecXHfit)
        gs.print_sxylist(out_pref+'_Ctint.dat', resXH, dt, np.stack( (Ct.T,dCt.T), axis=-1) )
        del Ct, dCt

    # Compress 4D down to 3D for the rest of the calculations to simplify matters.
    sh = vecXHfit.shape
    #vecXH    = vecXH.reshape( ( sh[0]*sh[1], sh[-2], sh[-1]) )
    vecXHfit = vecXHfit.reshape( ( sh[0]*sh[1], sh[-2], sh[-1]) )
    # = = = All sections below assume simple 3D arrays = = =
    del vecXH

    if bRotVec:
        print "= = = Rotating all fitted vectors by the input quaternion into PAF."
        try:
            vecXHfit = qs.rotate_vector_simd(vecXHfit, q_rot)
        except MemoryError:
            print >> sys.stderr, "= = WARNING: Ran out of memory running vector rotations! Doing this the slower way."
            for i in range(sh[0]*sh[1]):
                vecXHfit[i] = qs.rotate_vector_simd(vecXHfit[i], q_rot)

    if bDoVecAverage:
        # Note: gs-script normalises along the last axis, after this mean operation
        vecXHfitavg = (gs.normalise_vector_array( np.mean(vecXHfit, axis=0) ))
        gs.print_xylist(out_pref+'_avgvec.dat', resXH, np.array(vecXHfitavg).T, True)
        del vecXHfitavg

    if bDoVecDistrib:
        print "= = = Converting vectors into spherical coordinates."
        try:
            rtp = gm.xyz_to_rtp(vecXHfit)
            # print type(rtp[0,0,0])
        except MemoryError:
            print >> sys.stderr, "= = WARNING: Ran out of memory running spherical conversion!"
            sys.exit(9)
            # = = = Don't bother rescuing.
            #for i in range(sh[0]*sh[1]):
            #    vecXHfit[i] = get_spherical_coords(vecXHfit[i])
            #vecXHfit = np.transpose(vecXHfit,axes=(1,0,2)) ;# Convert from time first, to resid first.
            #gs.print_s3d(out_pref+'_PhiTheta.dat', resXH, vecXHfit, (1,2))
            #sys.exit(9)

        rtp = np.transpose(rtp,axes=(1,0,2)) ;# Convert from time first, to resid first.
        print "= = = Debug: shape of the spherical vector distribution:", rtp.shape
        if not bDoVecHist:
            if bBinary:
                np.savez_compressed(out_pref+'_vecPhiTheta.npz', names=resXH, \
                        dataType='PhiTheta', axisLabels=['phi','theta'], bHistogram=False, data=rtp[...,1:3])
            else:
                gs.print_s3d(out_pref+'_vecPhiTheta.dat', resXH, rtp, (1,2))
        else:
            # = = = Conduct histograms on the dimension of phi, cos(theta). The Lambert Cylindrical projection preserves sample area, and therefore preserves the bin occupancy rates.
            # = = = ...this leads to relative bin occupancies that make more sense when plotted directly.
            rtp = np.delete( rtp, 0, axis=2)
            print "= = = Histgrams will use Lambert Cylindrical projection by converting Theta spanning (0,pi) to cos(Theta) spanning (-1,1)"
            rtp[...,1]=np.cos(rtp[...,1])

            hist_list=np.zeros((nBonds,histBinX,histBinY), dtype=rtp.dtype)
            bFirst=True
            for i in range(nBonds):
                hist, edges = np.histogramdd(rtp[i],bins=(histBinX,histBinY),range=((-np.pi,np.pi),(-1,1)),normed=False)
                if bFirst:
                    bFirst=False
                    edgesAll = edges
                #else:
                #    if np.any(edgesAll!=edges):
                #        print >> sys.stderr, "= = = ERROR: histogram borders are not equal. This should never happen!"
                #        sys.exit(1)
                hist_list[i]=hist

            if bBinary:
                np.savez_compressed(out_pref+'_vecHistogram.npz', names=resXH, \
                        dataType='LambertCylindrical', bHistogram=True, edges=edgesAll, axisLabels=['phi','cos(theta)'], data=hist_list)
            else:
                for i in range(nBonds):
                    ofile=out_pref+'_vecXH_'+str(resXH[i])+'.hist'
                    gs.print_gplot_hist(ofile, hist, edges, header='# Lamber Cylindrical Histogram over phi,cos(theta).', bSphere=True)
                    #gs.print_R_hist(ofile, hist, edges, header='# Lamber Cylindrical Histogram over phi,cos(theta).')
                    print "= = = Written to output: ", ofile

    if bDoS2:
        if tau_memory != None:
            print "= = = Conducting S2 analysis using memory time to chop input-trajectories", tau_memory, "ps"
            S2 = calculate_S2_by_outerProduct(vecXHfit, deltaT, tau_memory)
        else:
            print "= = = Conducting S2 analysis directly from trajectories."
            S2 = calculate_S2_by_outerProduct(vecXHfit)

        gs.print_xylist(out_pref+'_S2.dat', resXH, (S2.T)*zeta, True )
        print "      ...complete."

    time_stop=time.clock()
    #Report time
    print "= = Finished. Total seconds elapsed: %g" % (time_stop - time_start)

sys.exit()
