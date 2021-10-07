from math import *
import sys, argparse, time
import numpy as np
import mdtraj as md

try:
    import psutil
    bPsutil = True
except:
    print "= = = NOTE: Module psutil isnot available, cannot respect memory usage on this run.."
    bPsutil = False

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

def obtain_XHvecs(traj, Hseltxt, Xseltxt, bSuppressPrint = False):
    if not bSuppressPrint:
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
    vecXH = vecnorm_NDarray(vecXH, axis=2)

    return  vecXH

def vecnorm_NDarray(v, axis=-1):
    """
    Vector normalisation performed along an arbitrary dimension, which by default is the last one.
    Comes with workaround by casting that to zero instead of keeping np.nan or np.inf.
    """
    # = = = need to protect against 0/0 errors when the vector is (0,0,0)
    if len(v.shape)>1:
        # = = = Obey broadcasting rules by applying 1 to the axis that is being reduced.
        sh=list(v.shape)
        sh[axis]=1
        return np.nan_to_num( v / np.linalg.norm(v,axis=axis).reshape(sh) )
    else:
        return np.nan_to_num( v/np.linalg.norm(v) )

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

def print_xylist(fn, x, ylist, bCols=False, header=""):
    """
    Array formats: x(nvals) y(nplots,nvals)
    bCols will stack all y contents in the same line, useful for errors.
    """
    fp = open( fn, 'w')
    if header != "":
        print >> fp, header
    ylist=np.array(ylist)
    shape=ylist.shape
    print shape
    if len(shape)==1:
        for j in range(len(x)):
            print >> fp, x[j], ylist[j]
        print >> fp, "&"
    elif len(shape)==2:
        nplot=shape[0]
        nvals=shape[1]
        if bCols:
            for j in range(nvals):
                print >> fp, x[j],
                for i in range(nplot):
                    print >> fp, ylist[i][j],
                print >> fp, ""
            print >> fp, "&"
        else:
            for i in range(nplot):
                for j in range(len(x)):
                    print >> fp, x[j], ylist[i][j]
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
    parser.add_argument('-f', '--infn', type=str, dest='infn', required=True, nargs='+',
                        help='One or more trajectories. Data from multiple trajectories will be analysed separately'
                             'in C(t)-calculations, but otherwise aggregated.' )
    parser.add_argument('-o', '--outpref', type=str, dest='out_pref', default='out',
                        help='Output file prefix.')
    parser.add_argument('-t','--tau', type=float, dest='tau', required=False, default=None,
                        help='Use the isotropic global tumbling time to split trjeactory into samples.'
                             'This excludes internal motions that are slower than measured by NMR relaxation.'
                             'Same time units as trajectory, usually ps.')
    parser.add_argument('--split', type=int, dest='nSplitFrames', default=-1,
                        help='Optionally split the reading of large trajectories into N frames to reduce memory footprint.')
    parser.add_argument('--zeta', action='store_true', dest='bZeta', default=False,
                        help='Apply a prefactor that accounts for the librations of the XH-vector not seen in classical MD.'
                             'See, e.g. Trbovic et al., Proteins, 2008 -who references- Case, J. Biomol. NMR (1999)')
    parser.add_argument('--vecAvg', dest='bDoVecAverage', action='store_true', default=False,
                         help='Print the average unit XH-vector.')
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
    bDoVecAverage=args.bDoVecAverage
    if args.bZeta:
        zeta=(1.02/1.04)**6
    else:
        zeta=1.0

    if args.nSplitFrames > 0:
        bSplitRead=True
        nSplitFrames=args.nSplitFrames
    else:
        bSplitRead=False

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
    resXH = [] ; vecXHfit = []
    deltaT  = np.nan ; nFrames = np.nan ; nBonds = np.nan
    bFirst=True
    for i in range(n_trjs):
        if bMultiRef:
            top_filename=in_reflist[i]
            ref = md.load(top_filename)
            print "= = = Loaded reference file %i: %s" % (i, top_filename)
            fit_indices = get_indices_mdtraj( top=ref.topology, filename=top_filename, seltxt=fittxt)
            print "= = = Debug: fit_indices number: %i" % len(fit_indices)

        if bSplitRead:
            # = = = To tackle trajectory files that take too much memory, split into N frames
            print "= = = Loading trajectory file %s in chunks..." % (in_flist[i])
            nFrames_loc = 0
            for trjChunk in md.iterload(in_flist[i], chunk=nSplitFrames, top=top_filename):
                trjChunk.center_coordinates()
                trjChunk.superpose(ref, frame=0, atom_indices=fit_indices )
                tempV2 = obtain_XHvecs(trjChunk, Hseltxt, Xseltxt, bSuppressPrint=True)
                if nFrames_loc == 0:
                    confirm_seltxt(trjChunk, Hseltxt, Xseltxt)
                    resXH_loc  = obtain_XHres(trjChunk, Hseltxt)
                    deltaT_loc = trjChunk.timestep
                    nFrames_loc = trjChunk.n_frames
                    vecXHfit_loc = tempV2
                else:
                    nFrames_loc += trjChunk.n_frames
                    vecXHfit_loc = np.concatenate( (vecXHfit_loc, tempV2), axis=0 )

                print "= = = ...loaded %i frames so far." % (nFrames_loc)

            print "= = = Finished loading trajectory file %s. It has %i atoms and %i frames." % (in_flist[i], trjChunk.n_atoms, nFrames_loc)
            print vecXHfit_loc.shape
            nBonds_loc  = vecXHfit_loc.shape[1]
	    	    
	    del trjChunk

        else:
            # = = = Faster single-step read
            print "= = = Reading trajectory file %s ..." % (in_flist[i])
            trj = md.load(in_flist[i], top=top_filename)
            print "= = = File loaded - it has %i atoms and %i frames." % (trj.n_atoms, trj.n_frames)
            # = = Run sanity check
            confirm_seltxt(trj, Hseltxt, Xseltxt)
            deltaT_loc = trj.timestep ; nFrames_loc = trj.n_frames
            resXH_loc  = obtain_XHres(trj, Hseltxt)

            trj.center_coordinates()
            trj.superpose(ref, frame=0, atom_indices=fit_indices )
            print "= = = Molecule centered and fitted."
            #msds = md.rmsd(trj, ref, 0, precentered=True)
            vecXHfit_loc = obtain_XHvecs(trj, Hseltxt, Xseltxt)
            nBonds_loc  = vecXHfit_loc.shape[1]

            del trj

        if tau_memory is not None and deltaT > 0.5*tau_memory:
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

        vecXHfit.append(vecXHfit_loc)
        print "     ... XH-vector data added to memory, using 2x %.2f MB" % (  sys.getsizeof(vecXHfit_loc)/1024.0**2.0 )

#        print "= = = Loaded trajectory %s - Found %i XH-vectors %i frames." %  ( in_flist[i], nBonds_loc, vecXH_loc.shape[0] )

    del vecXHfit_loc
    # = = =
    print "= = Loading finished."
    vecXHfit=np.array(vecXHfit)

    if bPsutil:
        # = = = Check vector size and currently free memory. Units are in bytes.
        nFreeMem = 1.0*psutil.virtual_memory()[3]/1024.0**2
        nVecMem = 1.0*sys.getsizeof(vecXHfit)/1024.0**2
        if nFreeMem < 2*nVecMem:
            print " = = = WARNING: the size of vectors is getting close to the amount of free system memory!"
            print "    ... %.2f MB used by one vector vecXHfit." % nVecMem
            print "    ... %.2f MB free system memory." % nFreeMem
        else:
            print " = = = Memoryfootprint debug. vecXHfit uses %.2f MB versus %.2f MB free memory" % ( nVecMem, nFreeMem )
    else:
        print "= = = psutil module has not been loaded. Cannot check for memory footprinting."

    if tau_memory != None:
        print "= = Reformatting all vecXHfit information into chunks of tau ( %g ) " % tau_memory
        vecXHfit = reformat_vecs_by_tau(vecXHfit, deltaT, tau_memory)

    # Compress 4D down to 3D for the rest of the calculations to simplify matters.
    sh = vecXHfit.shape
    vecXHfit = vecXHfit.reshape( ( sh[0]*sh[1], sh[-2], sh[-1]) )
    # = = = All sections below assume simple 3D arrays = = =

    if bDoVecAverage:
        # Note: gs-script normalises along the last axis, after this mean operation
        vecXHfitavg = (gs.normalise_vector_array( np.mean(vecXHfit, axis=0) ))
        print_xylist(out_pref+'_avgvec.dat', resXH, np.array(vecXHfitavg).T, True)
        del vecXHfitavg

    if tau_memory != None:
        print "= = = Conducting S2 analysis using memory time to chop input-trajectories", tau_memory, "ps"
        S2 = calculate_S2_by_outerProduct(vecXHfit, deltaT, tau_memory)
    else:
        print "= = = Conducting S2 analysis directly from trajectories."
        S2 = calculate_S2_by_outerProduct(vecXHfit)

    print_xylist(out_pref+'_S2.dat', resXH, (S2.T)*zeta, True )
    print "      ...complete."

    time_stop=time.clock()
    #Report time
    print "= = Finished. Total seconds elapsed: %g" % (time_stop - time_start)

sys.exit()
