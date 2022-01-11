from math import *
import sys, os, argparse, time
from re import split as regexp_split
import numpy as np
import numpy.ma as ma
import general_scripts as gs
import general_maths as gm
import transforms3d.quaternions as qops
import transforms3d_supplement as qs
import fitting_Ct_functions as fitCt
import spectral_densities as sd
from scipy.optimize import fmin_powell

# There may be minor mistakes in the selection text. Try to identify what is wrong.
def confirm_seltxt(ref, Hseltxt, Xseltxt):
    bError=False
    indH = mol.topology.select(Hseltxt)
    indX = mol.topology.select(Xseltxt)
    numH = len(indH) ; numX = len(numX)
    if numH == 0:
        bError=True
        t1 = mol.topology.select('name H')
        t2 = mol.topology.select('name HN')
        t3 = mol.topology.select('name HA')
        print( "    .... ERROR: The 'name H' selects %i atoms, 'name HN' selects %i atoms, and 'name HA' selects %i atoms." % (t1, t2, t3) )
    if numX == 0:
        bError=True
        t1 = mol.topology.select('name N')
        t2 = mol.topology.select('name NT')
        print( "    .... ERROR: The 'name N' selects %i atoms, and 'name NT' selects %i atoms." % (t1, t2) )

    resH = [ mol.topology.atom(x).residue.resSeq for x in indH ]
    resX = [ mol.topology.atom(x).residue.resSeq for x in indX ]
    if resX != resH:
        bError=True
        print( "    .... ERROR: The residue lists are not the name between the two selections:" )
        print( "    .... Count for X (%i)" % numX, resX )
        print( "    .... Count for H (%i)" % numH, resH )
    if bError:
        sys.exit(1)

    return indH, indX, resX

def extract_vectors_from_structure( pdbFile, Hseltxt='name H', Xsel='name N and not resname PRO', trjFile=None ):

    print( "= = = Using vectors as found directly in the coordinate files, via MDTraj module." )
    print( "= = = NOTE: no fitting is conducted." )
    import mdtraj as md

    if not trjFile is None:
        mol = md.load(trjFile, top=pdbFile)
        print( "= = = PDB file %s and trajectory file %s loaded." % (pdbFile, trjFile) )

    else:
        omol = md.load(pdbFile)
        print( "= = = PDB file %s loaded." % pdbFile )

    indH, indX, resXH = confirm_seltxt(ref, Hsel, Xsel)

    # Extract submatrix of vector trajectory
    vecXH = np.take(mol.xyz, indH, axis=1) - np.take(mol.xyz, indX, axis=1)
    vecXH = qs.vecnorm_NDarray(vecXH, axis=2)

    # = = = Check shape and switch to match.
    if vecXH.shape[0] == 1:
        return resXH, vecXH[0]
    else:
        return resXH, np.swapaxes(vecXH,0,1)

def sanity_check_two_list(listA, listB, string, bVerbose=False):
    if not gm.list_identical( listA, listB ):
        print( "= = ERROR: Sanity checked failed for %s!" % string )
        if bVerbose:
            print( listA )
            print( listB )
        else:
            print( "    ...first residues:", listA[0], listB[0] )
            print( "    ...set intersection (unordered):", set(listA).intersection(set(listB)) )
        sys.exit(1)
    return

def _obtain_Jomega(RObj, nSites, S2, consts, taus, vecXH, weights=None):
    """
    The inputs vectors have dimensions (nSites, nSamples, 3) or just (nSites, 3)
    the datablock being returned has dimensions:
    - (nFrequencies, nSites)    of there is no uncertainty calculations. 5 being the five frequencies J(0), J(wN) J(wH+wN), J(wH), J(wH-wN)
    - (nFrequencies, nSites, 2) if there is uncertainty calculations.
    """
    nFrequencies= len(RObj.omega)
    if RObj.rotdifModel.name == 'rigid_sphere':
        datablock=np.zeros((5,nSites), dtype=np.float32)
        for i in range(nSites):
            J = sd.J_combine_isotropic_exp_decayN(RObj.omega, 1.0/(6.0*RObj.rotdifModel.D), S2[i], consts[i], taus[i])
            datablock[:,i]=J
        return datablock
    elif RObj.rotdifModel.name == 'rigid_symmtop':
        # Automatically use the vector-form of function.
        if len(vecXH.shape) > 2:
            # An ensemble of vectors at each site. Measure values for all of them then average with/without weights.
            datablock=np.zeros( (nFrequencies, nSites, 2), dtype=np.float32)
            npts=vecXH.shape[1]
            for i in range(nSites):
                # = = = Calculate at each residue and sum over Jmat( nSamples, 2)
                Jmat = sd.J_combine_symmtop_exp_decayN(RObj.omega, vecXH[i], RObj.rotdifModel.D[0], RObj.rotdifModel.D[1], S2[i], consts[i], taus[i])
                if weights is None:
                    datablock[:,i,0] = np.mean(Jmat, axis=0)
                    datablock[:,i,1] = np.std(Jmat, axis=0)
                else:
                    datablock[:,i,0] = np.average(Jmat, axis=0, weights=weights[i])
                    datablock[:,i,1] = np.sqrt( np.average( (Jmat - datablock[:,i,0])**2.0, axis=0, weights=weights[i]) )
            return datablock
        else:
            #Single XH vector at each site. Ignore weights, as they should not exist.
            datablock=np.zeros((5,nSites), dtype=np.float32)
            for i in range(nSites):
                Jmat = sd.J_combine_symmtop_exp_decayN(RObj.omega, vecXH[i], RObj.rotdifModel.D[0], RObj.rotdifModel.D[1], S2[i], consts[i], taus[i])
                datablock[:,i]=Jmat
            return datablock

    # = = Should only happen with fully anisotropic models.
    print( "= = ERROR: Unknown rotdifModel in the relaxation object used in calculations!", file=sys.stderr )
    return []


def _obtain_R1R2NOErho(RObj, nSites, S2, consts, taus, vecXH, weights=None, CSAvaluesArray=None):
    """
    The inputs vectors have dimensions (nSites, nSamples, 3) or just (nSites, 3)
    the datablock being returned has dimensions:
    - ( 4, nSites)    of there is no uncertainty calculations. 4 corresponding each to R1, R2, NOE, and rho.
    - ( 4, nSites, 2) if there is uncertainty calculations.
    """
    if CSAvaluesArray is None:
        CSAvaluesArray = np.repeat(CSAvaluesArray, nSites)
    if RObj.rotdifModel.name == 'direct_transform':
        datablock=np.zeros((4,nSites), dtype=np.float32)
        for i in range(nSites):
            J = sd.J_direct_transform(RObj.omega, consts[i], taus[i])
            R1, R2, NOE = RObj.get_relax_from_J( J, CSAvalue=CSAvaluesArray[i] )
            rho = RObj.get_rho_from_J( J )
            datablock[:,i]=[R1,R2,NOE,rho]
        return datablock
    elif RObj.rotdifModel.name == 'rigid_sphere':
        datablock=np.zeros((4,nSites), dtype=np.float32)
        for i in range(nSites):
            J = sd.J_combine_isotropic_exp_decayN(RObj.omega, 1.0/(6.0*RObj.rotdifModel.D), S2[i], consts[i], taus[i])
            R1, R2, NOE = RObj.get_relax_from_J( J, CSAvalue=CSAvaluesArray[i] )
            rho = RObj.get_rho_from_J( J )
            datablock[:,i]=[R1,R2,NOE,rho]
        return datablock
    elif RObj.rotdifModel.name == 'rigid_symmtop':
        # Automatically use the vector-form of function.
        if len(vecXH.shape) > 2:
            # An ensemble of vectors for each site.
            datablock=np.zeros((4,nSites,2), dtype=np.float32)
            npts=vecXH.shape[1]
            #tmpR1  = np.zeros(npts) ; tmpR2 = np.zeros(npts) ; tmpNOE = np.zeros(npts)
            #tmprho = np.zeros(npts)
            for i in range(nSites):
                Jmat = sd.J_combine_symmtop_exp_decayN(RObj.omega, vecXH[i], RObj.rotdifModel.D[0], RObj.rotdifModel.D[1], S2[i], consts[i], taus[i])
                # = = = Calculate relaxation values from the entire sample of vectors before any averagins is to be done
                tmpR1, tmpR2, tmpNOE = RObj.get_relax_from_J_simd( Jmat, CSAvalue=CSAvaluesArray[i] )
                tmprho = RObj.get_rho_from_J_simd( Jmat )
                #for j in range(npts):
                #    tmpR1[j], tmpR2[j], tmpNOE[j] = RObj.get_relax_from_J( Jmat[j] )
                #    tmprho[j] = RObj.get_rho_from_J( Jmat[j] )
                if weights is None:
                    R1 = np.mean(tmpR1)  ; R2 = np.mean(tmpR2)   ; NOE = np.mean(tmpNOE)
                    R1sig = np.std(tmpR1); R2sig = np.std(tmpR2) ; NOEsig = np.std(tmpNOE)
                    rho = np.mean(tmprho); rhosig = np.std(tmprho)
                    datablock[:,i]=[[R1,R1sig],[R2,R2sig],[NOE,NOEsig],[rho,rhosig]]
                else:
                    datablock[0,i]=gm.weighted_average_stdev(tmpR1, weights[i])
                    datablock[1,i]=gm.weighted_average_stdev(tmpR2, weights[i])
                    datablock[2,i]=gm.weighted_average_stdev(tmpNOE, weights[i])
                    datablock[3,i]=gm.weighted_average_stdev(tmprho, weights[i])
            return datablock
        else:
            #Single XH vector for each site.
            datablock=np.zeros((4,nSites), dtype=np.float32)
            for i in range(nSites):
                Jmat = sd.J_combine_symmtop_exp_decayN(RObj.omega, vecXH[i], RObj.rotdifModel.D[0], RObj.rotdifModel.D[1], S2[i], consts[i], taus[i])
                if len(Jmat.shape) == 1:
                    # A single vector was given.
                    R1, R2, NOE = RObj.get_relax_from_J( Jmat, CSAvalue=CSAvaluesArray[i] )
                    rho = RObj.get_rho_from_J( Jmat )
                datablock[:,i]=[R1,R2,NOE,rho]
            return datablock

    # = = Should only happen with fully anisotropic models.
    print( "= = ERROR: Unknown rotdifModel in the relaxation object used in calculations!", file=sys.stderr )
    return []

def optfunc_R1R2NOE_inner(datablock, expblock):
    if len(datablock.shape)==3 and len(expblock.shape)==3 :
        chisq = np.square(datablock[...,0] - expblock[...,0])
        sigsq = np.square(datablock[...,1]) + np.square(expblock[...,1])
        return np.mean( chisq/sigsq )
    elif len(datablock.shape)==3:
        chisq = np.square(datablock[...,0] - expblock)
        sigsq = np.square(datablock[...,1])
        return np.mean( chisq/sigsq )
    elif len(expblock.shape)==3:
        chisq = np.square(datablock - expblock[...,0])
        sigsq = np.square(expblock[...,1])
        return np.mean( chisq/sigsq )
    else:
        return np.mean( np.square( datablock - expblock) )

def optfunc_R1R2NOE_new(params, *args):
    """
    Single residue CSA fitting. No global parameters used.
    
    """
#def optfunc_r1r2noe( params, args=(relax_obj, num_vecs, s2_list, consts_list, taus_list, vecXH, expblock), full_output=false )
    RObj=args[0]
    S2=args[1] ; consts=args[2] ; taus=args[3]
    vecXH=args[4] ; w=args[5]
    expblock=args[-1]
    if len(expblock.shape) > 1:
        exp   = expblock[:,0]
        sigsq = np.square(expblock[:,1])
    else:
        exp   = expblock
        sigsq = np.zeros(3, dtype=np.float32 )

    if len(vecXH.shape) > 1:
        # An ensemble of vectors for each site.
        datablock=np.zeros((3,2), dtype=np.float32)
        Jmat = sd.J_combine_symmtop_exp_decayN(RObj.omega, vecXH, RObj.rotdifModel.D[0], RObj.rotdifModel.D[1], S2, consts, taus)
        # = = = Calculate relaxation values from the entire sample of vectors before any averagins is to be done
        tmpR1, tmpR2, tmpNOE = RObj.get_relax_from_J_simd( Jmat, CSAvalue=params[0] )
        if w is None:
            R1 = np.mean(tmpR1)  ; R2 = np.mean(tmpR2)   ; NOE = np.mean(tmpNOE)
            R1sig = np.std(tmpR1); R2sig = np.std(tmpR2) ; NOEsig = np.std(tmpNOE)
            datablock[:]=[[R1,R1sig],[R2,R2sig],[NOE,NOEsig]]
        else:
            datablock[0]=gm.weighted_average_stdev(tmpR1, w)
            datablock[1]=gm.weighted_average_stdev(tmpR2, w)
            datablock[2]=gm.weighted_average_stdev(tmpNOE, w)

        chisq = np.square( datablock[:,0]- exp )
        sigsq = sigsq + np.square(datablock[:,1])
        return np.mean( chisq/sigsq )
    else:
        #Single XH vector for each site.
        datablock=np.zeros(3, dtype=np.float32)
        Jmat = sd.J_combine_symmtop_exp_decayN(RObj.omega, vecXH, RObj.rotdifModel.D[0], RObj.rotdifModel.D[1], S2, consts, taus)
        R1, R2, NOE = RObj.get_relax_from_J( Jmat, CSAvalue=params[0] )
        datablock[:]=[R1,R2,NOE]
        chisq = np.square([R1,R2,NOE] - exp)
        if sigsq[0] != 0.0 :
            return  np.mean( chisq/sigsq )
        else:
            return np.mean( chisq )

def optfunc_R1R2NOE_DisoS2CSA(params, *args):
    """
    3-parameter fitting: Diso, global S2, and global CSA.
    """
#def optfunc_r1r2noe( params, args=(relax_obj, num_vecs, s2_list, consts_list, taus_list, vecXH, expblock), full_output=true )
    Diso=params[0] ; S2s=params[1] ; csa=params[2]
    RObj=args[0]
    nVecs=args[1] ; listS2=args[2] ; consts=args[3] ; taus=args[4]
    vecXH=args[5] ; w=args[6]
    expblock=args[-1]
    RObj.rotdifModel.change_Diso( Diso )
    S2loc  = [ S2s*k for k in listS2 ]
    consts_loc = [ [ S2s*k for k in j] for j in consts ]
    RObj.gX.csa=csa
    datablock = _obtain_R1R2NOErho( RObj, nVecs, S2loc, consts_loc, taus, vecXH , weights=w)
    chisq = optfunc_R1R2NOE_inner(datablock[0:3,...], expblock)
    print( "= = optimisations params( %s ) returns chi^2 %g" % (params, chisq) )
    return chisq

def optfunc_R1R2NOE_DisoCSA(params, *args):
#def optfunc_r1r2noe( params, args=(relax_obj, num_vecs, s2_list, consts_list, taus_list, vecXH, expblock), full_output=true )
    Diso=params[0] ; csa = params[1]
    RObj=args[0]
    nVecs=args[1] ; S2=args[2] ; consts=args[3] ; taus=args[4]
    vecXH=args[5] ; w=args[6]
    expblock=args[-1]
    RObj.rotdifModel.change_Diso( Diso )
    RObj.gX.csa=csa
    datablock = _obtain_R1R2NOErho( RObj, nVecs, S2, consts, taus, vecXH, weights=w)
    chisq = optfunc_R1R2NOE_inner(datablock[0:3,...], expblock)
    print( "= = optimisations params( %s ) returns chi^2 %g" % (params, chisq) )
    return chisq

def optfunc_R1R2NOE_DisoS2(params, *args):
#def optfunc_r1r2noe( params, args=(relax_obj, num_vecs, s2_list, consts_list, taus_list, vecXH, expblock), full_output=true )
    Diso=params[0] ; S2s=params[1]
    RObj=args[0]
    nVecs=args[1] ; S2=args[2] ; consts=args[3] ; taus=args[4]
    vecXH=args[5] ; w=args[6] ; CSAarray=args[7]
    expblock=args[-1]
    RObj.rotdifModel.change_Diso( Diso )
    S2loc  = [ S2s*k for k in S2 ]
    consts_loc = [ [ S2s*k for k in j] for j in consts ]
    #print( "...Scaling:", consts[0][0], consts_loc[0][0] )
    datablock = _obtain_R1R2NOErho( RObj, nVecs, S2loc, consts_loc, taus, vecXH, weights=w, CSAvaluesArray = CSAarray )
    chisq = optfunc_R1R2NOE_inner(datablock[0:3,...], expblock)
    print( "= = optimisations params( %s ) returns chi^2 %g" % (params, chisq) )
    return chisq

def optfunc_R1R2NOE_Diso(params, *args):
#def optfunc_R1R2NOE( params, args=(relax_obj, num_vecs, S2_list, consts_list, taus_list, vecXH, expblock), full_output=True )
    Diso=params[0]
    RObj=args[0]
    nVecs=args[1] ; S2=args[2] ; consts=args[3] ; taus=args[4]
    vecXH=args[5] ; w=args[6] ; CSAarray=args[7]
    expblock=args[-1]
    RObj.rotdifModel.change_Diso( Diso )
    datablock = _obtain_R1R2NOErho( RObj, nVecs, S2, consts, taus, vecXH, weights=w, CSAvaluesArray = CSAarray )
    chisq = optfunc_R1R2NOE_inner(datablock[0:3,...], expblock)
    print( "= = Optimisations params( %s ) returns Chi^2 %g" % (params, chisq) )
    return chisq

def do_global_scan_Diso( D_init, step, smin, smax, args):
    print( "= = Conduct inital global scan of D_iso. " )
    power=np.arange(smin, smax) ; nval = len(power)
    Dloc=np.zeros( nval, dtype=np.float32 ) ; chisq=np.zeros( nval, dtype=np.float32)
    for i in range(nval) :
        Dloc[i] = D_init*np.power( step, power[i] )
        chisq[i] = optfunc_R1R2NOE_Diso( [ Dloc[i] ] , *args )
        #print( " ... (Dloc, chisq): %g %g " % (Dloc[i], chisq[i]) )
    imin = np.argmin(chisq)
    Dmin = Dloc[ imin ]
    print( " ... Found minimum: (%d) %g %g" % ( imin, Dmin[imin], chisq[imin] ) )
    return Dmin

def print_fitting_params_headers( names, values, units, bFit ):
    sumstr=""
    for i in range(len(names)):
        if bFit[i]:
            s1="Optimised"
        else:
            s1="Fixed"
        tmpstr="# %s %s: %g %s\n" % (s1, names[i], values[i], units[i])
        sumstr=sumstr+tmpstr
    return sumstr

# = ============= New def section.   

def parse_rotdif_params(D=None, tau=None, aniso=None):
    #Determine diffusion model.
    if args.D is None:
        if args.tau is None:
            print("= = ERROR: No global tumbling parameters given!", file=sys.stderr)
            sys.exit(1)
        else:
            Diso  = 1.0/(6*args.tau)
            if aniso is None or aniso == 1.0:
                return sd.globalRotationalDiffusion_Isotropic(D=Diso)
            else:
                return sd.globalRotationalDiffusion_Axisymmetric(D=[Diso, aniso])
    else:
        tmp   = [float(x) for x in regexp_split('[, ]', D) if len(x)>0]
        Diso  = tmp[0]
        if len(tmp)==1:
            if aniso is None:
                return sd.globalRotationalDiffusion_Isotropic(D=Diso)
            else:
                return sd.globalRotationalDiffusion_Axisymmetric(D=[Diso, aniso])
        elif len(tmp)==2:
            return sd.globalRotationalDiffusion_Axisymmetric(D=tmp, bConvert=True)
        else:
            print("WARNING: fully anisotropic global rotdif not implemented.", file=sys.stderr)
            return None        
        
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Optimisation of spin relaxation parameters based on multiple experiments, '
                                     'based on the simulated and given local and global tumlbing chaaracteristics.\n'
                                     'operates similarly to calcualte_relaxations_from_Ct. All internal units are in picoseconds',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('expFiles', type=str, nargs='+',
                        help='One or more formatted spin relaxation experimental data, resembling this:\n'
                             '# Type NOE\n'                        
                             '# NucleiA   15N\n'
                             '# NucleiB    1H\n'
                             '# Frequency 600.133\n'
                             '1 0.9 0.05\n'
                             '...'
                       )
    parser.add_argument('--ROTDIF', dest='inputFileROTDIF', type=str, default=None,
                        help='As an alternative to the above, a ROTDIF-format input file can be given.')
    parser.add_argument('-o', '--outpref', type=str, dest='out_pref', default='out',
                        help='Output file prefix.')
    parser.add_argument('-f', '--infn', type=str, dest='in_Ct_fn', required=True,
                        help='Read a formatted file with fitted C_internal(t), taking from it the parameters.')                       
    parser.add_argument('--distfn', type=str, dest='distfn', default=None,
                        help='Vector orientation distribution of the X-H dipole, in a spherical-polar coordinates.'
                             'Without an accompanying quaternion, this is assumed to be in the principal axes frame.')
    parser.add_argument('--shiftres', type=int, default=0,
                        help='Shift the MD residue indices, e.g., to match the experiment.')
    parser.add_argument('--tu', '--time_units', type=str, dest='time_unit', default='ps',
                        help='Time units of the autocorrelation file.')
    parser.add_argument('--tau', type=float, dest='tau', default=None,
                        help='Isotropic relaxation time constant. Overwritten by Diffusion tensor values when given.')
    parser.add_argument('--aniso', type=float, dest='aniso', default=None,
                        help='Diffusion anisotropy (prolate/oblate). Overwritten by Diffusion tensor values when given.')
    parser.add_argument('-D', '--DTensor', type=str, dest='D', default=None,
                        help='The Diffusion tensor, given as Diso, Daniso, Drhomb. Entries are either comma-separated or space separated in a quote. '
                             'Note: In axisymmetric forms, when Daniso < 1 the unique axis is considered to point along x, and'
                             'when Daniso > 1 the unique axis is considered to point along z.')
    parser.add_argument('--zeta', type=float, default=0.890023,
                        help='Input optional manual zeta factor to scale-down'
                             'the S^2 of MD-based derivations by zero-point vibrations known in QM.'
                             'This is by convention 0.890023 (1.02/1.04)^6 - see Trbovic et al., Proteins, 2008 and Case, J Biomol NMR, 1999.')
    parser.add_argument('--csa', type=str, default=None,
                        help='Input manual average CSA value, if different from the assumed value, e.g. -170 ppm for 15N.'
                             'Residue-specific variations is set if a file name is given. This file should '
                             'specify on each line the residue index and then its respective CSA value.')
    parser.add_argument('--opt', '--fit', type=str, default=None,
                        help='Optimise over list of possible parameters that perturb the systematic baseline of R1/R2/NOE:\n'
                             '  (1) Diso - global tumbling rate,\n'
                             '  (2) S2   - The zeta-prefactor representing contribution of zero-point vibrations,\n'
                             '  (3) CSA  - The mean chemical-shift anisotropy of the heavy nucleus.\n'
                             '  (4) new  - WIP fitting for residue-specific chemical-shift anisotropy.\n'
                             'Using each will treat input values as an initial estimate.\n'
                             'Valid inputs: Diso, DisoS2, DisoCSA, DisoS2CSA, new, none.')
    parser.add_argument('--cycles', type=int, default=100,
                        help='For the new per-residue CSA fitting algorithm, do a maximum of N cycles between the global Diso versus the local CSA fits. '
                             'Each cycle begins by optimising Diso over all residues, then optimising each CSA separately against the new global value. '
                             'An additional convergence graph will be created.')
    parser.add_argument('--tol', type=float, default=1e-6,
                        help='The tolerance criteria for terminating the global/local optimisation cycles early, as a fractional change.')

    # = = = Initial parameter setting.
    time_start=time.time()
    args = parser.parse_args()

    # = = = Declare internal time units.
    standardTimeUnit = 'ps'
    
    # = = = Parse and set up local tumbling model
    localCtModel = fitCt.read_fittedCt_parameters( args.in_Ct_fn )
    if localCtModel.nModels == 0:
        print( "= = = ERROR: The fitted-Ct file %s was read, but did not yield any usable parameters!" % fittedCt_file )
        sys.exit(1)
    #localCtModel.report()
    #if args.time_unit != standardTimeUnit:
    #    f = sd._return_time_fact(standardTimeUnit)/sd._return_time_fact(args.time_unit)
    #    localCtModel.scale_time(f)        
    nResidSim = localCtModel.nModels
    residSim  = [ int(k) for k in localCtModel.model.keys() ]
    #if diff_type == 'symmtop' or diff_type == 'anisotropic':
    #    sanity_check_two_list(sim_resid, resNH, "resid from fitted_Ct -versus- vectors as defined in anisotropy")
                        
    # = = = Parse and set up global tumbling model
    globalRotDif = parse_rotdif_params(args.D, args.tau, args.aniso)
    if not args.distfn is None:
        globalRotDif.import_frame_vectors(args.distfn)
    #globalRotDif.report()
    
    #if bQuatRot:
    #    print( "    ....rotating input vectors into PAF frame using q_rot." )
    #    vecXH = qs.rotate_vector_simd(vecXH, q_rot)
    #    print( "    ....X-H vector input processing completed." )
    
    # = = = Parse and set up individual experiments.
    objExpts = sd.spinRelaxationExperiments(globalRotDif, localCtModel)

    for f in args.expFiles:
        objExpts.add_experiment( f )
    if args.zeta != 1.0:
        print( " = = Applying scaling of all C(t) magnitudes to account for zero-point QM vibrations (zeta) of %g" % args.zeta )
        objExpts.set_zeta( args.zeta )
    #objExpts.report()
    objExpts.map_experiment_peaknames_to_models()
    
    # = = = Parameters for global/local meta-optimisation.
    nRefinementCycles   = args.cycles
    refinementTolerance = args.tol

# = = = Section interpreting CSA input, check if it a custom single value for backwards compatibility, or an actual file.
    CSAvaluesArray = None
    if args.csa is None:
        print( "= = = Using default CSA value respective to each experiment.")
        #print( objExpts.spinrelax[0].angFreq.gA.csa )
        #CSAvaluesArray = np.repeat( relax_obj.gX.csa, num_vecs )
    else:
        # = = = Check is it is a numeric input and/or a file. File takes precedence.
        try:
            fp = open(args.csa, 'r')
            fp.close()
            bFileFound=True
        except:
            bFileFound=False

        try:
            tmp=float(args.csa)
            bIsNumeric=True
        except ValueError:
            bIsNumeric=False

        if bFileFound:            
            residCSA, CSAvaluesArray = gs.load_xy( args.csa )
            print( "= = = Using input CSA values from file %s - please ensure that the resid definitions are identical to the other files." % args.csa )
            sanity_check_two_list(sim_resid, residCSA, "resid from fitted_Ct -versus- as defined in CSA file ")
            if np.fabs(CSAvaluesArray[0]) > 1.0:
                print( "= = = NOTE: the first value is > 1.0, so assume a necessary conversion to ppm." )
                CSAvaluesArray *= 1e-6
        elif bIsNumeric:
            print( "= = = Using user-input CSA value: %g" % tmp )
            relax_obj.gX.csa=tmp
            if np.fabs(tmp)>1.0:
                print( "= = = NOTE: this value is > 1.0, so assume a necessary conversion to ppm." )
                relax_obj.gX.csa*=1e-6
            CSAvaluesArray = np.repeat( relax_obj.gX.csa, num_vecs )
        else:
            print( "= = = ERROR at parsing the --csa argument!", file=sys.stderr )
            sys.exit(1)
    
    objExpts.set_CSA_values(CSAvaluesArray)
    
    # = = = evaluation section. Determine what is to be done wioth the data.
    listOpt = args.opt
    
    if listOpt is None:
        # = = = No optimisation. Simply print the direct prediction for each experiment.
        objExpts.eval_all()
        for sp in objExpts.spinrelax:
            outputFile="%s_%sT_%s.dat" % (args.out_pref, str(round(sp.get_magnetic_field())), sp.get_name())
            fp=open(outputFile,'w')
            sp.print_values(style='xmgrace', fp=fp)
            fp.close()        
        for i, sp in enumerate(objExpts.spinrelax):
            print( sp.calc_chisq(objExpts.data[i]['y'],objExpts.data[i]['dy'], objExpts.mapModelNames[i] ))
        sys.exit()
    
    
    # = = = Based on simulation fits, obtain R1, R2, NOE for this X-H vector
    param_names=("Diso", "zeta", "CSA", "chi")
    param_scaling=( 1.0, zeta, 1.0e6, 1.0 )
    param_units=(relax_obj.timeUnit+"^-1", "a.u.", "ppm", "a.u." )
    optHeader=''
    #relax_obj.rotdif_model.change_Diso( Diso )
    if not bOptPars:
        # = = = Section 1. No minimisation is applied against experimental data.
        if bReportJomega:
            datablock = _obtain_Jomega(relax_obj, num_vecs, S2_list, consts_list, taus_list, vecXH, weights=vecXHweights)
        else:
            datablock = _obtain_R1R2NOErho(relax_obj, num_vecs, S2_list, consts_list, taus_list, vecXH, weights=vecXHweights, CSAvaluesArray = CSAvaluesArray )

        optHeader=print_fitting_params_headers(names=param_names,
                  values=np.multiply(param_scaling, (Diso, 1.0, relax_obj.gX.csa, 0.0)),
                  units=param_units,
                  bFit=(False, False, False, False) )
    else:
        # = = = Section 2. Minimisation is applied against experimental data.
        print( "= = = Reading Experimental relaxation parameter files" )
        # New code. Take into account holes in data and multiple fields
        # Where holes are, put False in the Truth Matrix
        #exp_Bfields, exp_resid, expblock, truthblock = read_exp_relaxations(args.expfn)
        # Old code below:
        exp_resid, expblock = gs.load_xys(expt_data_file)
        nres = len(exp_resid)
        ny   = expblock.shape[1]
        rho = np.zeros(nres, dtype=np.float32)
        if ny == 6:
            expblock = expblock.reshape( (nres,3,2) )
            # = = = Check if there are entries with zero uncertainty, which may break the algorithm.
            if ( np.any(expblock[...,1]==0) ):
                print("= = = WARNING: Experimental data %s contains entries with 0.00 uncertainty!" % expt_data_file, file=sys.stderr)
                if relax_obj.rotdifModel.name=='rigid_sphere':
                    print("= = = ERROR: Experimental data with partial zero uncertainties will break isotropic rotational diffusion optimisations.\n"
                          "      Please clean up your data with an appropriate uncertainty estimator.")
                    sys.exit(1)
        elif ny != 3:
            print( "= = = ERROR: The column format of the experimental relaxation file is not recognised!", file=sys.stderr )
            sys.exit(1)

        if ny == 3:
            expblock = expblock.T
        elif ny == 6:
            expblock = np.swapaxes(expblock,0,1)
        # = = Desired dimensions:
        # When no errors are given expblock is (nres, R1/R2/NOE)
        # When errors are given, 3 dimensions: (nres, R1/R2/NOE, data/error)

        # = = = Sanity Check
        # Compare the two resid_lists.
        sim_ind=None
        if not gm.list_identical( sim_resid, exp_resid ):
            print( "= = WARNING: The resids between the simulation and experiment are not the same!", file=sys.stderr )
            print( "...removing elements from the vector files that do not match.", file=sys.stderr )
            
            if not vecXH is None:
                print( "Debug (before):", len(S2_list), vecXH.shape, expblock.shape )
            else:
                print( "Debug (before):", len(S2_list), None, expblock.shape )
            print( "(resid - sim)", sim_resid )
            print( "(resid - exp)", exp_resid )
            shared_resid = np.sort( list( set(sim_resid) & set(exp_resid) ) )
            print( "(resid - shared)", shared_resid )
            fnum_vecs = len( shared_resid )
            if fnum_vecs == 0:
                print( "= = ERROR: there is no overlap between experimental and simulation residue indices!", file=sys.stderr )
                sys.exit(1)
            fnum_vecs = len( shared_resid )
            sim_ind = np.array( [ (np.where(sim_resid==x))[0][0] for x in shared_resid ] )
            fsim_resid   = [ sim_resid[x] for x in sim_ind ]
            fS2_list     = [ S2_list[x] for x in sim_ind ]
            fconsts_list = [ consts_list[x] for x in sim_ind ]
            ftaus_list   = [ taus_list[x] for x in sim_ind ]
            if not vecXH is None:
                fvecXH = vecXH.take(sim_ind, axis=0)
            else:
                fvecXH = None
            fCSAs        = CSAvaluesArray.take(sim_ind)
            if not vecXHweights is None:
                fvecXHweights = vecXHweights.take(sim_ind, axis=0)
            else:
                fvecXHweights = None
            exp_ind = np.array( [ (np.where(exp_resid==x))[0][0] for x in shared_resid ] )
            expblock = np.take(expblock, exp_ind, axis=1)

            if not vecXH is None:
                print( "Debug (after):", len(fS2_list), fvecXH.shape, expblock.shape )
            else:
                print( "Debug (after):", len(S2_list), None, expblock.shape )
            print( "(resid)", sim_resid )
        else:
            fnum_vecs    = num_vecs
            fsim_resid   = sim_resid
            fS2_list     = S2_list
            fconsts_list = consts_list
            ftaus_list   = taus_list
            fvecXH       = vecXH
            fvecXHweights = vecXHweights
            fCSAs        = CSAvaluesArray

        # = = = Do global scan of Diso = = =
        bDoGlobalScan = False
        if bDoGlobalScan:
            Diso_init = do_global_scan_Diso( Diso, step=1.05, smin=-10, smax=10, \
                args=(relax_obj, fnum_vecs, fS2_list, fconsts_list, ftaus_list, fvecXH, fvecXHweights, CSAvaluesArray, expblock) )
        else:
            Diso_init = Diso

        # = = DEBUG = =
        #print(relax_obj.rotdifModel.name)
        #sys.exit()

        if  optMode == 'new':
            print( "= = Conducting global-Diso + local-CSA refinement... this may take a while." )
            DisoOpt    = Diso_init 
            fCSAsOpt   = np.copy(fCSAs)
            fCSAsChiSq = np.zeros( fnum_vecs, dtype=np.float32 )
            DisoPrev = None ; fCSAsPrev = None
            bFirst = True
            #DisoValuesAll = np.zeros( nRefinementCycles, dtype=np.float32 )
            #DisoChiSqAll  = np.zeros( nRefinementCycles, dtype=np.float32 )
            #CSAvaluesAll  = np.zeros( (fnum_vecs, nRefinementCycles), dtype=np.float32 )
            #CSAChiSqAll   = np.zeros( (fnum_vecs, nRefinementCycles), dtype=np.float32 )
            for r in range(nRefinementCycles):
                # = = = Global Stage
                out = fmin_powell(optfunc_R1R2NOE_Diso, x0=DisoOpt, direc=[0.1*DisoOpt], \
                    args=(relax_obj, fnum_vecs, fS2_list, fconsts_list, ftaus_list, fvecXH, fvecXHweights, fCSAsOpt, expblock), full_output=True )
                DisoOpt = out[0] ; ChiSqDiso = out[1]
                #relax_obj.rotdifModel.change_Diso( DisoOpt )
                #print( relax_obj.rotdifModel.D, DisoOpt )
                #DisoValuesAll[r] = DisoOpt
                #DisoChiSqAll[r]  = ChiSqDiso
                if (not bFirst) and np.allclose(DisoOpt, DisoPrev, rtol=refinementTolerance):
                    print("= = = BREAK at Diso test.")
                    break
                DisoPrev = DisoOpt

                # = = = Local Stage
                for i in range(fnum_vecs):
                    out = fmin_powell( optfunc_R1R2NOE_new, x0=fCSAsOpt[i], \
                        args=(relax_obj, fS2_list[i], fconsts_list[i], ftaus_list[i], fvecXH[i], fvecXHweights[i], expblock[:,i,:]), full_output=True )
                    fCSAsOpt[i] = out[0] ; fCSAsChiSq[i] = out[1]
                #CSAvaluesAll[:,r] = CSAvaluesArrayOpt
                #CSAChiSqAll[:,r]  = CSAChiSqOpt

                if (not bFirst) and np.allclose(fCSAsOpt, fCSAsPrev, rtol=refinementTolerance):
                    print("= = = BREAK at CSA test")
                    break
                fCSAsPrev = fCSAsOpt
                # = = = Final check
                if bFirst:
                    bFirst=False
                print("    ...round %i complete." % r)

            print("    ....optimisation complete at round %i." % r)
            #gs.print_xy('temp_Diso_value.xvg', np.arange(1,r+1), DisoValuesAll[:r])
            #gs.print_xy('temp_Diso_chi.xvg', np.arange(1,r+1), DisoChiSqAll[:r])
            #gs.print_xylist('temp_CSA_value.xvg', np.arange(1,r+1), CSAvaluesAll[:,:r])
            #gs.print_xylist('temp_CSA_chi.xvg', np.arange(1,r+1), CSAvaluesAll[:,:r])
            optHeader=print_fitting_params_headers(names=param_names,
                      values=np.multiply(param_scaling, (DisoOpt, 1.0, np.nan, np.sqrt(ChiSqDiso)) ),
                      units=param_units,
                      bFit=(True, False, False, True) )
            optHeader=optHeader+"\n# See %s_CSA_values.dat for individual CSA optimisations." % out_pref
            print( optHeader )

            # = = = Copy back over to overall array
            if not sim_ind is None:
                for i,j in enumerate(sim_ind):
                    CSAvaluesArray[j] = fCSAsOpt[i]
            else:
                CSAvaluesArray = fCSAsOpt
            gs.print_xy(out_pref+'_CSA_values.dat', sim_resid, CSAvaluesArray )

        elif optMode == 'DisoS2CSA':
            print( "= = Fitting both Diso, S2, as well as average CSA.." )
            p_init=( Diso_init, 1.0, relax_obj.gX.csa )
            # The Powell minimisation search directions are set like this because CSA and S2 both compensate for Diso.
            dmat=np.array([[ np.sqrt(1.0/3.0), np.sqrt(1.0/3.0), np.sqrt(1.0/3.0)],
                           [-np.sqrt(2.0/3.0), np.sqrt(1.0/6.0), np.sqrt(1.0/6.0)],
                           [                0, np.sqrt(1.0/2.0),-np.sqrt(1.0/2.0)]])
            d_init=np.multiply(0.1*dmat, p_init)
            fminOut=fmin_powell( optfunc_R1R2NOE_DisoS2CSA, x0=p_init, direc=d_init, \
                args=(relax_obj, fnum_vecs, fS2_list, fconsts_list, ftaus_list, fvecXH, fvecXHweights, expblock), full_output=True )
            print( fminOut )
            Diso_opt=fminOut[0][0]
            S2s_opt  =fminOut[0][1]
            csa_opt =fminOut[0][2]
            chisq=fminOut[1]
            optHeader=print_fitting_params_headers(names=param_names,
                      values=np.multiply(param_scaling, (Diso_opt,S2s_opt,csa_opt,np.sqrt(chisq)) ),
                      units=param_units,
                      bFit=(True, True, True, True) )
            print( optHeader )

        elif optMode == 'DisoCSA':
            print( "= = Fitting both Diso and the average CSA.." )
            # = = = We need to fit two parameters: D_iso and CSA
            p_init=( Diso_init, relax_obj.gX.csa )
            # The Powell minimisation search directions are set like this because CSA and Diso compensate for each other's effects.
            d_init=( (0.1*p_init[0], 0.1*p_init[1]), (0.1*p_init[0], -0.1*p_init[1]) )
            fminOut=fmin_powell( optfunc_R1R2NOE_DisoCSA, x0=p_init, direc=d_init, \
                args=(relax_obj, fnum_vecs, fS2_list, fconsts_list, ftaus_list, fvecXH, fvecXHweights, expblock), full_output=True )
            print( fminOut )
            csa_opt=fminOut[0][1]
            Diso_opt=fminOut[0][0]
            chisq=fminOut[1]
            optHeader=print_fitting_params_headers(names=param_names,
                      values=np.multiply(param_scaling, (Diso_opt, 1.0, csa_opt, np.sqrt(chisq)) ),
                      units=param_units,
                      bFit=(True, False, True, True) )
            print( optHeader )
        elif optMode == 'DisoS2':
            print( "= = Fitting both D_iso and overall S2 scaling.." )
            p_init=( Diso_init, 1.0 )
            d_init=( (0.1*p_init[0], 0.1*p_init[1]), (0.1*p_init[0], -0.1*p_init[1]) )
            # The directions are set like this because S2 and Diso compensate for each other's effects.
            fminOut=fmin_powell( optfunc_R1R2NOE_DisoS2, x0=p_init, direc=d_init, \
                args=(relax_obj, fnum_vecs, fS2_list, fconsts_list, ftaus_list, fvecXH, fvecXHweights, CSAvaluesArray, expblock), full_output=True )
            print( fminOut )
            S2s_opt=fminOut[0][1]
            Diso_opt=fminOut[0][0]
            chisq=fminOut[1]
            optHeader=print_fitting_params_headers(names=param_names,
                      values=np.multiply(param_scaling, (Diso_opt, S2s_opt, relax_obj.gX.csa, np.sqrt(chisq)) ),
                      units=param_units,
                      bFit=(True, True, False, True) )
            print( optHeader )
            S2_list = [ S2s_opt*k for k in S2_list ]

        elif optMode == 'Diso':
            print( "= = Fitting D_iso.." )
            p_init=Diso_init
            # The directions are set like this because CSA and Diso compensate for each other's effects.
            d_init=[0.1*Diso_init]
            fminOut=fmin_powell(optfunc_R1R2NOE_Diso, x0=p_init, direc=d_init, \
                args=(relax_obj, fnum_vecs, fS2_list, fconsts_list, ftaus_list, fvecXH, fvecXHweights, CSAvaluesArray, expblock), full_output=True )
            print( fminOut )
            Diso_opt=fminOut[0]
            chisq=fminOut[1]
            optHeader=print_fitting_params_headers(names=param_names,
                      values=np.multiply(param_scaling, (Diso_opt, 1.0, relax_obj.gX.csa, np.sqrt(chisq)) ),
                      units=param_units,
                      bFit=(True, False, False, True) )
            print( optHeader )
        else:
            print( "= = Invalid optimisation mode!", file=sys.stderr )
            sys.exit(1)

        datablock = _obtain_R1R2NOErho(relax_obj, num_vecs, S2_list, consts_list, taus_list, vecXH, weights=vecXHweights, CSAvaluesArray = CSAvaluesArray )

    print( " = = Completed Relaxation calculations." )

# = = = Print
    if bReportJomega:
        fp = open(out_pref+'_Jw.dat', 'w')
        if optHeader != '':
            print( '%s' % optHeader, file=fp )
        if bHaveDy:
            print( '@type xydy', file=fp )
        s=0
        num_omega=relax_obj.num_omega
        xdat = np.fabs(relax_obj.omega)
        for i in range(num_vecs):
            print( '@s%d legend "Resid: %d"' % (i, sim_resid[i]), file=fp )
            for j in np.argsort(xdat):
                if bHaveDy:
                    print( '%g %g %g' % (xdat[j], datablock[j,i,0], datablock[j,i,1]), file=fp )
                else:
                    print( '%g %g' % (xdat[j], datablock[j,i]), file=fp )
            print( '&', file=fp )
            s+=1
        fp.close()
    else:
        if not bHaveDy:
            gs.print_xy(out_pref+'_R1.dat',  sim_resid, datablock[0,:], header=optHeader)
            gs.print_xy(out_pref+'_R2.dat',  sim_resid, datablock[1,:], header=optHeader)
            gs.print_xy(out_pref+'_NOE.dat', sim_resid, datablock[2,:], header=optHeader)
            gs.print_xy(out_pref+'_rho.dat', sim_resid, datablock[3,:])
        else:
            gs.print_xydy(out_pref+'_R1.dat',  sim_resid, datablock[0,:,0], datablock[0,:,1], header=optHeader)
            gs.print_xydy(out_pref+'_R2.dat',  sim_resid, datablock[1,:,0], datablock[1,:,1], header=optHeader)
            gs.print_xydy(out_pref+'_NOE.dat', sim_resid, datablock[2,:,0], datablock[2,:,1], header=optHeader)
            gs.print_xydy(out_pref+'_rho.dat', sim_resid, datablock[3,:,0], datablock[3,:,1])

    time_stop=time.time()
    #Report time
    print( "= = Finished. Total seconds elapsed: %g" % (time_stop - time_start) )

sys.exit()
