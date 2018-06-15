from math import *
import sys, os, argparse, time
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

def sanity_check_two_list(listA, listB, string, bVerbose=False):
    if not np.all( np.equal(listA, listB) ):
        print "= = ERROR: Sanity checked failed for %s!" % string
        if bVerbose:
            print listA
            print listB
        else:
            print "    ...first residues:", listA[0], listB[0]
            print "    ...set intersection (unordered):", set(listA).intersection(set(listB))
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
    print >> sys.stderr, "= = ERROR: Unknown rotdifModel in the relaxation object used in calculations!"
    return []


def _obtain_R1R2NOErho(RObj, nSites, S2, consts, taus, vecXH, weights=None):
    """
    The inputs vectors have dimensions (nSites, nSamples, 3) or just (nSites, 3)
    the datablock being returned has dimensions:
    - ( 4, nSites)    of there is no uncertainty calculations. 4 corresponding each to R1, R2, NOE, and rho.
    - ( 4, nSites, 2) if there is uncertainty calculations.
    """
    if RObj.rotdifModel.name == 'direct_transform':
        datablock=np.zeros((4,nSites), dtype=np.float32)
        for i in range(nSites):
            J = sd.J_direct_transform(RObj.omega, consts[i], taus[i])
            R1, R2, NOE = RObj.get_relax_from_J( J )
            rho = RObj.get_rho_from_J( J )
            datablock[:,i]=[R1,R2,NOE,rho]
        return datablock
    elif RObj.rotdifModel.name == 'rigid_sphere':
        datablock=np.zeros((4,nSites), dtype=np.float32)
        for i in range(nSites):
            J = sd.J_combine_isotropic_exp_decayN(RObj.omega, 1.0/(6.0*RObj.rotdifModel.D), S2[i], consts[i], taus[i])
            R1, R2, NOE = RObj.get_relax_from_J( J )
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
                tmpR1, tmpR2, tmpNOE = RObj.get_relax_from_J_simd( Jmat )
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
                    R1, R2, NOE = RObj.get_relax_from_J( Jmat )
                    rho = RObj.get_rho_from_J( Jmat )
                datablock[:,i]=[R1,R2,NOE,rho]
            return datablock

    # = = Should only happen with fully anisotropic models.
    print >> sys.stderr, "= = ERROR: Unknown rotdifModel in the relaxation object used in calculations!"
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

def optfunc_R1R2NOE_DisoS2CSA(params, *args):
#def optfunc_r1r2noe( params, args=(relax_obj, num_vecs, s2_list, consts_list, taus_list, vecxh, expblock), full_output=true )
    Diso=params[0] ; S2s=params[1] ; csa=params[2]
    robj=args[0]
    nvecs=args[1] ; S2=args[2] ; consts=args[3] ; taus=args[4]
    vecxh=args[5] ; w=args[6]
    expblock=args[-1]
    robj.rotdifModel.change_Diso( Diso )
    S2loc  = [ S2s*k for k in S2 ]
    consts_loc = [ [ S2s*k for k in j] for j in consts ]
    robj.gX.csa=csa
    datablock = _obtain_R1R2NOErho( robj, nvecs, S2, consts, taus, vecxh , weights=w)
    chisq = optfunc_R1R2NOE_inner(datablock[0:3,...], expblock)
    print "= = optimisations params( %s ) returns chi^2 %g" % (params, chisq)
    return chisq

def optfunc_R1R2NOE_DisoCSA(params, *args):
#def optfunc_r1r2noe( params, args=(relax_obj, num_vecs, s2_list, consts_list, taus_list, vecxh, expblock), full_output=true )
    Diso=params[0] ; csa = params[1]
    robj=args[0]
    nvecs=args[1] ; S2=args[2] ; consts=args[3] ; taus=args[4]
    vecxh=args[5] ; w=args[6]
    expblock=args[-1]
    robj.rotdifModel.change_Diso( Diso )
    robj.gX.csa=csa
    datablock = _obtain_R1R2NOErho( robj, nvecs, S2, consts, taus, vecxh, weights=w)
    chisq = optfunc_R1R2NOE_inner(datablock[0:3,...], expblock)
    print "= = optimisations params( %s ) returns chi^2 %g" % (params, chisq)
    return chisq

def optfunc_R1R2NOE_DisoS2(params, *args):
#def optfunc_r1r2noe( params, args=(relax_obj, num_vecs, s2_list, consts_list, taus_list, vecxh, expblock), full_output=true )
    Diso=params[0] ; S2s=params[1]
    robj=args[0]
    nvecs=args[1] ; S2=args[2] ; consts=args[3] ; taus=args[4]
    vecxh=args[5] ; w=args[6]
    expblock=args[-1]
    robj.rotdifModel.change_Diso( Diso )
    S2loc  = [ S2s*k for k in S2 ]
    consts_loc = [ [ S2s*k for k in j] for j in consts ]
    #print "...Scaling:", consts[0][0], consts_loc[0][0]
    datablock = _obtain_R1R2NOErho( robj, nvecs, S2loc, consts_loc, taus, vecxh, weights=w)
    chisq = optfunc_R1R2NOE_inner(datablock[0:3,...], expblock)
    print "= = optimisations params( %s ) returns chi^2 %g" % (params, chisq)
    return chisq

def optfunc_R1R2NOE_Diso(params, *args):
#def optfunc_R1R2NOE( params, args=(relax_obj, num_vecs, S2_list, consts_list, taus_list, vecXH, expblock), full_output=True )
    Diso=params[0]
    RObj=args[0]
    nvecs=args[1] ; S2=args[2] ; consts=args[3] ; taus=args[4]
    vecXH=args[5] ; w=args[6]
    expblock=args[-1]
    RObj.rotdifModel.change_Diso( Diso )
    datablock = _obtain_R1R2NOErho( RObj, nvecs, S2, consts, taus, vecXH, weights=w)
    chisq = optfunc_R1R2NOE_inner(datablock[0:3,...], expblock)
    print "= = Optimisations params( %s ) returns Chi^2 %g" % (params, chisq)
    return chisq

def do_global_scan_Diso( D_init, step, smin, smax, args):
    print "= = Conduct inital global scan of D_iso. "
    power=np.arange(smin, smax) ; nval = len(power)
    Dloc=np.zeros( nval, dtype=np.float32 ) ; chisq=np.zeros( nval, dtype=np.float32)
    for i in range(nval) :
        Dloc[i] = D_init*np.power( step, power[i] )
        chisq[i] = optfunc_R1R2NOE_Diso( [ Dloc[i] ] , *args )
        #print " ... (Dloc, chisq): %g %g " % (Dloc[i], chisq[i])
    imin = np.argmin(chisq)
    Dmin = Dloc[ imin ]
    print " ... Found minimum: (%d) %g %g" % ( imin, Dmin[imin], chisq[imin] )
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

# Read the formatted file headers in _fittedCt.dat. These are of the form:
# # Residue: 1
# # Chi-value: 1.15659e-05
# # Param XXX: ### +- ###
def read_fittedCt_file(filename):

    resid=[]
    param_name=[]
    param_val=[]
    tmp_name=[]
    tmp_val=[]
    for raw in open(filename):
        if raw == "" or raw[0]!="#":
            continue

        line=raw.split()
        if 'Residue' in line[1]:
            resid.append(int(line[-1]))
            if len(tmp_name)>0:
                param_name.append(tmp_name)
                tmp_name=[]
            if len(tmp_val)>0:
                param_val.append(tmp_val)
                tmp_val=[]
        elif 'Param' in line[1]:
            tmp_name.append(line[2][:-1])
            tmp_val.append(float(line[-3]))

    if len(tmp_name)>0:
        param_name.append(tmp_name)
        tmp_name=[]
    if len(tmp_val)>0:
        param_val.append(tmp_val)
        tmp_val=[]

    if len(resid) != len(param_name) != len(param_val):
        print >> sys.stderr, "= = ERROR in read_fittedCt_file: the header entries don't have the same number of residues as entries!"
        sys.exit(1)
    return resid, param_name, param_val

def parse_parameters(names, values):

    if len(names) != len(values):
        print >> sys.stderr, "= = ERROR in parse_parameters! The lengths of names and values arrays are not the same!"
        sys.exit(1)

    S2=np.nan ; consts=[] ; taus=[] ; Sf=np.nan
    for i in range(len(names)):
        if names[i]=='S2_0':
            S2=values[i]
        elif 'C_' in names[i]:
            consts.append(values[i])
        elif 'tau_' in names[i]:
            taus.append(values[i])
        elif 'S2_fast'==names[i]:
            Sf=values[i]
        else:
            print >> sys.stderr, "= = ERROR: parameter name not recognised! %s %s" % (names[i], values[i])

    # Relic code for when order parameters are not explicitly written:
    if np.isnan(S2):
        S2 = 1.0-np.sum( consts )
        Sf = 0.0
    if np.isnan(Sf):
        Sf = 1-S2-np.sum(consts)

    return S2, consts, taus, Sf

#def read_exp_relaxations(filename):
#    """
#    This will return a masked array.
#    """
#    bHasHeader=False
#    Bfields=[]
#    resID=[]
#    block=[]
#
#    #Read file line by line as usual
#    for l in open(fn):
#        if len(l)==0:
#            continue
#        line = l.split()
#        if line[0] == "#":
#            if "Bfield" in l:
#                print >> sys.stderr, "= = Reading Experimental File: found a Magnetib field entry", l
#                Bfields.append( float(line[-1]) )
#            elif "R1" in l:
#                bHasHeader=True
#        else:
#            resid=float(l[0])
#            data=[ float(i) for i in l[1:] ]

def convert_LambertCylindricalHist_to_vecs(hist, edges):
    print "= = = Reading histogram in Lambert-Cylindral projection, and returning distribution of non-zero vectors."
    # = = = Expect histograms as a list of 2D entries: (nResidues, phi, cosTheta)
    nResidues   = hist.shape[0]
    phis   = 0.5*(edges[0][:-1]+edges[0][1:])
    thetas = np.arccos( 0.5*(edges[1][:-1]+edges[1][1:]) )
    pt = np.moveaxis( np.array( np.meshgrid( phis, thetas, indexing='ij') ), 0, -1)
    binVecs = gm.rtp_to_xyz( pt, vaxis=-1, bUnit=True )
    del pt, phis, thetas
    print "    ...shapes of first histogram and average-vector array:", hist[0].shape, binVecs.shape
    nPoints = hist[0].shape[0]*hist[0].shape[1]
    # = = = just in case this is a list of histograms..
    # = = = Keep all of the zero-weight entries vecause it keeps the broadcasting speed.
    #vecs    = np.zeros( (nResidues, nPoints, 3 ), dtpye=binVecs.dtype )
    #weights = np.zeros_like( vecs )
    return np.repeat( binVecs.reshape(nPoints,3)[np.newaxis,...], nResidues, axis=0), \
           np.reshape( hist, ( nResidues, nPoints) )
    #vecs = [] ; weights = []
    #bFirst=True
    #for res in range(nResidues):
    #    mapp = np.where( hist[res] > 0 )
    #    nVals = len(mapp[0])
    #    tmpW = [ hist[res][i,j] for i,j in zip(mapp[0],mapp[1])   ]
    #    tmpV = [ averageVecs[i,j] for i,j in zip(mapp[0],mapp[1]) ]
    #    vecs.append(tmpV)
    #    weights.append(tmpW)
    #    if bFirst:
    #        bFirst=False
    #        print "    ...weight sum: ", np.sum(tmpW)
    # return vecs, weights

def read_vector_distribution_from_file( fileName ):
    """
    Returns the vectors, and mayber weights whose dimensions are (nResidue, nSamples, 3).
    Currently supports only phi-theta formats of vector definitions.
    For straight xmgrace data files, this corresponds to the number of plots, then the data-points in each plot.
    """
    weights = None
    if fileName.endswith('.npz'):
        # = = = Treat as a numpy binary file.
        obj = np.load(fileName)
        # = = = Determine data type
        resIDs = obj['names']
        if obj['bHistogram']:
            if obj['dataType'] != 'LambertCylindrical':
                print >> sys.stderr, "= = = Histogram projection not supported! %s" % obj['dataType']
                sys.exit(1)
            vecs, weights = convert_LambertCylindricalHist_to_vecs(obj['data'], obj['edges'])
        else:
            if obj['dataType'] != 'PhiTheta':
                print >> sys.stderr, "= = = Numpy binary datatype not supported! %s" % obj['dataType']
                sys.exit(1)
            # = = = Pass phi and theta directly to rtp_to_xyz
            vecs = gm.rtp_to_xyz( obj['data'], vaxis=-1, bUnit=True )
    else:
        resIDs, dist_phis, dist_thetas, dum = gs.load_sxydylist(args.distfn, 'legend')
        vecs = gm.rtp_to_xyz( np.stack( (dist_phis,dist_thetas), axis=-1), vaxis=-1, bUnit=True )
    if not weights is None:
        print "    ...converted input phi_theta data to vecXH / weights, whose shapes are:", vecs.shape, weights.shape
    else:
        print "    ...converted input phi_theta data to vecXH, whose shape is:", vecs.shape
    return resIDs, vecs, weights

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Read fitted-Ct values and calculation of relaxation parameters'
                                     'based on a combination of the local autocorrelation and global tumbling by assumption of'
                                     'separability: Ct= C_internal(t) * C_external(t).',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-f', '--infn', type=str, dest='in_Ct_fn',
                        help='Read a formatted file with fitted C_internal(t), taking from it the parameters.')
    parser.add_argument('-o', '--outpref', type=str, dest='out_pref', default='out',
                        help='Output file prefix.')
    parser.add_argument('-v','--vecfn', type=str, dest='vecfn', default='none',
                        help='Average vector orientations of the nuclei, e.g. N-H in a protein.'
                             'Without an accompanying quaternion, this is assumed to be in the principal axes frame.')
    parser.add_argument('--distfn', type=str, dest='distfn', default='none',
                        help='Vector orientation distribution of the X-H dipole, in a spherical-polar coordinates.'
                             'Without an accompanying quaternion, this is assumed to be in the principal axes frame.')
    parser.add_argument('--shiftres', type=int, default=0,
                        help='Shift the MD residue indices, e.g., to match the experiment.')
    parser.add_argument('-e','--expfn', type=str, dest='expfn', default='none',
                        help='Experimental values of R1, R2, and NOE in a 4- or 7-column format.'
                             '4-column data is assumed to be ResID/R1/R2/NOE. 7-column data is assumed to also have errors.'
                             'Giving this file currently will compute the rho equivalent and quit, unless --opt is also given.')
    parser.add_argument('--ref', type=str, dest='reffn', default='none',
                        help='Reference PDB file for an input trajectory to determine distribution of X-H vectors.'
                             'WARNING: not yet implemented.')
    parser.add_argument('--traj', type=str, dest='trjfn', default='none',
                        help='Input trajectory from which X-H vector distributions will be read.')
    parser.add_argument('-q', '--q_rot', type=str, dest='qrot_str', default='',
                        help='Rotation quaternion from the lab frame of the vectors into the PAF, where D is diagonalised.'
                             'Give as "q_w q_x q_y q_z"')
    parser.add_argument('-n', '--nuclei', type=str, dest='nuclei', default='NH',
                        help='Type of nuclei measured by the NMR spectrometer. Determines the gamma constants used.')
    parser.add_argument('-B', '--B0', type=float, dest='B0', default=-1,
                        help='Magnetic field of the nmr spectrometer in T. overwritten if the frequency is given.')
    parser.add_argument('-F', '--freq', type=float, dest='Hz', default=-1,
                        help='Proton frequency of the NMR spectrometer in Hz. Overwrites B0 argument when given.')
    parser.add_argument('--Jomega', action='store_true',
                        help='Calculate Jomega instead of R1, R2, NOE, and rho.')
    parser.add_argument('--tu', '--time_units', type=str, dest='time_unit', default='ps',
                        help='Time units of the autocorrelation file.')
    parser.add_argument('--tau', type=float, dest='tau', default=np.nan,
                        help='Isotropic relaxation time constant. Overwritten by Diffusion tensor values when given.')
    parser.add_argument('--aniso', type=float, dest='aniso', default=1.0,
                        help='Diffusion anisotropy (prolate/oblate). Overwritten by Diffusion tensor values when given.')
    parser.add_argument('-D', '--DTensor', type=str, dest='D', default='-1',
                        help='The Diffusion tensor, given as Diso, Daniso, Drhomb.'
                             'Note: In axisymmetric forms, when Daniso < 1 the unique axis is considered to point along x, and'
                             'when Daniso > 1 the unique axis is considered to point along z.')
    parser.add_argument('--rXH', type=float, default=np.nan,
                        help='Alternative formulation to zeta by setting the effective bond-length rXH, which in 15N--1H is 1.02 Angs. by default.'
                            'Case says this can be modified to 1.04, or 1.039 according to LeMasters.')
    parser.add_argument('--zeta', type=float, default=0.890023,
                        help='Input optional manual zeta factor to scale-down'
                             'the S^2 of MD-based derivations by zero-point vibrations known in QM.'
                             'This is by convention 0.890023 (1.02/1.04)^6 - see Trbovic et al., Proteins, 2008 and Case, J Biomol NMR, 1999.')
    parser.add_argument('--csa', type=float, default=np.nan,
                        help='Input manual average CSA value, if different from the assumed value, e.g. -170 ppm for 15N')
    parser.add_argument('--opt', type=str, default='none',
                        help='Optimise over one or two parameters that perturb the systematic baseline of R1/R2/NOE:'
                             'the chemical-shift anisotropy of the heavy nucleus (15N), and the global tumbling.'
                             'This will treat the input D_iso as an initial estimate.'
                             'Valid inputs: Diso, DisoS2, DisoCSA, none.')
    parser.add_argument('--rigid', dest='bRigid', action='store_true',
                        help='Compute the spin-relaxation corresponding to a rigid sphere with rotational diffusion coefficient D_iso/tau_iso, then exit.')


    time_start=time.clock()

    args = parser.parse_args()
    fittedCt_file = args.in_Ct_fn
    out_pref=args.out_pref
    bHaveDy = False
    if args.opt != 'none':
        bOptPars = True
        optMode = args.opt
    else:
        bOptPars = False
    bJomega = args.Jomega
    zeta = args.zeta
    if zeta != 1.0:
        print " = = Applying scaling to add zero-point QM vibrations (zeta) of %g" % zeta
    # Set up relaxation parameters.
    nuclei_pair  = args.nuclei
    timeUnit = args.time_unit
    if args.Hz != -1:
        B0 = 2.0*np.pi*args.Hz / 267.513e6
    elif args.B0 != -1:
        B0 = args.B0
    else:
        print >> sys.stderr, "= = = ERROR: Must give either the background magnetic field or the frequency! E.g., -B0 14.0956"
        sys.exit(1)

    relax_obj = sd.relaxationModel(nuclei_pair, B0)
    relax_obj.set_time_unit(timeUnit)
    relax_obj.set_freq_relaxation()
    print "= = = Setting up magnetic field:", B0, "T"
    print "= = = Angular frequencies in ps^-1 based on given parameters:"
    relax_obj.print_freq_order()
    print relax_obj.omega
    print "= = = Gamma values: (X) %g , (H) %g" % (relax_obj.gX.gamma, relax_obj.gH.gamma)
    if np.isnan(args.csa):
        print "= = = Using default CSA value: %g" % relax_obj.gX.csa
    else:
        tmp=args.csa
        if (np.fabs(tmp)>1):
            tmp*=1e-6
        relax_obj.gX.csa=tmp
        print "= = = Using input CSA value: %g" % relax_obj.gX.csa

    #Check if the experimental file is given
    if args.expfn != 'none':
        print "= = = Experimental relaxation parameters given."
        # New code. Take into account holes in data and multiple fields
        # Where holes are, put False in the Truth Matrix
        #exp_Bfields, exp_resid, expblock, truthblock = read_exp_relaxations(args.expfn)

        # Old code below:
        exp_resid, expblock = gs.load_xys(args.expfn)
        nres = len(exp_resid)
        ny   = expblock.shape[1]
        rho = np.zeros(nres, dtype=np.float32)
        if ny == 6:
            expblock = expblock.reshape( (nres,3,2) )
        elif ny != 3:
            print >> sys.stderr, "= = = ERROR: The column format of the experimental relaxation file is not recognised!"
            sys.exit(1)
        if not bOptPars:
            rho = np.zeros ( nres, dtype=np.float32)
            if ny==6:
                for i in range(nres):
                    rho[i]=relax_obj.calculate_rho_from_relaxation(expblock[i,:,0])
            else:
                for i in range(nres):
                    rho[i]=relax_obj.calculate_rho_from_relaxation(expblock[i])
            out_fn = out_pref+'_expRho.dat'
            if ny == 3:
                gs.print_xy(out_fn, exp_resid, rho)
            elif ny == 6:
                gs.print_xy(out_fn, exp_resid, rho)
                #gs.print_xydy(out_fn, exp_resid, rho[:,0], rho[:,1])
            sys.exit(0)
        # = = = Prepare datablock for fitting.
        else:
            if ny == 3:
                expblock = expblock.T
            elif ny == 6:
                expblock = np.swapaxes(expblock,0,1)
        # = = Desired dimensions:
        # When no errors are given expblock is (nres, R1/R2/NOE)
        # When errors are given, 3 dimensions: (nres, R1/R2/NOE, data/error)

    #Determine diffusion model.
    if args.D == '-1':
        if np.isnan(args.tau):
            diff_type = 'direct'
            Diso = 0.0
        else:
            tau_iso=args.tau
            Diso  = 1.0/(6*args.tau)
            if args.aniso != 1.0:
                aniso = args.aniso
                diff_type = 'symmtop'
            else:
                diff_type = 'spherical'
    else:
        tmp   = [ float(x) for x in args.D.split() ]
        Diso  = tmp[0]
        if len(tmp)==1:
            diff_type = 'spherical'
        elif len(tmp)==2:
            aniso = tmp[1]
            diff_type = 'symmtop'
        else:
            aniso = tmp[1]
            rhomb = tmp[2]
            diff_type = 'anisotropic'
        tau_iso = 1.0/(6*Diso)

    vecXH=None ; vecXHweights=None
    if diff_type=='direct':
        print "= = = No global rotational diffusion selected. Calculating the direct transform."
        relax_obj.set_rotdif_model('direct_transform')
    elif diff_type=='spherical':
        print "= = = Using a spherical rotational diffusion model."
        relax_obj.set_rotdif_model('rigid_sphere_D', Diso)
    elif diff_type=='symmtop':
        Dperp = 3.*Diso/(2+aniso)
        Dpar  = aniso*Dperp
        print "= = = Calculated anisotropy to be: ", aniso
        print "= = = With Dpar, Dperp: %g, %g %s^-1" % ( Dpar, Dperp, timeUnit)
        # This part is ignored for now..
        relax_obj.set_rotdif_model('rigid_symmtop_D', Dpar, Dperp)
        # Read quaternion
        if args.qrot_str != "":
            bQuatRot = True
            q_rot = [ float(v) for v in args.qrot_str.split() ]
            if not qops.qisunit(q_rot):
                q_rot = q_rot/np.linalg.norm(q_rot)
        else:
            bQuatRot = False

        # Read the source of vectors.
        bHaveVec = False ; bHaveVDist = False ; bHaveTraj = False
        if args.vecfn != 'none':
            print "= = = Using average vectors. Reading X-H vectors from %s ..." % args.vecfn
            resNH, vecXH = gs.load_xys(args.vecfn, dtype=float32)
            if bQuatRot:
                print "    ....rotating input vectors into PAF frame using q_rot."
                vecXH = qs.rotate_vector_simd(vecXH, q_rot)
                #for i in range(len(vecXH)):
                #    vecXH[i] = qops.rotate_vector(vecXH[i], q_rot)
            print "    ....X-H vector input processing completed."
            bHaveVec = True
        elif args.distfn != 'none':
            print "= = = Using vector distribution in spherical coordinates. Reading X-H vector distribution from %s ..." % args.distfn
            resNH, vecXH, vecXHweights = read_vector_distribution_from_file( args.distfn )

            resNH = [ int(x)+args.shiftres for x in resNH ]

            if bQuatRot:
                print "    ....rotating input vectors into PAF frame using q_rot."
                vecXH = qs.rotate_vector_simd(vecXH, q_rot)
                #shape = vecXH.shape
                #for i in range(shape[0]):
                #    for j in range(shape[1]):
                #        vecXH[i][j] = qops.rotate_vector(vecXH[i][j], q_rot)
            print "    ....X-H vector distribution input processing completed."
            bHaveVDist = True
            bHaveDy = True ;# We have an ensemble of vectors now.
        elif args.trjfn != 'none' and args.reffn != 'none' :
            # WARNING: not implemented
            import mdtraj as md
            print "= = = Using distribution of vectors from trajectory. Reading reference file %s ..." % args.reffn
            reffile = md.load(args.reffn)
            tmpH, tmpX = confirm_seltxt(ref, Hseltxt, Xseltxt)
            trjfile = md.load(args.trjfn, top=args.reffn)
            print "= = = trajectory file %s loaded." % args.trjfn
            bHaveTraj = True
            bHaveDy = True
        elif not args.bRigid:
            print >> sys.stderr, "= = = ERROR: non-spherical diffusion models require a vector source!" \
                                    "Please supply the average vectors or a trajectory and reference!"
            sys.exit(1)

# = = = Now that the basic stats habe been stored, check for --rigid shortcut.
#       This shortcut will exit after this if statement is run.
    if args.bRigid:
        if diff_type == 'direct':
            print >> sys.stderr, "= = = ERROR: Rigid-sphere argument cannot be applied without an input for the global rotational diffusion!"
            sys.exit(1)
        if diff_type == 'spherical' or diff_type == 'anisotropic':
            num_vecs=1 ; S2_list=[zeta] ; consts_list=[[0.]] ; taus_list=[[99999.]] ; vecXH=[]
        else:
            num_vecs=3 ; S2_list=[zeta,zeta,zeta] ; consts_list=[[0.],[0.],[0.]] ; taus_list=[[99999.],[99999.],[99999.]] ; vecXH=np.identity(3)
        datablock = _obtain_R1R2NOErho(relax_obj, num_vecs, S2_list, consts_list, taus_list, vecXH)
        if diff_type == 'spherical':
            print "...Isotropic baseline values:"
        else:
            print "...Anistropic axial baseline values (x/y/z):"
        print "R1:",  str(datablock[0]).strip('[]')
        print "R2:",  str(datablock[1]).strip('[]')
        print "NOE:", str(datablock[2]).strip('[]')
        sys.exit()

# = = = Read fitted C(t). For each residue, we expect a set of parameters
#       corresponding to S2 and the other parameters.
    sim_resid, param_names, param_vals = read_fittedCt_file(fittedCt_file)
    num_vecs = len(sim_resid)
    if diff_type == 'symmtop' or diff_type == 'anisotropic':
        sanity_check_two_list(sim_resid, resNH, "resid from fitted_Ct versus vectors")

    S2_list=[] ; taus_list=[] ; consts_list=[]
    for i in range(num_vecs):
        # Parse input parameters
        S2, consts, taus, Sf = parse_parameters( param_names[i], param_vals[i] )

        # = = = This section applies zero-point corrections to S2, 0.89 = = =
        S2 *= zeta
        consts = [ k*zeta for k in consts ]
        S2_list.append(S2) ; taus_list.append(taus) ; consts_list.append(consts)

    # = = = Based on simulation fits, obtain R1, R2, NOE for this X-H vector
    param_names=("Diso", "zeta", "CSA", "chi")
    param_scaling=( 1.0, zeta, 1.0e6, 1.0 )
    param_units=(relax_obj.timeUnit, "a.u.", "ppm", "a.u." )
    optHeader=''
    #relax_obj.rotdif_model.change_Diso( Diso )
    if not bOptPars:
        if bJomega:
            datablock = _obtain_Jomega(relax_obj, num_vecs, S2_list, consts_list, taus_list, vecXH, weights=vecXHweights)
        else:
            datablock = _obtain_R1R2NOErho(relax_obj, num_vecs, S2_list, consts_list, taus_list, vecXH, weights=vecXHweights)

        optHeader=print_fitting_params_headers(names=param_names,
                  values=np.multiply(param_scaling, (Diso, 1.0, relax_obj.gX.csa, 0.0)),
                  units=param_units,
                  bFit=(False, False, False, False) )
    else:
        # = = = Sanity Check
        # Compare the two resid_lists.
        if not np.all(sim_resid == exp_resid ):
            print >> sys.stderr, "= = WARNING: The resids between the simulation and experiment are not the same!"
            print >> sys.stderr, "...removing elements from the vector files that do not match."

            print "Debug (before):", len(S2_list), vecXH.shape, expblock.shape
            print "(resid - sim)", sim_resid
            print "(resid - exp)", exp_resid
            shared_resid = np.sort( list( set(sim_resid) & set(exp_resid) ) )
            print "(resid - shared)", shared_resid
            fnum_vecs = len( shared_resid )
            sim_ind = np.array( [ (np.where(sim_resid==x))[0][0] for x in shared_resid ] )
            fsim_resid   = [ sim_resid[x] for x in sim_ind ]
            fS2_list     = [ S2_list[x] for x in sim_ind ]
            fconsts_list = [ consts_list[x] for x in sim_ind ]
            ftaus_list   = [ taus_list[x] for x in sim_ind ]
            fvecXH       = vecXH.take(sim_ind, axis=0)
            if not vecXHweights is None:
                fvecXHweights = vecXHweights.take(sim_ind, axis=0)
            else:
                fvecXHweights = None
            exp_ind = np.array( [ (np.where(exp_resid==x))[0][0] for x in shared_resid ] )
            expblock = np.take(expblock, exp_ind, axis=1)

            print "Debug (after):", len(fS2_list), fvecXH.shape, expblock.shape
            print "(resid)", sim_resid
        else:
            fnum_vecs    = num_vecs
            fsim_resid   = sim_resid
            fS2_list     = S2_list
            fconsts_list = consts_list
            ftaus_list   = taus_list
            fvecXH       = vecXH
            fvecXHweights = vecXHweights

        # = = = Do global scan of Diso = = =
        bDoGlobalScan = False
        if bDoGlobalScan:
            Diso_init = do_global_scan_Diso( Diso, step=1.05, smin=-10, smax=10, args=(relax_obj, fnum_vecs, fS2_list, fconsts_list, ftaus_list, fvecXH, fvecXHweights, expblock) )
        else:
            Diso_init = Diso

        if optMode == 'DisoS2CSA':
            print "= = Fitting both Diso, S2, as well as average CSA.."
            p_init=( Diso_init, 1.0, relax_obj.gX.csa )
            # The directions are set like this because CSA and S2 both compensate for Diso.
            dmat=np.array([[ np.sqrt(1.0/3.0), np.sqrt(1.0/3.0), np.sqrt(1.0/3.0)],
                           [-np.sqrt(2.0/3.0), np.sqrt(1.0/6.0), np.sqrt(1.0/6.0)],
                           [                0, np.sqrt(1.0/2.0),-np.sqrt(1.0/2.0)]])
            d_init=np.multiply(0.1*dmat, p_init)
            fminOut=fmin_powell( optfunc_R1R2NOE_DisoS2CSA, x0=p_init, direc=d_init, args=(relax_obj, fnum_vecs, fS2_list, fconsts_list, ftaus_list, fvecXH, fvecXHweights, expblock), full_output=True )
            print fminOut
            Diso_opt=fminOut[0][0]
            S2s_opt  =fminOut[0][1]
            csa_opt =fminOut[0][2]
            chisq=fminOut[1]
            optHeader=print_fitting_params_headers(names=param_names,
                      values=np.multiply(param_scaling, (Diso_opt,S2s_opt,csa_opt,np.sqrt(chisq)) ),
                      units=param_units,
                      bFit=(True, True, True, True) )
            print optHeader

        elif optMode == 'DisoCSA':
            print "= = Fitting both Diso and the average CSA.."
            # = = = We need to fit two parameters: D_iso and CSA
            p_init=( Diso_init, relax_obj.gX.csa )
            # The directions are set like this because CSA and Diso compensate for each other's effects.
            d_init=( (0.1*p_init[0], 0.1*p_init[1]), (0.1*p_init[0], -0.1*p_init[1]) )
            fminOut=fmin_powell( optfunc_R1R2NOE_DisoCSA, x0=p_init, direc=d_init, args=(relax_obj, fnum_vecs, fS2_list, fconsts_list, ftaus_list, fvecXH, fvecXHweights, expblock), full_output=True )
            print fminOut
            csa_opt=fminOut[0][1]
            Diso_opt=fminOut[0][0]
            chisq=fminOut[1]
            optHeader=print_fitting_params_headers(names=param_names,
                      values=np.multiply(param_scaling, (Diso_opt, 1.0, csa_opt, np.sqrt(chisq)) ),
                      units=param_units,
                      bFit=(True, False, True, True) )
            print optHeader
        elif optMode == 'DisoS2':
            print "= = Fitting both D_iso and overal S2 scaling.."
            p_init=( Diso_init, 1.0 )
            d_init=( (0.1*p_init[0], 0.1*p_init[1]), (0.1*p_init[0], -0.1*p_init[1]) )
            # The directions are set like this because S2 and Diso compensate for each other's effects.
            fminOut=fmin_powell( optfunc_R1R2NOE_DisoS2, x0=p_init, direc=d_init, args=(relax_obj, fnum_vecs, fS2_list, fconsts_list, ftaus_list, fvecXH, fvecXHweights, expblock), full_output=True )
            print fminOut
            S2s_opt=fminOut[0][1]
            Diso_opt=fminOut[0][0]
            chisq=fminOut[1]
            optHeader=print_fitting_params_headers(names=param_names,
                      values=np.multiply(param_scaling, (Diso_opt, S2s_opt, relax_obj.gX.csa, np.sqrt(chisq)) ),
                      units=param_units,
                      bFit=(True, True, False, True) )
            print optHeader
            S2_list = [ S2s_opt*k for k in S2_list ]

        elif optMode == 'Diso':
            print "= = Fitting D_iso.."
            p_init=Diso_init
            # The directions are set like this because CSA and Diso compensate for each other's effects.
            d_init=[0.1*Diso_init]
            fminOut=fmin_powell(optfunc_R1R2NOE_Diso, x0=p_init, direc=d_init, args=(relax_obj, fnum_vecs, fS2_list, fconsts_list, ftaus_list, fvecXH, fvecXHweights, expblock), full_output=True )
            print fminOut
            Diso_opt=fminOut[0]
            chisq=fminOut[1]
            optHeader=print_fitting_params_headers(names=param_names,
                      values=np.multiply(param_scaling, (Diso_opt, 1.0, relax_obj.gX.csa, np.sqrt(chisq)) ),
                      units=param_units,
                      bFit=(True, False, False, True) )
            print optHeader
        else:
            print >> sys.stderr, "= = Invalid optimisation mode!"
            sys.exit(1)

        datablock = _obtain_R1R2NOErho(relax_obj, num_vecs, S2_list, consts_list, taus_list, vecXH, weights=vecXHweights)

    print " = = Completed Relaxation calculations."

# = = = Print
    if bJomega:
        fp = open(out_pref+'_Jw.dat', 'w')
        if optHeader != '':
            print >> fp, '%s' % optHeader
        if bHaveDy:
            print >> fp, '@type xydy'
        s=0
        num_omega=relax_obj.num_omega
        xdat = np.fabs(relax_obj.omega)
        for i in range(num_vecs):
            print >> fp, '@s%d legend "Resid: %d"' % (i, sim_resid[i])
            for j in np.argsort(xdat):
                if bHaveDy:
                    print >> fp, '%g %g %g' % (xdat[j], datablock[j,i,0], datablock[j,i,1])
                else:
                    print >> fp, '%g %g' % (xdat[j], datablock[j,i])
            print >> fp, '&'
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

    time_stop=time.clock()
    #Report time
    print "= = Finished. Total seconds elapsed: %g" % (time_stop - time_start)

sys.exit()
