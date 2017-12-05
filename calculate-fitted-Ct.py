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
        sys.exit(1)
    return

def _obtain_Jomega(RObj, nvecs, S2, consts, taus, vecXH):
    if RObj.rotdif_model.name == 'rigid_sphere':
        datablock=np.zeros((5,nvecs))
        for i in range(nvecs):
            J = sd.J_combine_isotropic_exp_decayN(RObj.omega, 1.0/(6.0*RObj.rotdif_model.D), S2[i], consts[i], taus[i])
            datablock[:,i]=J
        return datablock
    elif RObj.rotdif_model.name == 'rigid_symmtop':
        # Automatically use the vector-form of function.
        if len(vecXH.shape) > 2:
            # An ensemble of vector for each dipole.
            datablock=np.zeros((5,nvecs,2))
            npts=vecXH.shape[1]
            tmpJ = np.zeros( (5, npts) )
            for i in range(nvecs):
                Jmat = sd.J_combine_symmtop_exp_decayN(RObj.omega, vecXH[i], RObj.rotdif_model.D[0], RObj.rotdif_model.D[1], S2[i], consts[i], taus[i])
                datablock[:,i,0] = np.mean(Jmat, axis=0)
                datablock[:,i,1] = np.std(Jmat, axis=0)
            return datablock
        else:
            #Single XH vector for each dipole.
            datablock=np.zeros((5,nvecs))
            for i in range(nvecs):
                Jmat = sd.J_combine_symmtop_exp_decayN(RObj.omega, vecXH[i], RObj.rotdif_model.D[0], RObj.rotdif_model.D[1], S2[i], consts[i], taus[i])
                datablock[:,i]=Jmat
            return datablock

    # = = Should only happen with fully anisotropic models.
    print >> sys.stderr, "= = ERROR: Unknown rotdif_model in the relaxation object used in calculations!"
    return []


def _obtain_R1R2NOErho(RObj, nvecs, S2, consts, taus, vecXH):
    if RObj.rotdif_model.name == 'rigid_sphere':
        datablock=np.zeros((4,nvecs))
        for i in range(nvecs):
            J = sd.J_combine_isotropic_exp_decayN(RObj.omega, 1.0/(6.0*RObj.rotdif_model.D), S2[i], consts[i], taus[i])
            R1, R2, NOE = RObj.get_relax_from_J( J )
            rho = RObj.get_rho_from_J( J )
            datablock[:,i]=[R1,R2,NOE,rho]
        return datablock
    elif RObj.rotdif_model.name == 'rigid_symmtop':
        # Automatically use the vector-form of function.
        if len(vecXH.shape) > 2:
            # An ensemble of vector for each dipole.
            datablock=np.zeros((4,nvecs,2))
            npts=vecXH.shape[1]
            tmpR1  = np.zeros(npts) ; tmpR2 = np.zeros(npts) ; tmpNOE = np.zeros(npts)
            tmprho = np.zeros(npts)
            for i in range(nvecs):
                Jmat = sd.J_combine_symmtop_exp_decayN(RObj.omega, vecXH[i], RObj.rotdif_model.D[0], RObj.rotdif_model.D[1], S2[i], consts[i], taus[i])
                # = = = Calculate values from the entire sample of vectors
                for j in range(npts):
                    tmpR1[j], tmpR2[j], tmpNOE[j] = RObj.get_relax_from_J( Jmat[j] )
                    tmprho[j] = RObj.get_rho_from_J( Jmat[j] )
                R1 = np.mean(tmpR1)  ; R2 = np.mean(tmpR2)   ; NOE = np.mean(tmpNOE)
                R1sig = np.std(tmpR1); R2sig = np.std(tmpR2) ; NOEsig = np.std(tmpNOE)
                rho = np.mean(tmprho); rhosig = np.std(tmprho)
                datablock[:,i]=[[R1,R1sig],[R2,R2sig],[NOE,NOEsig],[rho,rhosig]]
            return datablock
        else:
            #Single XH vector for each dipole.
            datablock=np.zeros((4,nvecs))
            for i in range(nvecs):
                Jmat = sd.J_combine_symmtop_exp_decayN(RObj.omega, vecXH[i], RObj.rotdif_model.D[0], RObj.rotdif_model.D[1], S2[i], consts[i], taus[i])
                if len(Jmat.shape) == 1:
                    # A single vector was given.
                    R1, R2, NOE = RObj.get_relax_from_J( Jmat )
                    rho = RObj.get_rho_from_J( Jmat )
                datablock[:,i]=[R1,R2,NOE,rho]
            return datablock

    # = = Should only happen with fully anisotropic models.
    print >> sys.stderr, "= = ERROR: Unknown rotdif_model in the relaxation object used in calculations!"
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
    vecxh=args[5]
    expblock=args[6]
    robj.rotdif_model.change_Diso( Diso )
    S2loc  = [ S2s*k for k in S2 ]
    consts_loc = [ [ S2s*k for k in j] for j in consts ]
    robj.gX.csa=csa
    datablock = _obtain_R1R2NOErho( robj, nvecs, S2, consts, taus, vecxh )
    chisq = optfunc_R1R2NOE_inner(datablock[0:3,...], expblock)
    print "= = optimisations params( %s ) returns chi^2 %g" % (params, chisq)
    return chisq

def optfunc_R1R2NOE_DisoCSA(params, *args):
#def optfunc_r1r2noe( params, args=(relax_obj, num_vecs, s2_list, consts_list, taus_list, vecxh, expblock), full_output=true )
    Diso=params[0] ; csa = params[1]
    robj=args[0]
    nvecs=args[1] ; S2=args[2] ; consts=args[3] ; taus=args[4]
    vecxh=args[5]
    expblock=args[6]
    robj.rotdif_model.change_Diso( Diso )
    robj.gX.csa=csa
    datablock = _obtain_R1R2NOErho( robj, nvecs, S2, consts, taus, vecxh)
    chisq = optfunc_R1R2NOE_inner(datablock[0:3,...], expblock)
    print "= = optimisations params( %s ) returns chi^2 %g" % (params, chisq)
    return chisq

def optfunc_R1R2NOE_DisoS2(params, *args):
#def optfunc_r1r2noe( params, args=(relax_obj, num_vecs, s2_list, consts_list, taus_list, vecxh, expblock), full_output=true )
    Diso=params[0] ; S2s=params[1]
    robj=args[0]
    nvecs=args[1] ; S2=args[2] ; consts=args[3] ; taus=args[4]
    vecxh=args[5]
    expblock=args[6]
    robj.rotdif_model.change_Diso( Diso )
    S2loc  = [ S2s*k for k in S2 ]
    consts_loc = [ [ S2s*k for k in j] for j in consts ]
    #print "...Scaling:", consts[0][0], consts_loc[0][0]
    datablock = _obtain_R1R2NOErho( robj, nvecs, S2loc, consts_loc, taus, vecxh)
    chisq = optfunc_R1R2NOE_inner(datablock[0:3,...], expblock)
    print "= = optimisations params( %s ) returns chi^2 %g" % (params, chisq)
    return chisq

def optfunc_R1R2NOE_Diso(params, *args):
#def optfunc_R1R2NOE( params, args=(relax_obj, num_vecs, S2_list, consts_list, taus_list, vecXH, expblock), full_output=True )
    Diso=params[0]
    RObj=args[0]
    nvecs=args[1] ; S2=args[2] ; consts=args[3] ; taus=args[4]
    vecXH=args[5]
    expblock=args[6]
    RObj.rotdif_model.change_Diso( Diso )
    datablock = _obtain_R1R2NOErho( RObj, nvecs, S2, consts, taus, vecXH)
    chisq = optfunc_R1R2NOE_inner(datablock[0:3,...], expblock)
    print "= = Optimisations params( %s ) returns Chi^2 %g" % (params, chisq)
    return chisq

def do_global_scan_Diso( D_init, step, smin, smax, args):
    print "= = Conduct inital global scan of D_iso. "
    power=np.arange(smin, smax) ; nval = len(power)
    Dloc=np.zeros( nval ) ; chisq=np.zeros( nval )
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

# File Input. Reading the data accordingly.

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

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='This script reads in the raw autocorrelation functions C(t) '
                                     'and fits one or a set of simple exponential decay compnonets to them. '
                                     'The number of fitted components are be default determined based on increasing '
                                     'chi-agreement, where increases below a certain threshold is determined to '
                                     'be insufficiently beneficial and is treated as overfitting.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-f', '--infn', type=str, dest='in_Ct_fn', nargs='+',
                        help='One or more files containing autocorrelation functions from independent trajectories. '
                             'Each file contains a list of C(t) corresponding to residues to be fitted separately. '
                             'The relevant residue number is found via the legend text in xmgrace format. '
                             'If multiple files are given, each corresponging C(t) will be averaged '
                             'before a single fitting is calculated.')
    parser.add_argument('-o', '--outpref', type=str, dest='out_pref', default='out',
                        help='Output file prefix.')
    parser.add_argument('--nc', type=int, default=-1,
                        help='number of transient components to fit. '
                             'Leaving this at -1 allows the software to search for the maximum number of '
                             'components that should be used to fit the decay curve. '
                             'Setting a specific number allows test like using the Lipari-Szabo model.')
    parser.add_argument('--nofast', dest='bNoFast', action='store_true', default=False,
                        help='By default, this script permits an S_fast component, such that '
                             'the fitted parameters do not have to sum to unity. This wtichs turns that off '
                             'so that C(0) must be one.')
    #parser.add_argument('--zeta', type=float, default=1.0,
    #                    help='Premultiply all proportional constants by an optional correction factor, e.g. '
    #                         'representing quantum-mechanical, zero-point vibrations '
    #                         'that are not modelled in classical dynamics. '
    #                         'According to Case (1999), this is 0.89~(1.02/1.04)^6 in modelpeptides at MP2 theory.')

    time_start=time.clock()

    args = parser.parse_args()
    in_file_list = args.in_Ct_fn
    out_pref=args.out_pref
    nc=args.nc
    #zeta=args.zeta
    bHaveDy = False

    # = = = Read C(t), and averge if more than one
    num_files = len(in_file_list)
    print "= = = Found %d input C(t) files." % num_files
    if (num_files == 1):
        legs, dt, Ct, Cterr = gs.load_sxydylist(in_file_list[0], 'legend')
        legs = [ float(x) for x in legs ]
    else:
        print "    ...will perform averaging to obtain averaged C(t)."
        print "    ....WARNING: untested. Please verify!"
        dt_prev=[] ; Ct_list=[] ; Cterr_list=[]
        for ind in range(num_files):
            legs, dt, Ct, Cterr = gs.load_sxydylist(in_file_list[ind], 'legend')
            legs = [ float(x) for x in legs ]
            dt_prev=dt ; Ct_list.append(Ct) ; Cterr_list.append(Cterr)
        # = = = Ct_list is a list of 2D arrays.
        # = = = Perform grand average over individual observations. Assuming equal weights (dangerous!)
        Ct_list = np.array(Ct_list)
        Ct = np.mean(Ct_list, axes=0)
        if len(Cterr_list[0]) == 0:
            Cterr = np.std(Ct_list, axes=0)
        else:
            shape=Ct.shape
            Cterr_list = np.array(Cterr_list)
            Cterr = np.zeros(shape)
            for i in range(shape[0]):
                for j in range(shape[1]):
                    Cterr[i,j] = gm.simple_total_mean_square(Ct_list[:,i,j], Cterr_list[:,i,j])
        del Ct_list ; del Cterr_list
        # = = = Write averaged Ct as part of reporting
        out_fn = out_pref+'_averageCt.dat'
        fp = open(out_fn, 'w')
        for j in range(npts):
            print >> fp, dt[i], Ct[j], Cterr[j]
        print >> fp, '&'
        fp.close()

    sim_resid = legs
    num_vecs = len(dt)

    # = = = Fit simulated C(t) for each X-H vector with theoretical decomposition into a minimum number of time constants.
    #       This yields the fitting parameters required for the next step: the calculation of relaxations.
    # S2_list=[] ; taus_list=[] ; consts_list=[]

    # = = = Fit a gradually increasing number of parameters.
    num_comp=args.nc
    bUseSFast=(not args.bNoFast)
    dy_loc=[]
    chi_list=[] ; names_list=[] ; pars_list=[] ; errs_list=[] ; ymodel_list=[]

    out_fn=out_pref+'_fittedCt.dat'
    fp=open(out_fn, 'w')
    for i in range(num_vecs):
        print "...Running C(t)-fit for residue %i:" % sim_resid[i]
        if len(Cterr)>0:
            dy_loc=Cterr[i]
        if num_comp == -1:
            # Automatically find the best number of parameters
            if bUseSFast:
                #chi, names, pars, errs, ymodel = fitCt.findbest_LSstyle_fits(x[i], y[i], dy[i])
                chi, names, pars, errs, ymodel = fitCt.findbest_Expstyle_fits(dt[i], Ct[i], dy_loc,
                                             par_list=[2,3,5,7,9], threshold=1.0)
            else:
                chi, names, pars, errs, ymodel = fitCt.findbest_Expstyle_fits(dt[i], Ct[i], dy_loc,
                                             par_list=[2,4,6,8], threshold=1.0)
            num_pars=len(names)
        else:
            # Use a specified number of parameters
            if bUseSFast:
                num_pars=2*nc+1
            else:
                num_pars=2*nc
            chi, names, pars, errs, ymodel = fitCt.run_Expstyle_fits(dt[i], Ct[i], dy_loc, num_pars)

        S2, consts, taus, Sf = fitCt.sort_parameters(num_pars, pars)
        # Print header into the Ct model file
        print >> fp, '# Residue: %i ' % sim_resid[i]
        print >> fp, '# Chi-value: %g ' % chi
        if fmod( num_pars, 2 ) == 1:
            print >> fp, '# Param %s: %g +- %g' % ('S2_fast', Sf, 0.0)
        else:
            print >> fp, '# Param %s: %g +- %g' % ('S2_0', S2, 0.0)
        for j in range(num_pars):
            print >> fp, "# Param %s: %g +- %g" % (names[j], pars[j], errs[j])
        #Print the fitted Ct model into file
        print >> fp, "@s%d legend \"Res %d\"" % (i*2, sim_resid[i])
        for j in range(len(ymodel)):
            print >> fp, dt[i][j], ymodel[j]
        print >> fp, '&'
        for j in range(len(ymodel)):
            print >> fp, dt[i][j], Ct[i][j]
        print >> fp, '&'

        # S2, consts, taus, Sf = sort_parameters(num_pars, pars)
        # Parse parameters
        # Add elements to list.
        chi_list.append(chi)
        names_list.append(names)
        pars_list.append(pars)
        errs_list.append(errs)
        ymodel_list.append(ymodel)

    print " = = Completed C(t)-fits."


    #S2 *= zeta
    #consts = [ k*zeta for k in consts ]
    #S2 *= 0.89 ; consts = [ k*0.89 for k in consts ]
    #S2_list.append(S2) ; taus_list.append(taus) ; consts_list.append(consts)

    time_stop=time.clock()
    #Report time
    print "= = Finished. Total seconds elapsed: %g" % (time_stop - time_start)

sys.exit()
