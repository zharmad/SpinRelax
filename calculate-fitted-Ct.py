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
#from scipy.stats import sem

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
        print( "    .... Note: The 'name H' selects %i atoms, 'name HN' selects %i atoms, and 'name HA' selects %i atoms." % (t1, t2, t3) )
    if len(indX) == 0:
        bError=True
        t1 = mol.topology.select('name N')
        t2 = mol.topology.select('name NT')
        print( "    .... Note: The 'name N' selects %i atoms, and 'name NT' selects %i atoms." % (t1, t2) )
    if bError:
        sys.exit(1)
    return indH, indX

def sanity_check_two_list(listA, listB, string, bVerbose=False):
    if not np.all( np.equal(listA, listB) ):
        print( "= = ERROR: Sanity checked failed for %s!" % string )
        if bVerbose:
            print( listA )
            print( listB )
        sys.exit(1)
    return

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
                             'If multiple files are given, each corresponding C(t) will be averaged '
                             'before a single fitting is calculated. All time entries need to be identical here.'
                             'Note: The run-all.bash workflow will generate a single C(t) file based on all trajectories, '
                             'so this multifile option is generally not used.')
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

    time_start=time.time()

    args = parser.parse_args()
    in_file_list = args.in_Ct_fn
    out_pref=args.out_pref
    nc=args.nc
    #zeta=args.zeta
    bHaveDy = False

    # = = = Prepare the handling class
    autoCorrs = fitCt.autoCorrelations()

    # = = = Read C(t), and averge if more than one
    bError=True
    num_files = len(in_file_list)
    print( "= = = Found %d input C(t) files." % num_files )
    if (num_files == 1):
        # Note return array of (nLegs, nTime )
        legs, dt, Ct, Cterr = gs.load_sxydylist(in_file_list[0], 'legend')
        legs = [ int(x) for x in legs ]
        if len(Cterr)==0:
            Cterr=None
            bError=False
    else:
        print( "    ...will perform averaging to obtain averaged C(t)." )
        dtFirst=None ; legsFirst=None; Ct_list=[] ; Cterr_list=[]
        for ind in range(num_files):
            legs, dt, Ct, Cterr = gs.load_sxydylist(in_file_list[ind], 'legend')
            legs = [ int(x) for x in legs ]
            if not dtFirst is None and np.all(np.equal(dt, dt_prev)) and np.all(np.equal(legs, legsFirst)):
                print("= = = ERROR: time and legend entries are not identical between files! Last file read: %s" % in_file_list[ind], file=fp)
                sys.exit(1)
            else:
                dtFirst=dt ; legsFirst = legs
            Ct_list.append(Ct) ; Cterr_list.append(Cterr)
        # = = = Ct_list is a list of 2D arrays.
        # = = = Perform grand average over individual replicates. Assuming equal weights (dangerous!)
        Ct = np.mean(np.array(Ct_list, dtype=float), axis=0)
        del Ct_list
        if len(Cterr_list[0]) == 0:
            # = = = No error bars on input curves, used standard-deviation as uncertainties.
            Cterr = np.std(Ct_list, axis=0)
        else:
            # = = = Has error bars on input curves, calculate
            shape=Ct.shape
            Cterr_list = np.array(Cterr_list, dtype=float)
            Cterr = np.zeros(shape)
            for i in range(shape[0]):
                for j in range(shape[1]):
                    Cterr[i,j] = gm.simple_total_mean_square(Ct_list[:,i,j], Cterr_list[:,i,j])
        del Cterr_list
        # = = = Write averaged Ct as part of reporting
        out_fn = out_pref+'_averageCt.dat'
        fp = open(out_fn, 'w')
        for i in range(len(legs)):
            for j in range(npts):
                print( dt[i][j], Ct[i][j], Cterr[i][j], file=fp )
        print( '&', file=fp )
        fp.close()

    autoCorrs.import_target_array(keys=legs, DeltaT=dt, Decay=Ct , dDecay=Cterr)
    del dt, Ct, Cterr
    # = = = Fit simulated C(t) for each X-H vector with theoretical decomposition into a minimum number of time constants.
    #       This yields the fitting parameters required for the next step: the calculation of relaxations.
    # S2_list=[] ; taus_list=[] ; consts_list=[]

    # = = = Fit a gradually increasing number of parameters.
    num_comp=args.nc
    bUseSFast=(not args.bNoFast)
    dy_loc=[]

    out_fn=out_pref+'_fittedCt.dat'
    fp=open(out_fn, 'w')
    for k in autoCorrs.DeltaT.keys():
        obj = autoCorrs.add_model( k )
        print( "...Running C(t)-fit for residue %i:" % obj.name )
        if num_comp == -1:
            # Automatically find the best number of parameters
            if bUseSFast:
                listDoGs=[2,3,5,7,9]
            else:
                listDoGs=[2,4,6,8]
            obj.optimised_curve_fitting( autoCorrs.DeltaT[k], autoCorrs.Decay[k], autoCorrs.dDecay[k], listDoG=listDoGs, chiSqThreshold=0.5 )
        else:
            # Use a specified number of parameters
            if bUseSFast:
                obj.set_nParams( 2*nc+1 )
            else:
                obj.set_nParams( 2*nc )
            obj.conduct_curve_fitting( autoCorrs.DeltaT[k], autoCorrs.Decay[k], autoCorrs.dDecay[k], bReInitialise=True)

    autoCorrs.export(fileName=out_fn, style='xmgrace')
    print( " = = Completed C(t)-fits." )

    #S2 *= zeta
    #consts = [ k*zeta for k in consts ]
    #S2 *= 0.89 ; consts = [ k*0.89 for k in consts ]
    #S2_list.append(S2) ; taus_list.append(taus) ; consts_list.append(consts)

    time_stop=time.time()
    #Report time
    print( "= = Finished. Total seconds elapsed: %g" % (time_stop - time_start) )

sys.exit()
