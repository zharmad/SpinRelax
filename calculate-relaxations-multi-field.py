#from math import *
import sys, os, argparse, time
from re import split as regexp_split
import numpy as np
#import numpy.ma as ma
import general_scripts as gs
#import general_maths as gm
#import transforms3d.quaternions as qops
#import transforms3d_supplement as qs
import fitting_Ct_functions as fitCt
import spectral_densities as sd

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
#    parser.add_argument('--ROTDIF', dest='inputFileROTDIF', type=str, default=None,
#                        help='As an alternative to the above, a ROTDIF-format input file can be given.')
    parser.add_argument('-o', '--outpref', type=str, dest='out_pref', default='out',
                        help='Output file prefix.')
    parser.add_argument('-f', '--infn', type=str, dest='in_Ct_fn', required=True,
                        help='Read a formatted file with fitted C_internal(t), taking from it the parameters.')                       
    parser.add_argument('--refpdb', type=str, dest='refPDBFile', default=None,
                        help='Reference PDB file to compute axisymmetric rotational diffusion using X-H bond vectors wound within. '
                             'This assumes that the PDB file is already in the principal-axis frame aligned with the diffusion tensor.')
    parser.add_argument('--distfn', type=str, dest='distfn', default=None,
                        help='Vector orientation distribution of the X-H dipole in spherical-polar coordinates. '
                             'This is used to compute axisymmetric rotational diffusion and overrules '
                             'the reference PDB argument. Vectors assumed to be in the principal axes frame.')
#    parser.add_argument('--shiftres', type=int, default=0,
#                        help='Shift the MD residue indices, e.g., to match the experiment.')
#    parser.add_argument('--tu', '--time_units', type=str, dest='time_unit', default='ps',
#                        help='Time units of the autocorrelation file.')
    parser.add_argument('--tau', type=float, dest='tau', default=None,
                        help='Isotropic relaxation time constant. Overwritten by Diffusion tensor values when given.')
    parser.add_argument('--aniso', type=float, dest='aniso', default=None,
                        help='Diffusion anisotropy (prolate/oblate). Overwritten by Diffusion tensor values when Dpar and Dperp values are given.')
    parser.add_argument('-D', '--DTensor', type=str, dest='D', default=None,
                        help='The Diffusion tensor. Single values are parsed as isotropic component. '
                             'Two values are parsed as  Dpar,Dperp. '
                             'Three values are directly parsed as the diffusion tensor, although this is not currently accepted. '
                             'Entries are either comma-separated or space separated in a quote.\n'
                             'Note: In axisymmetric forms, when Daniso < 1 the unique axis is considered to point along x, and '
                             'when Daniso > 1 the unique axis is considered to point along z.')
    parser.add_argument('--zeta', type=float, default=0.890023,
                        help='Input optional manual zeta factor to scale-down '
                             'the S^2 of MD-based derivations by zero-point vibrations known in QM. '
                             'This is by convention 0.890023 (1.02/1.04)^6 - see Trbovic et al., Proteins, 2008 and Case, J Biomol NMR, 1999.')
    parser.add_argument('--csa', type=str, default=None,
                        help='Input manual average CSA value, if different from the assumed value, e.g. -170 ppm for 15N. '
                             'Residue-specific variations is set if a file name is given. This file should '
                             'specify on each line the residue index and then its respective CSA value.')
    parser.add_argument('--opt', '--fit', type=str, dest='listOptParams', default=None,
                        help='Perform optimisation against all given experimental data, over the following possible parameters %s:\n'
                             '  - Diso, isotropic tumbling rate of the domain,\n'
                             '  - Daniso, axisymmetric tumbling anisotropy,\n'
                             '  - zeta, QM zero-point vibration contribution to local autocorrelation,\n'
                             '  - CSA, mean chemical shift anisotropy of the heavy nuclei (nucleiA),\n'
                             '  - rsCSA, residue-specific chemical-shift anisotropies.\n'
                             'Set these using a single comma-separated string as the argument. '
                             'The order given will be the order used inthe optimisation loop\n' \
                             % (sd.spinRelaxationExperiments.listAllowedOptimisationVariables) )
    parser.add_argument('--cycles', type=int, default=10,
                        help='For the residue-specific CSA fitting algorithm, do a maximum of N cycles between the global Diso versus the local CSA fits. '
                             'Each cycle begins by optimising Diso over all residues, then optimising each CSA separately against the new global value.\n'
                             'Adjust in case algorithm fails to find convergence within 2 cycles.')
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
        
    nResidSim = localCtModel.nModels
    residSim  = [ int(k) for k in localCtModel.model.keys() ]
                        
    # = = = Parse and set up global tumbling model
    globalRotDif = parse_rotdif_params(args.D, args.tau, args.aniso)
    
    if not args.distfn is None:
        globalRotDif.import_frame_vectors(args.distfn)
    elif not args.refPDBFile is None:
        globalRotDif.import_frame_vectors_pdb(args.refPDBFile)
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
        objExpts.set_global_zeta( args.zeta )
    #objExpts.report()
    
    # = = = This mapping is necessary to match simulation names to experimental names,
    #       as not all peaks are resolved or assigned in a given experimental conditiosn.
    objExpts.map_experiment_peaknames_to_models()
    objExpts.report_maps()
    
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
            # = = = Temporary work-around
            residCSA, CSAvaluesArray = gs.load_xy( args.csa )
            residCSA=[ str(int(x)) for x in residCSA ]
            print( "= = = Using input CSA values from file %s - please ensure that the names match those found in C(t) models." % args.csa )
            if np.fabs(CSAvaluesArray[0]) > 1.0:
                print( "= = = NOTE: the first value is > 1.0, so assume a necessary conversion to ppm." )
                CSAvaluesArray *= 1e-6
            objExpts.initialise_CSA_array(residCSA, CSAvaluesArray)
        elif bIsNumeric:
            print( "= = = Using user-input CSA value: %g" % tmp )
            if np.fabs(tmp)>1.0:
                print( "= = = NOTE: this value is > 1.0, so assume a necessary conversion to ppm." )
                tmp*=1e-6
            CSAvaluesArray = np.repeat( tmp, objExpts.localCtModels.nModels )
            objExpts.initialise_CSA_array(objExpts.localCtModels.get_names(), CSAvaluesArray)
        else:
            print( "= = = ERROR at parsing the --csa argument!", file=sys.stderr )
            sys.exit(1)
                    
    # = = = No optimisation. Simply print the direct prediction for each experiment. = = =
    if args.listOptParams is None:
        objExpts.eval_all(bVerbose=True)
        objExpts.export_xvg(args.out_pref, bIncludeExpt=False)
        
        time_stop=time.time()
        print( "= = Finished. Total seconds elapsed: %g" % (time_stop - time_start) )
        sys.exit()
    # = = = End No optimisation. = = =

    # = = = evaluation section. Determine what is to be done wioth the data.
    listOpt = args.listOptParams.split(',')    
    objExpts.parse_optimisation_params(listOpt)
    print("= = = Parsed optimiser input %s." % args.listOptParams )
    print("    ... conducting global optimisations over parameters %s ..." % (objExpts.listUpdateVariables))
    if objExpts.bDoLocalOpt:
        print("    ... conducting local optimisations over residue-specific CSA...")
    chisq = objExpts.perform_optimisation(maxCycles=nRefinementCycles,tol=refinementTolerance)
    print("= = = Optimisation complete. Final chi-value: %g" % np.sqrt(chisq) )
    
    objExpts.export_xvg(args.out_pref, bIncludeExpt=True)
    if objExpts.bDoLocalOpt and objExpts.bOptCompleted:
        names       = objExpts.localCtModels.get_names() 
        optCSAArray = objExpts.get_first_csa()
        fp = open( args.out_pref+'_CSA_opt.dat', 'w')
        for x, y in zip(names, optCSAArray):
            print("%s %g" % (x,y), file=fp)
        fp.close()
    
    time_stop=time.time()
    print( "= = Finished. Total seconds elapsed: %g" % (time_stop - time_start) )
    sys.exit()
