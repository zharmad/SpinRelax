import sys, argparse
import numpy as np
import general_scripts as gs
import spectral_densities as sd

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Read an experimental file containing R1R2NOE data and output the R1/R2 ratio, '
                                     'also known as rho.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-f', type=str, dest='exptFile',
                        help='Experimental values of R1, R2, and NOE in a 4- or 7-column format.'
                             '4-column data is assumed to be ResID/R1/R2/NOE. 7-column data is assumed to also have errors.'
                             'Giving this file currently will compute the rho equivalent and quit, unless --opt is also given.')
    parser.add_argument('-o', type=str, dest='outputFile', default=None,
                        help='Name of the output file. Defaults to out_expRho.dat')
    parser.add_argument('-n', '--nuclei', type=str, dest='nuclei', default='NH',
                        help='Type of nuclei measured by the NMR spectrometer. Determines the gamma constants used.')
#    parser.add_argument('--tu', '--time_units', type=str, dest='time_unit', default='ps',
#                        help='Time units of the autocorrelation file.')

    args = parser.parse_args()

    if args.outputFile is None:
        outputFile = "out_expRho.dat"
    else:
        outputFile = args.outputFile
        
    # Set up relaxation parameters.
    nuclei_pair  = args.nuclei
#    timeUnit = args.time_unit
    relax_obj = sd.relaxationModel(nuclei_pair, 600.133e6 )
#    relax_obj.set_time_unit(timeUnit)
    print( "= = = Gamma values: (X) %g , (H) %g rad s^-1 T^-1" % (relax_obj.gX.gamma, relax_obj.gH.gamma) )

    exp_resid, expblock = gs.load_xys(args.exptFile)
    nres = len(exp_resid)
    ny   = expblock.shape[1]
    rho = np.zeros(nres, dtype=float)
    if ny == 6:
        expblock = expblock.reshape( (nres,3,2) )
    elif ny != 3:
        print( "= = = ERROR: The column format of the experimental relaxation file is not recognised!", file=sys.stderr )
        sys.exit(1)

    rho = np.zeros(nres, dtype=float)
    if ny==6:
        for i in range(nres):
            rho[i]=relax_obj.calculate_rho_from_relaxation(expblock[i,:,0])
    else:
        for i in range(nres):
            rho[i]=relax_obj.calculate_rho_from_relaxation(expblock[i])
    
    gs.print_xy(outputFile, exp_resid, rho)
