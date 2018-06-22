import numpy as np
import argparse
import sys

def read_hydronmr_results( fileName ):
    """
    Goes through the HYDRONMR results (coorect as of version 7c2)
    and looks for key lines by the initial strings.
    Returns three items:
    - the rotational diffusion coefficients in Dx, Dy, and Dz re-sorted as necessary.
    - the eigenvector matrix, which is a left-multiplication rotation matrix to bring the PDB into PAF, AFAICT.
    - the name of the PDB file according to the input. HYDRONMR's fortran-code forces this to the in the same folder by default.
    """
    diffValues=np.zeros(3, dtype=np.float64)
    diffMatrix=np.zeros((3,3), dtype=np.float64)
    for line in open(fileName):
        l = line.split()
        if len(l) == 0:
            continue
        if l[0] == "Structural" and l[1] == "file:":
            pdbFile = l[-1]
            continue
        elif l[0] == "Dx":
            id=0
        elif l[0] == "Dy":
            id=1
        elif l[0] == "Dz":
            id=2
        else:
            continue
        diffValues[id] = l[1]
        diffMatrix[id] = ( l[-3],l[-2],l[-1] )

    # = = = Check if HYDRONMR has swapped Dx and Dy. If so, rotate z-axis by 90 degrees after transform to PAF
    if diffValues[0]> diffValues[1]:
        tmpMatrix=np.array([[0,-1,0],[1,0,0],[0,0,1]])
        print diffMatrix
        diffMatrix = np.matmul(tmpMatrix,diffMatrix)
        print diffMatrix
        tmp=diffValues[1]
        diffValues[1]=diffValues[0]
        diffValues[0]=tmp

    for i in range(3):
        print np.linalg.norm(diffMatrix[i])
        diffMatrix[i]=diffMatrix[i]/np.linalg.norm(diffMatrix[i])

    return diffValues, diffMatrix, pdbFile

def translate_D( D ):
    """
    Change from Dx,, Dy, Dz to D_iso, D_aniso, D_rhomb.
    """
    outD = np.zeros_like( D )
    outD[0] = np.mean(D)
    outD[1] = 2*D[2]/(D[1]+D[0])
    outD[2] = 3*(D[1]-D[0])/(2*D[2]-D[1]-D[0])
    return outD

parser = argparse.ArgumentParser(description='Extracts the rotationsl diffusion tensor from HYDRONMR results.'
                                    'Optionally, rotates the HYDRONMR input PDB file into the PAF frame.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-f', '--infn', type=str, default='output.res',
                        help='HYDRONMR output file')
parser.add_argument('--rotate', action='store_true',
                        help='Rotate the PDB file used by HYDRONMR into its Principal-Axis Frame (PAF).')
parser.add_argument('-t', dest='timeFactor', type=float, default=1e-12,
                        help='Convert the input time-units from s^-1 to ps^-1 by default. Change for some other time unit.')
parser.add_argument('-o', '--outPDB', type=str, default='rotated.pdb',
                        help='Filename of the rotated PDB file.')

args = parser.parse_args()

hydroFile=args.infn
outputPDB=args.outPDB
bRotate = args.rotate
timeFactor=args.timeFactor

D, mat, pdbFile = read_hydronmr_results( hydroFile )
DD = D*timeFactor
Dp = translate_D( DD )
print "= = = Read the diffusion tensor value (s^-1): %g %g %g" % (D[0], D[1], D[2])
print "= = = Translated into ps^-1: %g %g %g" % ( DD[0], DD[1], DD[2] )
print "= = = Translated into axisymmetric-expansion: %g %g %g" % (Dp[0], Dp[1], Dp[2])
print "= = = Rotation matrix:"
for i in range(mat.shape[0]):
    print "%16g %16g %16g" % (mat[i,0], mat[i,1], mat[i,2])

if not bRotate:
    sys.exit()

import mdtraj as md
mol = md.load(pdbFile)
print "= = = Loaded %s, rotating..." % pdbFile
# = = = Rotate molecule in place
cog = np.average(mol.xyz[0],axis=0)
rotatedXYZ = np.zeros_like(mol.xyz)
rotatedXYZ[0] = np.matmul( mol.xyz[0]-cog, mat.T ) + cog
mol.xyz = rotatedXYZ
mol.save_pdb(outputPDB)
print "= = = Done."



