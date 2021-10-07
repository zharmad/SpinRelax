import numpy as np
import argparse
import sys

def reorder_axes( D, mat ):
    """
    Sort the eigenvector matrix so that the eigenvalues D come in ascending order.
    """
    if D[0]<=D[1]<=D[2]:
        return D, mat
    if D[0]<=D[2]<D[1]:
        print( "    ...rotate X by 90-degrees." )
        rot=np.array([[ 1., 0., 0.],\
                      [ 0., 0.,-1.],\
                      [ 0., 1., 0.]])
        return np.array([D[0],D[2],D[1]]), np.matmul(rot,mat)
    if D[2]<=D[1]<=D[0]:
        print( "    ...rotate Y by 90-degrees." )
        rot=np.array([[ 0., 0., 1.],\
                      [ 0., 1., 0.],\
                      [-1., 0., 0.]])
        return np.array([D[2],D[1],D[0]]), np.matmul(rot,mat)
    if D[1]<D[0]<D[2]:
        print( "    ...rotate Z by 90-degrees." )
        rot=np.array([[ 0.,-1., 0.],\
                      [ 1., 0., 0.],\
                      [ 0., 0., 1.]])
        return np.array([D[1],D[0],D[2]]), np.matmul(rot,mat)
    if D[1]<D[2]<D[0]:
        print( "    ...forward permutation." )
        rot=np.array([[ 0., 0., 1.],\
                      [ 1., 0., 0.],\
                      [ 0., 1., 0.]])
        return np.array([D[1],D[2],D[0]]), np.matmul(rot,mat)
    if D[2]<D[0]<D[1]:
        print( "    ...backward permutation." )
        rot=np.array([[ 0., 1., 0.],\
                      [ 0., 0., 1.],\
                      [ 1., 0., 0.]])
        return np.array([D[2],D[0],D[1]]), np.matmul(rot,mat)
    # = = = ERROR section
    print( "= = = ERROR in reorder axes, the ordering if statements somehow did not accoutn for the following combination!", file=sys.stderr )
    print( D, file=sys.stderr )
    sys.exit(1)

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

    print( "= = = HYDRONMR diffusion tensor value (s^-1): %g %g %g" % (diffValues[0], diffValues[1], diffValues[2]) )
    print( "= = = HYDRONMR rotation matrix:" )
    for i in range(3):
        print( "%16g %16g %16g" % (diffMatrix[i,0], diffMatrix[i,1], diffMatrix[i,2]) )


    # = = = Check if HYDRONMR has swapped Dx and Dy. If so, rotate z-axis by 90 degrees after transform to PAF
    diffValues, diffMatrix = reorder_axes( diffValues, diffMatrix)

    for i in range(3):
        print( np.linalg.norm(diffMatrix[i]) )
        diffMatrix[i]=diffMatrix[i]/np.linalg.norm(diffMatrix[i])

    return diffValues, diffMatrix, pdbFile

def translate_D( D ):
    """
    Change from Dx, Dy, Dz to D_iso, D_aniso, D_rhomb.
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
parser.add_argument('--pdb', type=str, default=None,
                        help='Rotate this PDB file instead of the one used by HYDRONMR.')
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
if not args.pdb is None:
    pdbFile = args.pdb

DD = D*timeFactor
Dp1 = translate_D( DD )
Dp2 = translate_D( DD[::-1] )
print( "= = = Read the diffusion tensor value (s^-1): %g %g %g" % (D[0], D[1], D[2]) )
print( "= = = Translated into ps^-1: %g %g %g" % ( DD[0], DD[1], DD[2] ) )
print( "= = = Translated into axisymmetric-expansion  (long-axis): %g %g %g" % (Dp1[0], Dp1[1], Dp1[2]) )
print( "= = = Translated into axisymmetric-expansion (short-axis): %g %g %g" % (Dp2[0], Dp2[1], Dp2[2]) )
print( "= = = Rotation matrix:" )
for i in range(mat.shape[0]):
    print( "%16g %16g %16g" % (mat[i,0], mat[i,1], mat[i,2]) )

try:
    import transforms3d.quaternions as qops
    print( "= = = Equivalent quaternion:" )
    q = qops.mat2quat( mat )
    print( "%g %g %g %g" % (q[0],q[1],q[2],q[3]) )
except:
    # = = = Do nothing.
    print( "= = = transforms3d not available. Will not report equivalent quaternion." )

if not bRotate:
    sys.exit()

# = = = Output D values to file.
fp = open(pdbFile[:-4]+'.Dxyz','w')
print( "%g %g %g" % ( DD[0], DD[1], DD[2] ), file=fp )
fp.close()

fp = open(pdbFile[:-4]+'.Dsymm','w')
if Dp1[2]<=1:
    print( "%g %g %g" % (Dp1[0], Dp1[1], Dp1[2]), file=fp )
else:
    print( "%g %g %g" % (Dp2[0], Dp2[1], Dp2[2]), file=fp )
fp.close()

import mdtraj as md
mol = md.load(pdbFile)
print( "= = = Loaded %s, rotating..." % pdbFile )
# = = = Rotate molecule in place
cog = np.average(mol.xyz[0],axis=0)
rotatedXYZ = np.zeros_like(mol.xyz)
rotatedXYZ[0] = np.matmul( mol.xyz[0]-cog, mat.T ) + cog
mol.xyz = rotatedXYZ
mol.save_pdb(outputPDB)
print( "= = = Done." )



