import mdtraj as md
import numpy as np
import sys, argparse
import transforms3d_supplement as qs
from re import split as regexp_split

def rotate_vectors(xyz, qRot, bAroundCOM=False):
    """
    Following the conventions of MdTraj, xyz takes the shape of (frame, atoms, xyz).
    """
    if bAroundCOM:
        com = np.mean(xyz, axis=1)
        tmp = qs.rotate_vector_simd(xyz-com[:,np.newaxis,:], qRot, axis=-1)
        return tmp + com[:,np.newaxis,:]
    else:
        return qs.rotate_vector_simd(xyz, qRot, axis=-1)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='This script conducts a simple quaternion rotation on an input PDB file.',
                                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-f', type=str, dest='fileInput', default=None, required=True,
            help='Input coordinate file acceptable to MDTraj. This will be loaded as its topology input.')
    parser.add_argument('-q', type=str, dest='qRot', default='1,0,0,0', required=True,
            help='Rotation quaternion, input as a string. Can be comma-separated like w,x,y,z or space-separated like "w x y z"')
    parser.add_argument('-o', type=str, dest='fileOutput', default='rotated.pdb',
            help='Output file. The file suffix determines the file type.')

    args = parser.parse_args()

    qRot = np.array([ float(x) for x in regexp.split('[, ]', args.qRot) ])
    mol = md.load(args.fileInput)
    print( "= = = Loaded single reference file: %s" % (args.fileInput) )

    mol.xyz = rotate_vectors(mol.xyz, qRot, True)
    mol.save(args.fileOutput)
    print( "= = = Done. Outfile file %s has been written." % (args.fileOutput) )
