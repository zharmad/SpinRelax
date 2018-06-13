#!/bin/python
# This files supplements the transforms3D module
# by augmenting it with the following operations:
#    quaternion to rotate v1 to v2.
#    quaternion to transform between the coordinate axes.

from math import acos, sin
import transforms3d.quaternions as qops
import numpy as np
import random

def decompose_quat(q, axis=-1, bReshape=False):
    """
    Dynamic decomposition of quaternions into their w-component and v-component while preserving input axes.
    If bReshape is True, then the shape of q_w is augmented to have the same dimensionality as before
    with a value of 1 for broadcasting purposes.
    The reshaping apparently does not survive the function return....!
    """
    q=np.array(q)
    if axis==-1:
        q_w = q[...,0]
        q_v = q[...,1:4]
    elif axis==0:
        q_w = q[0,...]
        q_v = q[1:4,...]
    else:
        print sys.stderr, "= = = ERROR: decompose_quat does not support arbitrary axis definitions."
        sys.exit(1)

    if bReshape and len(q_w.shape)>1:
        newShape=list(q_w.shape)
        if axis==-1:
            newShape.append(1)
        elif axis==0:
            newShape.insert(0,1)
        q_w = q_w.reshape(newShape)

    return q_w, q_v

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

def axangle2quat_simd(ax, th, bNormalised=False):
    """
    Local Version that can take additional dimension data with the axis at the end. Base maths from transforms3d.quaternions:
    t2 = theta / 2.0
    st2 = math.sin(t2)
    return np.concatenate(([math.cos(t2)],
                           vector * st2))
    """
    if not bNormalised:
        ax = vecnorm_NDarray(ax)
    half = th / 2.0
    sine = np.sin(half)
    if len(ax.shape)>1:
        return np.concatenate( ( np.cos(half)[...,None], np.multiply(ax,sine[...,None]) ), axis=-1 )
    else:
        return np.concatenate( ([np.cos(half)],ax*sine) )

def quat_v1v2(v1, v2):
    """
    Return the minimum-angle quaternion that rotates v1 to v2.
    Non-SIMD version for clarity of maths.
    """
    th=acos(np.dot(v1,v2))
    ax=np.cross(v1,v2)
    if all( np.isnan(ax) ):
        # = = = This happens when the two vectors are identical
        return qops.qeye()
    else:
        # = = = Do normalisation within the next function
        return qops.axangle2quat(ax, th)

def quat_v1v2_simd(v1, v2, bNormalised=False):
    """
    Return the minimum-angle quaternion that rotates v1 so as to align it with v2.
    In this SIMD version, v1 and v2 can be single 3-vectors or 2D arrays with shape (N,3)
    Values clipped to [-1,1] in-case of a float error.
    """
    v1=np.array(v1)
    v2=np.array(v2)

    dim1=len(v1.shape)
    dim2=len(v2.shape)
    if not bNormalised:
        v1=vecnorm_NDarray(v1)
        v2=vecnorm_NDarray(v2)
    if dim1==2 and dim2==2:
        #th=np.arccos( np.clip( np.diag( np.matmul(v1,v2.T) ), -1.0, 1.0 ) )
        th=np.arccos(np.clip( np.einsum('ij,ij->i', v1,v2),-1.0,1.0) )
    else:
        th=np.arccos( np.clip( np.dot(v1,v2.T), -1.0, 1.0 ) )
    ax = np.cross(v1,v2)
    #ax = np.nan_to_num( np.cross(v1,v2) )
    return axangle2quat_simd(ax, th, bNormalised=False)

# This function returns the quaternions
# required to transform the given axes vectors
# to match the coordinate frame vectors, this is a COORDINATE transform.
# It is equivalent to a FRAME transform from the coordinate frame
# to the given frame vectors.
# Note that it does not test for orthogonality
# of length of the given axes.
# So, may give rubbish if you don't watch out.
# NB: actually calculates the rotation of vectors to
# the coordinate axes, which is a same rotation of
# the frame in the opposite direction.
def quat_frame_transform(axes):
    ref=np.array( ((1,0,0),(0,1,0),(0,0,1)) )
    # Define two rotations, first to match z with Z
    # then to match x/y with X/Y
    q1=quat_v1v2(axes[2],ref[2])
    arot=[ qops.rotate_vector(axes[i],q1) for i in range(3)]
    q2a=quat_v1v2(arot[0],ref[0])
    #Weak test for orthogonality here. Doesn't work since the second axes can be +-Y
    #q2b=quat_v1v2(arot[1],ref[1])
    #if qops.nearly_equivalent(q2a, q2b) == False:
    #    print "= = = ERROR in quat_frame_transform, found disagreement between "
    #    print "      second rotations to bring x/y to X/Y!"
    #    print q1, q2a, q2b
    #    return -1
    return qops.qmult(q2a,q1)

# Returns the minimum version that has the smallest cosine angle
# to the positive or negative axis.
def quat_frame_transform_min(axes):
    ref=np.array( ((1,0,0),(0,1,0),(0,0,1)) )

    q1a=quat_v1v2(axes[2],(0,0, 1))
    q1b=quat_v1v2(axes[2],(0,0,-1))
    q1 = q1a if q1a[0]>q1b[0] else q1b
    arot=[ qops.rotate_vector(axes[i],q1) for i in range(3)]

    q2a=quat_v1v2(arot[0],( 1,0,0))
    q2b=quat_v1v2(arot[0],(-1,0,0))
    q2 = q2a if q2a[0]>q2b[0] else q2b

    return qops.qmult(q2,q1)

# The opposite function to bring an frame back to the coordinate axes.
# Although the elements of the input axes are given in the final coordinate axes... -_-
# Note that the quaternion rotation is identical in the two frames.
def quat_frame_transform_inv(axes):
    return qops.qconjugate(quat_frame_transform(axes))

def quat_mult(q1,q2):
    q=np.zeros_like(q1)
    q[0]  = q1[0]*q2[0] - np.einsum('...i,...i',q1[1:4],q2[1:4])
    q[1:4]= q1[0]*q2[1:4] + q2[0]*q1[1:4] + np.cross(q1[1:4],q2[1:4])
    return q

def quat_mult_simd(q1, q2):
    """
    SIMD-version of transforms3d.quaternions.qmult using vector operations.
    when at least one of these is N-dimensional, it expects that q1 is the N-dimensional quantity.
    The axis is assumed to be the last one.
    To do this faster, we'll need a C-code ufunc.
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 + y1*w2 + z1*x2 - x1*z2
    z = w1*z2 + z1*w2 + x1*y2 - y1*x2
    return np.array([w, x, y, z])
    """
    #w1, v1 = decompose_quat(q1)
    #w2, v2 = decompose_quat(q2)
    #del v1 ; del v2
    out=np.zeros_like(q1)
    out[...,0]  = q1[...,0]*q2[...,0] - np.einsum('...i,...i',q1[...,1:4],q2[...,1:4])
    out[...,1:4]= q1[...,0,None]*q2[...,1:4] + q2[...,0,None]*q1[...,1:4] + np.cross(q1[...,1:4],q2[...,1:4])
    return out

def quat_invert(q):
    return q*[1.0,-1.0,-1.0,-1.0]

def quat_negate(q):
    """
    return -q.
    """
    return q*-1.0

def quat_normalise(q):
    """
    Return q-hat, which has unit length. Use vecnorm_NDarray for q matrices with components as the last axis.
    """
    return q/np.linalg.norm(q)

def quat_rand(n=1, dtype=np.float64, bReduce=True, qref=(1,0,0,0)):
    """
    Return N randomised quaternions in base float64. Calls random.uniform 3*N and converts to quaternion form.
    See Shoemake, K. "Uniform random rotations", 1992.
    Returns in dimensions (n, 4) for n>1 .
    """
    r = np.reshape( [ random.uniform(0,1) for x in range(3*n) ], (3,n) )
    q = np.zeros( (4,n), dtype=dtype)
    q[0] = np.sqrt(1.0-r[0])*np.sin(2.0*np.pi*r[1])
    q[1] = np.sqrt(1.0-r[0])*np.cos(2.0*np.pi*r[1])
    q[2] = np.sqrt(r[0])*np.sin(2.0*np.pi*r[2])
    q[3] = np.sqrt(r[0])*np.cos(2.0*np.pi*r[2])

    if bReduce:
        qref=np.array(qref)
        return quat_reduce_simd(q.T, qref)
    else:
        return q.T

def quat_reduce_simd(q, qref=(1,0,0,0),axis=-1):
    """
    Return the closer image of q or -q to a reference quaternion qref.
    The default is (1,0,0,0) to restrict rotations to be less than 180 degrees.
    """
    if axis==-1:
        sgn=np.sign( np.einsum('...i,i',q,qref) )
        sgn[sgn==0]=1.0
        return q*sgn[:,None]
    elif axis==0:
        sgn=np.sign( np.einsum('i...,i',q,qref) )
        sgn[sgn==0]=1.0
        return q*sgn[None,:]
    else:
        print sys.stderr, "= = = ERROR: quat_reduce_simd does not support arbitrary axis definitions!"
        sys.exit(1)

def quat_reduce(q, qref=(1,0,0,0)):
    """
    Return the closer image of q or -q to a reference quaternion qref.
    The defualt is (1,0,0,0) to restrict rotations to be less than 180 degrees.
    """
    m=np.dot(q,qref)
    if m>=0:
        return q
    else:
        return -1.0*q

def quat_sub(q1, q2):
    """
    Returns q1 * cong(q2)
    """
    return qops.qmult(q2,qops.qconjugate(q1))

def quat_slerp(q1, q2, r):
    """
    Calculate the interpolation between two quaternions, with the ratio r, where q=q1 @ r=0, and q=q2 @ r=1
    WARNING: This does NOT return the result of evenly spaced quaternion movements, somehow.
    """
    qdiff=quat_sub(q1, q2)
    th=2*acos(qdiff[0])

    return quat_normalise( np.multiply(sin((1-r)*th)/sin(th),q1) + np.multiply(sin(r*th)/sin(th),q2) )

def rotate_vector(v,q, bNormalised=False):
    if not bNormalised:
        q = vecnorm_NDarray(q)
    a = np.cross(q[1:4], v) + q[0]*v
    b = np.cross(q[1:4], a)
    return b+b+v

def rotate_vector_simd(v, q, axis=-1, bNormalised=False):
    """
    Alternative formulation of quaternion multiplication on a set of vectors that I hope should be quicker.
    Uses numpy broadcasting. So allowed combination are 1 vector/ ND-vectors with 1 quaternion/ND-quaternions,
    as long a ND are the same shape in the non-vector components.

    For all rotations, q must be normalised to a unit quaternion.

    The axis vector components must be either the first or last axis to satisfy broadcasting.
    i.e. v[...,3] & q[..,4] or alternatively v[3,...] & q[4,...]
    """

    if not bNormalised:
        q = vecnorm_NDarray(q)
    v=np.array(v)
    q_w, q_v = decompose_quat(q, axis=axis, bReshape=True)
    if axis==-1:
        a = np.cross(q_v, v, axisa=axis, axisb=axis) + q_w[...,None]*v
    elif axis==0:
        a = np.cross(q_v, v, axisa=axis, axisb=axis) + q_w[None,...]*v
    else:
        print sys.stderr, "= = = ERROR: rotate_vector_simd does not support arbitrary axis definitions."
        sys.exit(1)

    b = np.cross(q_v, a, axisa=axis, axisb=axis)
    return b+b+v

