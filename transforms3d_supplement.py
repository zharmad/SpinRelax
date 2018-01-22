#!/bin/python
# This files supplements the transforms3D module
# by augmenting it with the following operations:
#    quaternion to rotate v1 to v2.
#    quaternion to transform between the coordinate axes.

from math import acos, sin
import transforms3d.quaternions as qops
import numpy as np

def quat_v1v2(v1, v2):
    """
    Return the minimum-angle quaternion that rotates v1 to v2.
    """
    th=acos(np.dot(v1,v2))
    ax=np.cross(v1,v2)
    #Does not assume normalisation
    return qops.axangle2quat(ax, th)

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

def quat_negate(q):
    """
    return -q.
    """
    return q*-1.0

def quat_normalise(q):
    """
    Return q-hat, which has unit length.
    """
    return q/np.linalg.norm(q)

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
    WARNING: This does NOT return the result or evenly spaced quaternion movements, somehow.
    """
    qdiff=quat_sub(q1, q2)
    th=2*acos(qdiff[0])

    return quat_normalise( np.multiply(sin((1-r)*th)/sin(th),q1) + np.multiply(sin(r*th)/sin(th),q2) )

def rotate_vector_simd(v, q, axis=-1):
    """
    Alternative formulation of quaternion multiplication on a set of vectors that I hope should be quicker.
    v is given as a array of vectors with the last axis as its x/y/z cartesion component.
    q = (q_w, q_x, q_y, q_z)
    """
    v=np.array(v)
    sh=v.shape
    q_v = np.array(q[1:4])
    a = np.cross(q_v, v) + np.multiply(q[0],v)
    b = np.cross(q_v, a)
    return b+b+v
    #v=np.array(v)
    #q_v = np.array(q[1:4])
    #a = np.cross(q_v, v)
    #a += np.multiply(q[0],v)
    #a = np.cross(q_v, a)
    #return a+a+v

