# File I/O module designed to make manipulation easier for:
# - DX volumetric files
import numpy as np

# In this form, data is treated as a 3D numpy matrix.
# It needs to convert between linear representation of DX files into the 3D internally.
# The default volume units of VMD is in Angstroms, so if we wish to transform the data
# then we specify a conversion scale.
# Note, this returns data in Angstroms as well.
# Uses by default Crow-major order. This is C-like..?
def read_from_dx(fname, units='A'):
    selfname='dxio.py - read_from_dx'
    if units=='A':
        scale=0.1
    elif units=='nm':
        scale=1.0
    else:
        print '= = ERROR in %s: Units argument of write_to_dx accepts only \'nm\' and \'A\'!' % selfname
        return

    #Initialise variables.
    dims = np.zeros(3)
    orig = np.zeros(3)
    abc  = np.zeros((3,3))
    bHeader = True
    deltadim = 0
    count = 0
    with open(fname) as fp:
        for line in fp:
            #Ignore comments and empty lines
            if len(line)==0 or line[0] == '#':
                continue
            l = line.split()
            if bHeader:
                if l[0] == 'origin':
                    orig = np.multiply(scale, np.array(map(float, l[1:4])))
                    print '= = Input origin: ', orig
                elif l[0] == 'object':
                    if l[1] == '1':
                        dims = np.array(map(int, l[-3:]))
                        print '= = Input dimension: ', dims
                    if l[1] == '3':
                        ntot = int(l[-3])
                        print '= = Expected total data points: ', ntot
                        if ntot != dims[0]*dims[1]*dims[2]:
                            print '= = ERROR in %s: Total data points is not equal to dimensions!?' % selfname
                            return
                        else:
                            data = np.zeros(ntot)
                            bHeader = False
                elif l[0] == 'delta':
                    abc[deltadim] = np.multiply(scale, np.array(map(float, l[1:4])))
                    deltadim=deltadim+1
            else:
                #Data entries
                if count < ntot:
                    for a in range(len(l)):
                        data[count]=float(l[a])
                        count=count+1
                else:
                    if l == []:
                        continue
                    if l[0] == 'object' and l[-1] == 'field':
                        print l[:]

    data_order='C'
    data = np.multiply(1.0/scale**3, np.reshape(data, dims, order=data_order))

    return [data, dims, orig, abc]

#Here the data format read:
#    - dims: number of points in each dimension.
#    - orig: the minimum in each dimensions.
#    - abc: unit cell vectors, or bindwith in each dimension. Needs to be (3,3)
def write_to_dx(fname, data, dims, orig, abc, units='A', bScaleDat=True):
    selfname='dxio.py - write_to_dx'
    fp = open(fname, 'w')
    if units=='A':
        scale=10.0
    elif units=='nm':
        scale=1.0
    else:
        print '= = ERROR in %s: units argument of write_to_dx accepts only \'nm\' and \'A\'!' % selfname
        return
    #Sanity checks.
    if dims[0] != data.shape[0] or dims[1] != data.shape[1] or dims[2] != data.shape[2]:
        print '= = ERROR in %s: Data dimensions do not match with the matrix dimenstions!' % selfname
        print dims, data.shape
        outorig=np.multiply(scale,orig)
        return

    # Check if abc is given as a matrix oa sert of vectors.
    outabc =np.multiply(scale,abc)
    if len(outabc.shape)==1:
        outabc=np.array([abc[0],0,0],[0,abc[1],0],[0,0,abc[2]])
    outorig=np.multiply(scale,orig)
    #Writing headers
    print >> fp, '#DX-file written by dxio.py'
    print >> fp, 'object 1 class gridpositions counts %i %i %i' % (dims[0], dims[1], dims[2])
    print >> fp, 'origin %g %g %g' % (outorig[0], outorig[1], outorig[2])
    for i in range(0,3):
        print >> fp, 'delta %g %g %g' % (outabc[i,0], outabc[i,1], outabc[i,2])
    print >> fp, 'object 2 class gridpositions counts %i %i %i' % (dims[0], dims[1], dims[2])
    ntot=dims[0]*dims[1]*dims[2]
    print >> fp, 'object 3 class array type double rank 0 items %i data follows' % ntot

    #Data files itself. The DX format is actually C-order, so will reformulate F-order matrices.
    if bScaleDat:
        flat = np.multiply(1.0/scale**3,data.flatten(order='C'))
    else:
        flat = data.flatten(order='C')
    #ntot=len(data)
    #print >> fp, 'object 3 class array type double rank 0 items %i data follows' % ntot
    #for i in range(0,len(flat), 3):
    #    print >> fp, '%g %g %g' % (flat[i], flat[i+1], flat[i+2])
    for i in range(len(flat)):
        print >> fp, '%g' % flat[i],
        if i%3==2:
            print >> fp, ''
    if len(flat)%3!=0:
        print >> fp, ''
    print >> fp, ''
    print >> fp, 'object "density [%s^-3]" class field' % units
    fp.close()

def debug_all_vars(data, dims, orig, abc):
    print "= = Data dimensions:", data.shape
    print "= = Interpreted dimensions:", dims
    print "= = Origin:", orig
    print "= = Box vectors (below):"
    print abc

    return
