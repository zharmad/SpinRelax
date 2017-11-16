import numpy as np
import sys
# This function returns a string that contains
# the value of the function with its error formatted
# to the same exponent.
# e.g. f(0.00302, 4.52e-5, 2) = 3.020e-5 +- 0.045e-5
# The precision argument N sets the number of decimal points
# such at most N+1 sig. figs. are in both values

def normalise_vector_array(v):
    """
    Takes an ND-array of vectors of shape (i, j, k, ..., 3)
    and noramlises it along the last axis.
    """
    return v/np.sqrt((v**2).sum(-1))[..., np.newaxis]

def format_float_with_error(val, err, prec=4):
    # NB: int floors anyway.
    exp_val = np.floor(np.log10(np.abs(val)))
    exp_err = np.floor(np.log10(np.abs(err)))

    exp_out = max(exp_val,exp_err)
    sig_out = prec
    #sig_out = prec+int(np.abs(exp_val-exp_err))
    #r = " %.*e%i +- %.*e%i " % ( sig_val, val*10**(exp_out-exp_val), exp_out, sig_err, )
    return "%.*fe%i +- %.*fe%i" % (sig_out, val*10**(-exp_out), exp_out, sig_out, err*10**(-exp_out), exp_out)

def load_xy(fn):
    x=[]
    y=[]
    for l in open(fn):
        if l[0]=="#" or l[0]=="@" or l[0]=="&" or l=="" or l[0]=="\n":
            continue
        lines = l.split()
        x.append(float(lines[0]))
        y.append(float(lines[1]))
    return np.array(x), np.array(y)

def load_xys(fn):
    x=[]
    y=[]
    for l in open(fn):
        if l[0]=="#" or l[0]=="@" or l[0]=="&" or l=="":
            continue
        vars = [ float(i) for i in l.split() ]
        x.append(vars[0])
        y.append(vars[1:])
    return np.array(x), np.array(y)

def load_xydy(fn):
    x=[]
    y=[]
    dy=[]
    for l in open(fn):
        if l[0]=="#" or l[0]=="@" or l[0]=="&" or l=="" or l=="\n":
            continue
        lines = l.split()
        x.append(float(lines[0]))
        y.append(float(lines[1]))
        if len(lines) <=2:
            print >> sys.stderr, '= = ERROR: in general_scripts.load_xydy, input file does not have a third column as dy! Aborting..'
            print >> sys.stderr, '- - line content: '+l
            sys.exit(1)
        dy.append(float(lines[2]))
    return np.array(x), np.array(y), np.array(dy)

def load_xylist(fn):
    xlist=[] ; ylist=[]
    x=[] ; y=[]
    for l in open(fn):
        if l[0]=="#" or l[0]=="@" or l=="":
            continue
        if l[0]=="&":
            xlist.append(x) ; ylist.append(y)
            x=[] ; y=[]
            continue
        lines = l.split()
        x.append(float(lines[0])) ; y.append(float(lines[1]))
    if x != []:
        xlist.append(x) ; ylist.append(y)

    return xlist, ylist

def load_xydylist(fn):
    xlist=[] ; ylist=[] ; dylist=[]
    x=[] ; y=[] ; dy=[]
    for l in open(fn):
        if l[0]=="#" or l[0]=="@" or l=="":
            continue
        if l[0]=="&":
            xlist.append(x) ; ylist.append(y) ; dylist.append(dy)
            x=[] ; y=[] ; dy=[]
            continue
        lines = l.split()
        x.append(float(lines[0]))
        y.append(float(lines[1]))
        dy.append(float(lines[2]))

    if x != []:
        xlist.append(x) ; ylist.append(y) ; dylist.append(dy)

    return xlist, ylist, dylist

def load_sxydylist(fn, key="legend"):
    leglist=[]
    xlist=[]
    ylist=[]
    dylist=[]
    x=[] ; y=[] ; dy=[]
    for l in open(fn):
        lines = l.split()
        if l[0]=="#" or l[0]=="@" or l=="":
            if key in l:
                leglist.append(lines[-1].strip('"'))
            continue
        if l[0]=="&":
            xlist.append(x) ; ylist.append(y)
            if len(dy)>0:
                dylist.append(dy)
            x=[] ; y=[] ; dy=[]
            continue
        x.append(float(lines[0]))
        y.append(float(lines[1]))
        if len(lines) > 2:
            dy.append(float(lines[2]))

    if x != []:
        xlist.append(x) ; ylist.append(y) ; dylist.append(dy)

    if dylist != []:
        return leglist, np.array(xlist), np.array(ylist), np.array(dylist)
    else:
        return leglist, np.array(xlist), np.array(ylist), []

#def print_flex(fn, datablock=[], header="", legs=[]):
#    fp=open(fn,'w')
#    if header != "":
#        print >>fp, header
#
#    nlegs = len(legs)
#    if nlegs>0:
#        bLegend = True
#    else:
#        bLegend = False
#
#    for i in range(len(x)):
#        print >> fp, x[i], y[i]
#    fp.close()


def print_xy(fn, x, y, dy=[], header=""):
    fp=open(fn,'w')
    if header != "":
        print >>fp, header
    if dy==[]:
        for i in range(len(x)):
            print >> fp, x[i], y[i]
    else:
        for i in range(len(x)):
            print >> fp, x[i], y[i], dy[i]
    fp.close()

def print_xydy(fn, x, y, dy, header=""):
    print_xy(fn, x, y, dy, header)

def print_xylist(fn, x, ylist, bCols=False, header=""):
    """
    Array formats: x(nvals) y(nplots,nvals)
    bCols will stack all y contents in the same line, useful for errors.
    """
    fp = open( fn, 'w')
    if header != "":
        print >> fp, header
    ylist=np.array(ylist)
    shape=ylist.shape
    print shape
    if len(shape)==1:
        for j in range(len(x)):
            print >> fp, x[j], ylist[j]
        print >> fp, "&"
    elif len(shape)==2:
        nplot=shape[0]
        nvals=shape[1]
        if bCols:
            for j in range(nvals):
                print >> fp, x[j],
                for i in range(nplot):
                    print >> fp, ylist[i][j],
                print >> fp, ""
            print >> fp, "&"
        else:
            for i in range(nplot):
                for j in range(len(x)):
                    print >> fp, x[j], ylist[i][j]
                print >> fp, "&"
    fp.close()

def print_sxylist(fn, legend, x, ylist, header=[]):
    fp = open( fn, 'w')
    for line in header:
        print >> fp, "%s" % line

    ylist=np.array(ylist)
    shape=ylist.shape
    nplot=len(ylist)
    s=0
    for i in range(nplot):
        print >> fp, "@s%d legend \"%s\"" % (s, legend[i])
        for j in range(len(x)):
            print >> fp, x[j], str(ylist[i][j]).strip('[]')
        print >> fp, "&"
        s+=1
    fp.close()

def print_s3d(fn, legend, arr, cols, header=[]):
    fp = open( fn, 'w')
    for line in header:
        print >> fp, "%s" % line

    shape=arr.shape
    ncols=len(cols)
    nplot=shape[0]
    s=0
    for i in range(nplot):
        print >> fp, "@s%d legend \"%s\"" % (s, legend[i])
        for j in range(shape[1]):
            for k in range(ncols):
                print >> fp, arr[i,j,cols[k]],
                print >> fp, " ",
            print >> fp, ""
        print >> fp, "&"
        s+=1
    fp.close()


def print_R_hist(fn, hist, edges, header=''):
    fp = open(fn, 'w')
    nbins=hist.shape
    dim=len(nbins)

    if header != '':
        print >> fp, '%s' % header
    print >> fp, '# DIMENSIONS: %i' % dim
    binwidth=np.zeros(dim)
    print >> fp, '# BINWIDTH: ',
    for i in range(dim):
        binwidth[i]=(edges[i][-1]-edges[i][0])/nbins[i]
        print >> fp, '%g ' % binwidth[i],
    print >> fp, ''
    print >> fp, '# NBINS: ',
    for i in range(dim):
        print >> fp, '%g ' % nbins[i],
    print >> fp, ''
    for index, val in np.ndenumerate(hist):
        for i in range(dim):
            print >> fp, '%g %g ' % (edges[i][index[i]], edges[i][index[i]+1]),
        print >> fp, '%g' % val

def print_gplot_hist(fn, hist, edges, header='', bSphere=False):

    fp = open(fn, 'w')
    nbins=hist.shape
    dim=len(nbins)

    # For Gnuplot, we should plot the average between the two edges.
    # ndemunerate method iterates across all specified dimensions
    # First gather and print some data as comments to aid interpretation and future scripting.
    # Use final histogram output as authoritative source!
    if header != '':
        print >> fp, '%s' % header
    print >> fp, '# DIMENSIONS: %i' % dim
    binwidth=np.zeros(dim)
    print >> fp, '# BINWIDTH: ',
    for i in range(dim):
        binwidth[i]=(edges[i][-1]-edges[i][0])/nbins[i]
        print >> fp, '%g ' % binwidth[i],
    print >> fp, ''
    print >> fp, '# NBINS: ',
    for i in range(dim):
        print >> fp, '%g ' % nbins[i],
    print >> fp, ''

    if bSphere:
        if dim != 2:
            print >> sys.stderr, "= = = ERROR: histogram data is not in 2D, but spherical histogram plotting is requested!"
            sys.exit(1)
        # Handle spherical data by assuming that X is wrapped, and y is extended.
        # Can assume data is only 2D.
        xmin=0.5*(edges[0][0]+edges[0][1])
        ymin=edges[1][0]
        ymax=edges[1][-1]
        for eX in range(nbins[0]):
            xavg=0.5*(edges[0][eX]+edges[0][eX+1])
            # Print polar-caps to complete sphere
            print >> fp, '%g %g %g' % (xavg, ymin, hist[eX][0])
            for eY in range(nbins[1]):
                yavg=0.5*(edges[1][eY]+edges[1][eY+1])
                print >> fp, '%g %g %g' % (xavg, yavg, hist[eX][eY])
            print >> fp, '%g %g %g' % (xavg, ymax, hist[eX][-1])
            print >> fp, ''
        # Print first line again to complete sphere
        print >> fp, '%g %g %g' % (xmin, ymin, hist[0][0])
        for eY in range(nbins[1]):
            yavg=0.5*(edges[1][eY]+edges[1][eY+1])
            print >> fp, '%g %g %g' % (xmin, yavg, hist[0][eY])
        print >> fp, '%g %g %g' % (xmin, ymax, hist[0][-1])
        print >> fp, ''
    else:
        for index, val in np.ndenumerate(hist):
            for i in range(dim):
                x=(edges[i][index[i]]+edges[i][index[i]+1])/2.0
                print >> fp, '%g ' % x ,
            print >> fp, '%g' % val
            if index[-1] == nbins[-1]-1:
                print >> fp, ''

    fp.close()

def print_gplot_4d(fn, datablock, x, y, z, header=''):
    sh = datablock.shape
    dims = len(sh)
    if dims != 3:
        print >> sys.stderr, " = = ERROR: general_scripts.print_gplot_3d requires the data to be in 3D."
        sys.exit(1)

    fp = open(fn, 'w')
    if header != '':
        print >> fp, header

    for i in range(sh[0]):
        for j in range(sh[1]):
            for k in range(sh[2]):
                print >> fp, "%g %g %g %g" % (x[i], y[j], z[k], datablock[i,j,k])

    fp.close()
