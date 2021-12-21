import sys, argparse
import numpy as np
import fitting_Ct_functions as fitCt

#import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
#from matplotlib import rc

plt.ioff()

#rc('text', usetex=True)

# Residue: 1
# Chi-value: 0.000753848
# Param S2_0: 0.0914724 +- 0
# Param C_a: 0.0713342 +- 0.0011718
# Param tau_a: 11.6127 +- 0.354249
# Param C_b: 0.336821 +- 0.00268592
# Param tau_b: 195.19 +- 2.4886
# Param C_g: 0.500372 +- 0.00280338
# Param tau_g: 1150.16 +- 8.33356


def _determine_point_size(frac, sizeMin, sizeMax):
    return (sizeMin+frac*(sizeMax-sizeMin))**2.0

def _update_range_if_given( range, min, max):
    out=list(range)
    out[0] = min if min != None else out[0]
    out[1] = max if max != None else out[1]
    return out

def _int_round(val, mod, bUp=False):
    if bUp:
        return val+mod-(val%mod)
    else:
        return val-(val%mod)

parser = argparse.ArgumentParser(description='Plots the fitted parameters from *_fittedCt.dat as a scatter plot.',
                                             formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-v', action='store_true', dest='bVerbose', default=False,
                    help='Turn on debug output.')
parser.add_argument('-f', dest='inFile', type=str, required=True, default='out_fittedCt.dat',
                    help='Input file with formatted Ct-parameter data.')
parser.add_argument('-o', dest='outFile', type=str, default=None,
                    help='Output to file instead of showing.')
parser.add_argument('--cmap', type=str, default='jet',
                    help='The color map name fed into pyplot, unless you want a custom one. See: https://matplotlib.org/examples/color/colormaps_reference.html')
parser.add_argument('--tmin', type=float, default=None,
                    help='Fix a minimum time constant to plot within the components, assigning timescales faster than this to S2_fast.')
parser.add_argument('--tmax', type=float, default=None,
                    help='Fix a maximum time constant to plot within the components, assigning timescales slower than this to S2_slow.')
parser.add_argument('--noshift', action='store_true', default=False,
                    help='Do not shift tau-components outside the plot range, but instead ignore them unscientifically.')
parser.add_argument('--tu', type=str, default='ps',
                    help='Cosmetic labelling of time units for the fitted constants.')
parser.add_argument('--figx', type=float, default=5.0,
                    help='Figure size along X-dimension in inches.')
parser.add_argument('--figy', type=float, default=3.0,
                    help='Figure size along Y-dimension in inches.')
parser.add_argument('--smin', type=float, default=5.0,
                    help='Cosmetic resizing of the points.')
parser.add_argument('--smax', type=float, default=10.0,
                    help='Cosmetic resizing of the points.')
parser.add_argument('--xmin', type=float, default=None,
                    help='Cosmetic x-axis minimum.')
parser.add_argument('--xmax', type=float, default=None,
                    help='Cosmetic x-axis maximum.')
parser.add_argument('--xlabel', type=str, default='Residue index',
                    help='Cosmetic X-label')
parser.add_argument('--sequence', type=str, default=None,
                    help='Cosmetic switch of tick labelling from numbers to a given sequence. Given as a list or a single string.')
parser.add_argument('--xshift', type=float, default=None,
                    help='Shift x-range numbering system. When custom sequence labels are given, this determine the position of the first residue.')
parser.add_argument('--title', type=str, default=None,
                    help='Cosmetic Figure title')


args = parser.parse_args()
bVerbose = args.bVerbose
paramList = fitCt.read_fittedCt_parameters(args.inFile)
print( "= = = Read %s and found %i sets of parameters." % (args.inFile, len(paramList)) )

timeUnits=args.tu
sizeMin=args.smin
sizeMax=args.smax
xMin=args.xmin
xMax=args.xmax
tauMin=args.tmin
tauMax=args.tmax
bDoTauShift=not args.noshift

if args.title != None:
    bDoTitle=True
    figTitle=args.title
else:
    bDoTitle=False

symbolLW=0.5
# or None
symbolColor='black'
# or 'face'

if args.cmap != 'custom':
    colorMap=plt.get_cmap(args.cmap)
else:
    # = = = Not implemented by default. Please edit this segment as you wish. = = =
    print( "= = = Using custom color-mapping." )
    colorMap

if bVerbose:
    for x in paramList:
        x.report()

sumPoints = sum([ x.nComponents for x in paramList])
print( "= = = ..,with a total count of %i transient components." % sumPoints )

posX=[]
posY=[]
points = []
S2slow = []
S2fast = []
for p in paramList:
    resid = float(p.name)
    if args.sequence is None and not args.xshift is None:
        resid += float(args.xshift)

    tmpS2s = p.S2_slow if p.S2_slow != None else 0.0
    tmpS2f = p.S2_fast if p.S2_fast != None else 0.0

    for c in p.components:
        tau   = c[1]
        const = c[0]
        # = = = Segment to shift components into order parameters if the timescales are clearly overfitted.
        bShifted=False
        if bDoTauShift:
            if tauMin != None and tau < tauMin:
                bShifted=True
                tmpS2f += const
            elif tauMax != None and tau > tauMax:
                bShifted=True
                tmpS2s += const

        if not bShifted:
            size  = _determine_point_size(const, sizeMin, sizeMax)
            points.append( [resid,tau,size,const] )

    # = = = Compute modified order parameters with any tau components shifted into this.
    S2slow.append( [resid, tmpS2s, _determine_point_size(tmpS2s, sizeMin, sizeMax)] )
    S2fast.append( [resid, tmpS2f, _determine_point_size(tmpS2f, sizeMin, sizeMax)] )

S2slow = np.array( S2slow )
S2fast = np.array( S2fast )
points = np.array( points )
print( "= = Shape of the read data:", S2slow.shape, points.shape, S2fast.shape )
residFirst=points[0,0]
residLast =points[-1,0]
print( "= = First and last resid label read:", residFirst, residLast )

# = = = = Ignore subplot if all components are 0.0 to prvent an empty plot.
bPlotS2s = True if np.any(S2slow>0) else False
bPlotS2f = True if np.any(S2fast>0) else False

# = = = matplotlib Time! = = =
fig = plt.figure(figsize=(args.figx,args.figy))
fig.subplots_adjust(hspace=0.05)

if bPlotS2f and bPlotS2s:
    ax1 = plt.subplot2grid((5,1),(0,0))
    ax2 = plt.subplot2grid((5,1),(1,0), rowspan=3)
    ax3 = plt.subplot2grid((5,1),(4,0))
    axList=[ax1,ax2,ax3]
elif bPlotS2s:
    ax1 = plt.subplot2grid((5,1),(0,0))
    ax2 = plt.subplot2grid((5,1),(1,0), rowspan=4)
    axList=[ax1,ax2]
elif pPlotS2f:
    ax2 = plt.subplot2grid((5,1),(0,0), rowspan=4)
    ax3 = plt.subplot2grid((5,1),(4,0))
    axList=[ax2,ax3]
else:
    # Lol.
    ax2 = plt.subplot2grid((5,1),(0,0), rowspan=5)
    axList=[ax2]

# = = = specific to the central panel of components: consts and tau
sc = ax2.scatter(points[:,0], points[:,1], s=points[:,2], c=points[:,3], alpha=1.0, cmap=colorMap, vmin=0.0, vmax=1.0, linewidths=symbolLW, edgecolors=symbolColor)
ax2.set_yscale('log')
ax2.set_ylabel('$\\tau$ components [%s]' % timeUnits)

if xMin != None or xMax != None:
    ax2.set_xlim( _update_range_if_given( ax2.get_xlim(), xMin, xMax ) )
else:
    xMin, xMax = ax2.get_xlim()

if tauMin != None or tauMax != None:
    ax2.set_ylim( _update_range_if_given( ax2.get_ylim(), tauMin, tauMax ) )

# = = = Specific to slow order parameter
if bPlotS2s:
    ax1.scatter(S2slow[:,0], S2slow[:,1], s=S2slow[:,2], c=S2slow[:,1], alpha=1.0, cmap=colorMap, vmin=0.0, vmax=1.0, linewidths=symbolLW, edgecolors=symbolColor)
    ax1.set_ylabel('S$^2$')
    ax1.set_xlim(ax2.get_xlim())
    ax1.set_ylim([0.0,1.0])
    ax1.set_yticks([0.5,1.0])
    ax1.set_yticks([0.1,0.2,0.3,0.4,0.6,0.7,0.8,0.9], minor=True)

# = = = specific to fast order parameter
if bPlotS2f:
    ax3.scatter(S2fast[:,0], S2fast[:,1], s=S2fast[:,2], c=S2fast[:,1], alpha=1.0, cmap=colorMap, vmin=0.0, vmax=1.0, linewidths=symbolLW, edgecolors=symbolColor)
    ax3.set_ylabel('S$^2_{fast}$')
    ax3.set_xlim(ax2.get_xlim())
    ax3.set_ylim([0.0,1.0])
    ax3.set_yticks([0.5,1.0])
    ax3.set_yticks([0.1,0.2,0.3,0.4,0.6,0.7,0.8,0.9], minor=True)

# = = = operations on all subplot axes
for a in axList:
    a.set_xlabel(args.xlabel)
    a.grid(which='both', color='lightgrey', linestyle=':', linewidth=1)
    a.set_axisbelow(True)

if args.sequence == None:
    # = = = Automated x-axis tick labeling.
    xRange = ax2.get_xlim()[1] - ax2.get_xlim()[0]
    for a in axList:
        if xRange <= 10:
            a.set_xticks( np.arange(xMin,xMax) )
        elif xRange <= 50:
            a.set_xticks( np.arange(xMin,xMax), minor=True )
            a.set_xticks( np.arange(_int_round(xMin,5, True),xMax,5) )
        elif xRange <= 200:
            a.set_xticks( np.arange(_int_round(xMin,2, True),xMax,2), minor=True )
            a.set_xticks( np.arange(_int_round(xMin,10, True),xMax,10) )
        else:
            a.set_xticks( np.arange(_int_round(xMin,5, True),xMax,5), minor=True )
            a.set_xticks( np.arange(_int_round(xMin,20, True),xMax,20) )
else:
    if args.xshift is None:
        args.xshift=1.0

    l=args.sequence.split()
    if len(l)==1:
        # = = = single string processing.
        l=l[0]
    nPts=len(l)
    for a in axList:
#        a.set_xticks( np.arange(args.xshift+residFirst-1,nPts+args.xshift+residFirst-1) )
        a.set_xticks( np.arange(args.xshift,nPts+args.xshift) )
        a.set_xticklabels( [ x for x in l ] )

labels = [ x.get_text() for x in ax2.get_xticklabels() ]
emptyLabels = ['']*len(labels)
ax1.set_xticklabels(emptyLabels)
if bPlotS2f:
    ax2.set_xticklabels(emptyLabels)

# = = = Insert bells and whistles like titles, colorbars and others.
if bDoTitle:
    plt.suptitle(figTitle)
fig.subplots_adjust(right=0.9)
cbar_ax = fig.add_axes([0.92, 0.1, 0.01, 0.8])
cbar = fig.colorbar(sc, cax=cbar_ax)
#cbar.set_label("Contribution", rotation=90)

# = = = Output.
if args.outFile == None:
    plt.show()
else:
    plt.savefig(args.outFile)
