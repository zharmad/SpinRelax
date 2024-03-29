import numpy as np
from scipy.optimize import curve_fit
import sys
from collections import OrderedDict

"""
This header script stores a number of autocorrelation classes and handlers.

It's intent is to containerise and obscure the varied levels of accuracy needed to fit
a large number of C(t) from external scripts.
"""
class autoCorrelations:
    """
    Container class for handling a set of autocorrelation models, e.g. from a single file or protein.

    Adapted to also contain target data

    """
    def __init__(self):
        self.nModels  = 0
        self.model    = OrderedDict()
        self.nTargets = 0
        self.DeltaT   = OrderedDict()
        self.Decay    = OrderedDict()
        self.dDecay   = OrderedDict()
        
    def get_names(self):
        return np.array( [k for k in self.model.keys()] )
    
    def get_params_as_list(self):
        keys = self.model.keys()
        S2  = [ self.model[k].S2  for k in keys ]
        C   = [ self.model[k].C   for k in keys ]
        tau = [ self.model[k].tau for k in keys ]
        S2Fast = [self.model[k].calc_S2Fast() for k in keys ]
        return S2, C, tau, S2Fast

    def set_zeta(self, zeta):
        for m in self.model.values():
            m.set_zeta(zeta)
    def get_zeta(self):
        for m in self.model.values():
            return m.get_zeta()        
    
    def add_model(self, key, name=None, listC=[], listTau=[], S2=None, bS2Fast=False, bSort=True):
        """
        Return the last added model for further manipulation.
        """
        if name is None:
            name=key
        self.model[key] = autoCorrelationModel(name, listC, listTau, S2, bS2Fast, bSort)
        self.nModels = len(self.model)
        return self.model[key]

    def get_nth_model(self, n):
        keys=self.get_names()
        return self.model[keys[n]]
    
    def remove_model(self, key=None, index=None):
        if not key is None:
            self.model.pop(key)
        elif not index is None:
            keys = [k for k in self.model.keys()]
            self.model.pop(keys[index])
        else:
            print("= = = ERROR in autoCorrelations.remove_model(); it needs at least one optional argument!)", file=sys.stderr)
            return
        self.nModels=len(self.model)
        
    def rename_models(self, listNames):
        if len(listNames) != len(self.model):
            print("= = = ERROR in autoCorrelations.rename_model(); length of lists are not equal!", file=sys.stderr)
            return
        for k, n in zip(self.model.keys(),listNames):
            self.model[k].name=n

    def report(self):
        print("Number of C(t) models loaded:", self.nModels )
        print("Number of targets loaded:", self.nTargets )
            
    def report_all_models(self):
        for m in self.model.values():
            m.report()
            
    def add_target(self, key, DeltaT, Decay, dDecay):
        self.DeltaT[key] = DeltaT
        self.Decay[key]  = Decay
        self.dDecay[key] = dDecay
        self.nTargets    = len(self.DeltaT)

    def import_target_array(self, keys, DeltaT, Decay, dDecay=None):
        if dDecay is None:
            for i, k in enumerate(keys):
                self.add_target(k, DeltaT[i], Decay[i], None)
        else:
            for i, k in enumerate(keys):
                self.add_target(k, DeltaT[i], Decay[i], dDecay[i])

    def rescale_time(self, f):
        if self.nModels > 0:
            for k in self.model.keys():
                self.model[k].tau *= f
        if self.nTargets > 0:
            for k in self.model.keys():
                self.DeltaT[k] *= f
                            
    def export(self, fileName, style='xmgrace'):
        fp = open(fileName, 'w')
        s = 0
        for k,m in self.model.items():
            # = = = Report outputs the parameters in a way that is encoded.
            m.report(style='xmgrace', fp=fp)
            dt=self.DeltaT[k] ; Ct=self.Decay[k]
            ymodel=m.eval(dt)
            #Print the fitted Ct model into file
            print( "@s%d legend \"Res %d\"" % (s, m.name), file=fp )
            for j in range(len(ymodel)):
                print("%8g %8g" % (dt[j], ymodel[j]), file=fp )
            print( '&', file=fp )
            for j in range(len(ymodel)):
                print("%8g %8g" % (dt[j], Ct[j]), file=fp )
            print( '&', file=fp )
            s+=2

        # WIP
        return

class autoCorrelationModel:
    """
    A class that handles modelling of an autocorrelation function via a set of exponentials.
    Contains basic fitting and reporting funcationalities.

    The transient components are normally sorted on creation from fast to slow.
    """
    # = = = Class dictionary.
    dictGreek=np.array(['a','b','g','d','e','z','h'])
    
    def __init__(self, name='Fit', listC=[], listTau=[], S2=None, bS2Fast=False, bSort=True):
        self.name    = name
        self.nParams = 0 
        self.tau     = np.array(listTau, dtype=float)
        self.C       = np.array(listC, dtype=float)
        self.bS2Fast = bS2Fast
        self.S2      = S2
        self.nComps  = len(self.C)
        self.nParams = len(self.C)+len(self.tau)
        self.bHasFit = False
        self.zeta    = 1.0
        if bS2Fast:
            self.nParams += 1
            if self.S2 == None:
                print("= = = ERROR: S2 must be given in fitPatam initialisation is bS2Fast is set to True!")
                sys.exit(1)
        if self.S2 == None:
            self.S2 = 1.0 - np.sum(self.C)
        self.check_consistency()
        if self.nComps>1 and bSort:
            self.sort_components()

    def check_consistency(self):
        if self.nComps<1:
            return
        if len(self.C) != len(self.tau):
            print("= = = ERROR: transient components in fitParam initialisation do not have matching number of parameters!")
            sys.exit(1)
        if not self.bS2Fast:
            #  All components must add to 1.0
            sumS = self.S2+np.sum(self.C)
            if not np.all( (np.isclose(sumS, 1.0, rtol=1e-6)) ):
                print("= = = ERROR: Contribution of components in fitParam initialisation do not sum sufficeintly close to 1.00!")
                sys.exit(1)

    def copy(self):
        new = autoCorrelationModel()
        new.copy_from(self)
        return new

    def copy_from(self, src):
        self.name    = src.name
        self.nParams = src.nParams
        self.tau     = np.copy(src.tau)
        self.C       = np.copy(src.C)
        self.bS2Fast = src.bS2Fast
        self.S2      = src.S2
        self.nComps  = src.nComps
        self.bHasFit = src.bHasFit
        if src.bHasFit:
            self.set_uncertainties_from_list( src.get_uncertainties_as_list() )
            self.chiSq = src.chiSq

    def add_transient_component(self, C, tau):
        self.tau = np.append(self.tau, tau)
        self.C   = np.append(self.C, C)
        self.nComps  += 1
        self.nParams += 2

    def calc_S2Fast(self):
        if self.bS2Fast:
            return 1.0 - self.S2 - np.sum(self.C)
        else:
            return 0.0

    def sort_components(self):
        inds     = np.argsort(self.tau)
        self.tau = self.tau[inds]
        self.C   = self.C[inds]
        if self.bHasFit:
            self.dtau = self.dtau[inds]
            self.dC   = self.dC[inds]

    def set_zeta(self, zeta):
        """
        QM zero-point vibrations, which universally damps all components constants C and S2,
        leaving the sum to be <1.0 where S2Fast is not involved.
        This is meant for downstram spin relaxation computations that ignore the S2Fast component.
        It currently does not affect the test for unity or the computation of S2fast.
        The fast component should ideally incorporate for zeta such that the total sum is still 1.0.
        """
        self.zeta=zeta

    def get_zeta(self):
        return self.zeta
        
    def report(self, style='stdout', fp=sys.stdout ):
        if style == 'stdout':
            print( "Name: %s" % self.name, file=fp )
            if self.bHasFit:
                print( '  chi-Square: %g ' % self.chiSq, file=fp )
                if self.bS2Fast:
                    print( "  S2_fast: %g" % self.calc_S2Fast(), file=fp)
                for i in range(self.nComps):
                    print( "  component %s, const.: %g +- %g" % (autoCorrelationModel.dictGreek[i], self.C[i], self.dC[i]), file=fp )
                    print( "  component %s, tau: %g +- %g" % (autoCorrelationModel.dictGreek[i], self.tau[i], self.dtau[i]), file=fp )
                print( "  S2_0: %g +- %g" % (self.S2, self.dS2), file=fp )
            else:
                if self.bS2Fast:
                    print( "  S2_fast: %g" % self.calc_S2Fast(), file=fp)
                for i in range(self.nComps):
                    print( "  component %s, const.: %g " % (autoCorrelationModel.dictGreek[i], self.C[i]), file=fp )
                    print( "  component %s, tau: %g " % (autoCorrelationModel.dictGreek[i], self.tau[i]), file=fp )
                print( "  S2_0: %g" % self.S2, file=fp )
        elif style == 'xmgrace':
            # Print header into the Ct model file
            print( '# Residue: %s ' % self.name, file=fp )
            if self.bHasFit:
                print( '# Chi-Square: %g ' % self.chiSq, file=fp )
                if self.bS2Fast:
                    print( '# Param S2_fast: %g +- 0.0' % self.calc_S2Fast(), file=fp )
                    print( '# Param S2_0: %g +- %g' % (self.S2, self.dS2), file=fp )
                else:
                    print( '# Param S2_0: %g +- 0.0' % self.S2, file=fp )
                for i in range(self.nComps):
                    print( '# Param C_%s: %g +- %g' % (autoCorrelationModel.dictGreek[i], self.C[i], self.dC[i]), file=fp )
                    print( '# Param tau_%s: %g +- %g' % (autoCorrelationModel.dictGreek[i], self.tau[i], self.dtau[i]), file=fp )
            else:
                if self.bS2Fast:
                    print( '# Param S2_fast: %g' % self.calc_S2Fast(), file=fp )
                print( '# Param S2_0: %g' % self.S2, file=fp )
                for i in range(self.nComps):
                    print( '# Param C_%s: %g'   % (autoCorrelationModel.dictGreek[i], self.C[i]), file=fp )
                    print( '# Param tau_%s: %g' % (autoCorrelationModel.dictGreek[i], self.tau[i]), file=fp )
        else:
            print("= = = ERROR: fitParam.report() does not recognise the style argument! "
                  "Choices are: stdout, xmgrace", file=sys.stderr)

    def eval(self, DeltaT):
        """
        Vectorised computation function. dt is expected to be a 1-D array that is broadcast to a new axis 0.
        """
        return self.zeta*( self.S2+np.sum(self.C[:,np.newaxis]*np.exp(-1.0*DeltaT[np.newaxis,:]/self.tau[:,np.newaxis]),axis=0) )

    def calc_chiSq(self, DeltaT, Decay, dDecay=None):
        if dDecay is None:
            return np.mean(np.square(self.eval(DeltaT)-Decay))
        else:
            return np.mean(np.square(self.eval(DeltaT)-Decay)/dDecay)

    def optimised_curve_fitting(self, DeltaT, Decay, dDecay=None, listDoG=[2,3,5,7,9], chiSqThreshold=0.5, fp=sys.stdout):
        """
        Conduct multiple curve fits over a set of degreee of freedoms given by listDoG.
        """
        print("= = = Conducting optimised fit for %s with %s degrees of freedoms..." % (self.name, str(listDoG)), file=fp)
        bFirst=True ; prev=self.copy()
        for nParams in listDoG:
            self.set_nParams( nParams )
            chiSq, bQuality = self.conduct_curve_fitting(DeltaT, Decay, dDecay, bReInitialise=True)
            print("    ...fit with %i params yield chiSq of %g" % (nParams, chiSq), file=fp)
            if bFirst:
                if np.all(bQuality):
                    prev.copy_from(self)
                    bFirst=False
                continue
            if not np.all(bQuality):
                print("    ...fit with %i params failed >0 quality checks, will stop." % nParams, file=fp)
                break
            if chiSq >= prev.chiSq*chiSqThreshold:
                print("    ...fit with %i params did not show sufficiently improved chi values. Will stop." % nParams, file=fp)
                break
            prev.copy_from(self)
        if bFirst:
            print("    ...ERROR: fit with %i params has never generated a satisfactory outcome!" % nParams, file=fp)
        else:
            self.copy_from(prev)
        return self.chiSq

    def conduct_curve_fitting(self, DeltaT, Decay, dDecay=None, bReInitialise=False, fp=sys.stdout):
        """
        Uses this class as a framework for invoking scipy.optimize.curve_fitting, obscuring the details on 
        arrangement of variables within the curve_fitting script.
        Bounds are determined based on the input 1D-DeltaT vector, assumed to be montonically increasing.
        E.g., a maximum of 10*DeltaT is set for time-constant tau, as it's impractical to observe motions
        that are much greater than covered by autocorrelation.
        Returns chi value, uncertainties over all parameters, and the model fit itself as a bonus.
        """
        if bReInitialise:
            self.initialise_for_fit_advanced(DeltaT, Decay)
        #if True:
        #    self.report()
        #    print( curvefit_exponential(np.array([0.0,100.0,1000.0,10000.0]), *self.get_params_as_list()) )
        bQuality=[True,True,True]
        try:
            paramOpt, dParamMatrix = curve_fit(curvefit_exponential, DeltaT, Decay, sigma=dDecay,
                               p0     = self.get_params_as_list(),
                               bounds = self.get_bounds_as_list(tauMax=DeltaT[-1]*10))
        except:
            print( "= = = WARNING, curve fitting of %s with %i params failed!" % (self.name,self.nParams), file=fp)
            bQuality[0]=False
            return np.inf, bQuality
        dParam =  np.sqrt(np.diag( dParamMatrix ) )
        if not self.bS2Fast:
            self.S2=1.0-np.sum(self.C)
        # = = = Run checks
        if np.any( dParam > paramOpt ):
            print( "= = = WARNING, curve fitting of %s with %i params indicates overfitting." % (self.name,self.nParams), file=fp)
            bQuality[1]=False
        if self.S2+np.sum(self.C) > 1.0:
            print( "= = = WARNING, curve fitting of %s with %i params returns sum>1." % (self.name,self.nParams), file=fp)
            bQuality[2]=False

        self.set_params_from_list(paramOpt)
        self.set_uncertainties_from_list( dParam )
        self.bHasFit=True
        self.chiSq = self.calc_chiSq( DeltaT, Decay, dDecay )
        self.sort_components()
        return self.chiSq, bQuality

    def initialise_for_fit_basic(self, tMax, tStep, nParams=None):
        """
        This generalised algorithm distributes the starting timescale evenely in log-space between the maximum time delay, and the smallest differnce between times.
        It is meant to attempt to capture multiple timescales relatively evenly.
        """
        if not nParams is None:
            self.set_params( nParams )
        self.tau = np.logspace( np.log10(tStep), np.log10(tMax*2.0), self.nComps+2 )[1:-1]
        self.C  = [1.0/(self.nComps+1)]*self.nComps
        self.S2 = 1.0/(self.nComps+1)
        self.bHasFit = False

    def initialise_for_fit_advanced(self, DeltaT, Decay, nParams=None, nSample=10):
        if not nParams is None:
            self.set_params( nParams )
        self.tau = np.logspace( np.log10(np.mean(DeltaT[1:]-DeltaT[:-1])),
                                np.log10(DeltaT[-1]*2.0),
                                self.nComps+2 )[1:-1]

        nPoints=len(Decay)
        avgBeg=np.mean(Decay[:nSample])
        avgEnd=np.mean(Decay[-nSample:])
        self.C  = [np.fabs(avgBeg-avgEnd)/self.nComps]*self.nComps
        if self.bS2Fast:
            self.S2 = avgEnd
        else:
            self.S2 = 1.0-np.mean(self.C)
        self.bHasFit = False

    def set_nParams(self, n):
        self.nParams = n
        self.nComps  = int(n/2)
        if n%2==1:
            self.bS2Fast=True
        else:
            self.bS2Fast=False

    def get_params_as_list(self):
        if self.bS2Fast:
            return list(self.C)+list(self.tau)+[self.S2]
        else:
            return list(self.C)+list(self.tau)

    def set_params_from_list(self, l):
        self.C   = l[0:self.nComps]
        self.tau = l[self.nComps:2*self.nComps] 
        if self.bS2Fast:
            self.S2 = l[-1]
        else:
            self.S2 = 1.0-np.sum(self.C)

    def get_uncertainties_as_list(self):
        if self.bS2Fast:
            return list(self.dC)+list(self.dtau)+[self.dS2]
        else:
            return list(self.dC)+list(self.dtau)

    def set_uncertainties_from_list(self,l):
        self.dC = np.array(l[0:self.nComps], dtype=float)
        self.dtau = np.array(l[self.nComps:2*self.nComps], dtype=float)
        if self.bS2Fast:
            self.dS2 = l[-1]
        else:
            self.dS2 = 0.0

    def get_bounds_as_list(self, tauMax=np.inf):
        if self.bS2Fast:
            return (0.0,[1.0]*self.nComps+[tauMax]*self.nComps+[1.0])
        else:
            return (0.0,[1.0]*self.nComps+[tauMax]*self.nComps)


def curvefit_exponential(DeltaT, *params):
    n=len(params) ; nn=int(n/2)
    C=np.array(params[0:nn], dtype=float)
    tau=np.array(params[nn:2*nn], dtype=float)
    if n%2==1:
        S2=params[-1]
    else:
        S2=1.0-np.sum(C)
    return S2+np.sum(C[:,np.newaxis]*np.exp(-1.0*DeltaT[np.newaxis,:]/tau[:,np.newaxis]),axis=0)

def _get_key( index, var ):
    return str(index)+"-"+var

def read_fittedCt_parameters(fileName):
    """
    Reading from a SpinRelax output file with *_fittedCt.dat as suffix.
    """
    obj = autoCorrelations()
    index  = None ; S2_slow = None ; S2_fast = None
    tmpC   = OrderedDict() ; tmpTau = OrderedDict()
    bParamSection=False
    with open(fileName) as fp:
        for line in fp.readlines():
            if line.startswith("#"):
                l = line.split()
                if l[1].startswith("Residue"):
                    if bParamSection:
                        print( "= = = ERROR in read_fittedCt_parameters: New parameter section detected when old parameter section is still being read! %s " % fileName, file=sys.stderr )
                        sys.exit(1)
                    bParamSection=True
                    # = = Mark beginning of parameters
                    index = str(l[-1])
                elif l[1].startswith("Param"):
                    parName=l[2]
                    value=float(l[-3])
                    error=float(l[-1])
                    if parName.startswith("S2_0"):
                        S2_slow = value
                    elif parName.startswith("S2_fast"):
                        S2_fast = value
                    elif parName.startswith("C_"):
                        tmpKey = _get_key(index, parName[2])
                        #print( tmpKey, value )
                        tmpC[tmpKey]=value
                    elif parName.startswith("tau_"):
                        tmpKey = _get_key(index, parName[4])
                        #print( tmpKey, value )
                        tmpTau[tmpKey]=value
                    else:
                        # = = Comment line not containing relevant parameters.
                        continue
            else:
                # = = Mark end of parameters with anything that is not a comment, including an empty line.
                if bParamSection:
                    listC   = [ tmpC[k]   for k in tmpC.keys()]
                    listTau = [ tmpTau[k] for k in tmpC.keys()]
                    obj.add_model(index, S2=S2_slow, listC = listC, listTau = listTau, bS2Fast = not S2_fast is None )
                    bParamSection=False
                    tmpC={} ; tmpTau={} ; S2_fast=None ; S2_slow=None ; index=None
                continue

    # = = = Read finished.
    return obj
	   
#def func_exp_decay1(t, tau_a):
#    return np.exp(-t/tau_a)
#def func_LS_decay2(t, S2_a, tau_a):
#    return S2_a + (1-S2_a)*np.exp(-t/tau_a)
#def func_LS_decay3(t, S2_0, S2_a, tau_a):
#    return S2_0*(S2_a + (1-S2_a)*np.exp(-t/tau_a))
#def func_LS_decay4(t, S2_a, tau_a, S2_b, tau_b):
#    return (S2_a + (1-S2_a)*np.exp(-t/tau_a)) * (S2_b + (1-S2_b)*np.exp(-t/tau_b))
#def func_LS_decay5(t, S2_0, S2_a, tau_a, S2_b, tau_b ):
#    return S2_0*(S2_a + (1-S2_a)*np.exp(-t/tau_a)) * (S2_b + (1-S2_b)*np.exp(-t/tau_b))
#def func_LS_decay6(t, S2_a, tau_a, S2_b, tau_b, S2_g, tau_g ):
#    return (S2_a + (1-S2_a)*np.exp(-t/tau_a)) * (S2_b + (1-S2_b)*np.exp(-t/tau_b)) * (S2_g + (1-S2_g)*np.exp(-t/tau_g))
#def func_LS_decay7(t, S2_0, S2_a, tau_a, S2_b, tau_b, S2_g, tau_g ):
#    return S2_0*(S2_a + (1-S2_a)*np.exp(-t/tau_a)) * (S2_b + (1-S2_b)*np.exp(-t/tau_b)) * (S2_g + (1-S2_g)*np.exp(-t/tau_g))
#def func_LS_decay8(t, S2_a, tau_a, S2_b, tau_b, S2_g, tau_g, S2_d, tau_d):
#    return (S2_a + (1-S2_a)*np.exp(-t/tau_a)) * (S2_b + (1-S2_b)*np.exp(-t/tau_b)) * (S2_g + (1-S2_g)*np.exp(-t/tau_g)) * (S2_d + (1-S2_d)*np.exp(-t/tau_d))
#def func_LS_decay9(t, S2_0, S2_a, tau_a, S2_b, tau_b, S2_g, tau_g, S2_d, tau_d):
#    return S2_0*(S2_a + (1-S2_a)*np.exp(-t/tau_a)) * (S2_b + (1-S2_b)*np.exp(-t/tau_b)) * (S2_g + (1-S2_g)*np.exp(-t/tau_g)) * (S2_d + (1-S2_d)*np.exp(-t/tau_d))

"""
This is a series of exponential functions that are a simple sum of exponentials.
The Odd and even degrees of freedoms determine whether an order paramer S2 is included as a free parameter or not.

Subject to the notion that F(0) = 1, when S2 is included as a free parameter there is an implicit parameter S2_fast
that captures the motions too fast for the discretisation to capture
.
# The odd-numbered  set allows this S^2 to be fitted.
"""
def func_exp_decay1(t, tau_a):
    return np.exp(-t/tau_a)

def func_exp_decay2(t, A, tau_a):
    return (1-A) + A*np.exp(-t/tau_a)
def func_exp_decay4(t, A, tau_a, B, tau_b):
    return (1-A-B) + A*np.exp(-t/tau_a) + B*np.exp(-t/tau_b)
def func_exp_decay6(t, A, tau_a, B, tau_b, G, tau_g ):
    return (1-A-B-G) + A*np.exp(-t/tau_a) + B*np.exp(-t/tau_b) + G*np.exp(-t/tau_g)
def func_exp_decay8(t, A, tau_a, B, tau_b, G, tau_g, D, tau_d):
    return (1-A-B-G-D) + A*np.exp(-t/tau_a) + B*np.exp(-t/tau_b) + G*np.exp(-t/tau_g) + D*np.exp(-t/tau_d)
def func_exp_decay10(t, A, tau_a, B, tau_b, G, tau_g, D, tau_d, E, tau_e):
    return (1-A-B-G-D-E) + A*np.exp(-t/tau_a) + B*np.exp(-t/tau_b) + G*np.exp(-t/tau_g) + D*np.exp(-t/tau_d) + E*np.exp(-t/tau_e)

def func_exp_decay3(t, S2, A, tau_a):
    return S2 + A*np.exp(-t/tau_a)
def func_exp_decay5(t, S2, A, tau_a, B, tau_b ):
    return S2 + A*np.exp(-t/tau_a) + B*np.exp(-t/tau_b)
def func_exp_decay7(t, S2, A, tau_a, B, tau_b, G, tau_g ):
    return S2 + A*np.exp(-t/tau_a) + B*np.exp(-t/tau_b) + G*np.exp(-t/tau_g)
def func_exp_decay9(t, S2, A, tau_a, B, tau_b, G, tau_g, D, tau_d):
    return S2 + A*np.exp(-t/tau_a) + B*np.exp(-t/tau_b) + G*np.exp(-t/tau_g) + D*np.exp(-t/tau_d)
def func_exp_decay11(t, S2, A, tau_a, B, tau_b, G, tau_g, D, tau_d, E, tau_e):
    return S2 + A*np.exp(-t/tau_a) + B*np.exp(-t/tau_b) + G*np.exp(-t/tau_g) + D*np.exp(-t/tau_d) + E*np.exp(-t/tau_e)

def _bound_check(func, params):
    """
    Hack for now.
    """
    if len(params) == 1:
        return False
    elif len(params) %2 == 0 :
        s = sum(params[0::2])
        return (s>1)
    else:
        s = params[0]+sum(params[1::2])
        return (s>1)

def calc_chi(y1, y2, dy=[]):
    if dy != []:
        return np.sum( (y1-y2)**2.0/dy )/len(y1)
    else:
        return np.sum( (y1-y2)**2.0 )/len(y1)

def do_LSstyle_fit(num_pars, x, y, dy=[]):
    if num_pars==1:
        func=func_exp_decay1
        guess=(x[-1]/2.0)
        bound=(0.,np.inf)
    elif num_pars==2:
        func=func_LS_decay2
        guess=(0.5, x[-1]/2.0)
        bound=(0.,[1,np.inf])
    elif num_pars==3:
        func=func_LS_decay3
        guess=(0.69, 0.69, x[-1]/2.0)
        bound=(0.,[1.,1.,np.inf])
    elif num_pars==4:
        func=func_LS_decay4
        guess=(0.69, x[-1]/2.0, 0.69, x[-1]/20.0)
        bound=(0.,[1.,np.inf,1.,np.inf])
    elif num_pars==5:
        func=func_LS_decay5
        guess=(0.71, 0.71, x[-1]/2.0, 0.71, x[-1]/20.0)
        bound=(0.,[1.,1.,np.inf,1.,np.inf])
    elif num_pars==6:
        func=func_LS_decay6
        guess=(0.71, x[-1]/2.0, 0.71, x[-1]/8.0, 0.71, x[-1]/32.0)
        bound=(0.,[1.,np.inf,1.,np.inf,1.,np.inf])
    elif num_pars==7:
        func=func_LS_decay7
        guess=(0.72, 0.72, x[-1]/2.0, 0.72, x[-1]/8.0, 0.72, x[-1]/32.0)
        bound=(0.,[1.,1.,np.inf,1.,np.inf,1.,np.inf])
    elif num_pars==8:
        func=func_LS_decay8
        guess=(0.72, x[-1]/1.0, 0.72, x[-1]/4.0, 0.72, x[-1]/16.0, 0.72, x[-1]/64.0)
        bound=(0.,[1.,np.inf,1.,np.inf,1.,np.inf,1.,np.inf])
    elif num_pars==9:
        func=func_LS_decay9
        guess=(0.74, 0.74, x[-1]/1.0, 0.74, x[-1]/4.0, 0.74, x[-1]/16.0, 0.74, x[-1]/64.0 )
        bound=(0.,[1.,1.,np.inf,1.,np.inf,1.,np.inf,1.,np.inf])

    if dy != []:
        popt, popv = curve_fit(func, x, y, p0=guess, sigma=dy, bounds=bound)
    else:
        popt, popv = curve_fit(func, x, y, p0=guess, bounds=bound)


    ymodel=[ func(x[i], *popt) for i in range(len(x)) ]
    #print( ymodel )

    bExceed=_bound_check(func, popt)
    if bExceed:
        print( "= = = WARNING, curve fitting in do_LSstyle_fit returns a sum>1.", file=sys.stderr )
        return 9999.99, popt, np.sqrt(np.diag(popv)), ymodel
    else:
        return calc_chi(y, ymodel, dy), popt, np.sqrt(np.diag(popv)), ymodel

def do_Expstyle_fit(num_pars, x, y, dy=[]):
    if num_pars==1:
        func=func_exp_decay1
        guess=(x[-1]/2.0)
        bound=(0.,np.inf)
    elif num_pars==2:
        func=func_exp_decay2
        guess=(0.5, x[-1]/2.0)
        bound=(0.,[1,np.inf])
    elif num_pars==3:
        func=func_exp_decay3
        guess=(0.5, 0.5, x[-1]/2.0)
        bound=(0.,[1.,1.,np.inf])
    elif num_pars==4:
        func=func_exp_decay4
        guess=(0.33, x[-1]/20.0, 0.33, x[-1]/2.0)
        bound=(0.,[1.,np.inf,1.,np.inf])
    elif num_pars==5:
        func=func_exp_decay5
        guess=(0.33, 0.33, x[-1]/20.0, 0.33, x[-1]/2.0)
        bound=(0.,[1.,1.,np.inf,1.,np.inf])
    elif num_pars==6:
        func=func_exp_decay6
        guess=(0.25, x[-1]/50.0, 0.25, x[-1]/10.0, 0.25, x[-1]/2.0)
        bound=(0.,[1.,np.inf,1.,np.inf,1.,np.inf])
    elif num_pars==7:
        func=func_exp_decay7
        guess=(0.25, 0.25, x[-1]/50.0, 0.25, x[-1]/10.0, 0.25, x[-1]/2.0)
        bound=(0.,[1.,1.,np.inf,1.,np.inf,1.,np.inf])
    elif num_pars==8:
        func=func_exp_decay8
        guess=(0.2, x[-1]/64.0, 0.2, x[-1]/16.0, 0.2, x[-1]/4.0, 0.2, x[-1]/1.0)
        bound=(0.,[1.,np.inf,1.,np.inf,1.,np.inf,1.,np.inf])
    elif num_pars==9:
        func=func_exp_decay9
        guess=(0.2, 0.2, x[-1]/64.0, 0.2, x[-1]/16.0, 0.2, x[-1]/4.0, 0.2, x[-1]/1.0 )
        bound=(0.,[1.,1.,np.inf,1.,np.inf,1.,np.inf,1.,np.inf])

    if dy != []:
        popt, popv = curve_fit(func, x, y, p0=guess, sigma=dy, bounds=bound)
    else:
        popt, popv = curve_fit(func, x, y, p0=guess, bounds=bound)

    ymodel=[ func(x[i], *popt) for i in range(len(x)) ]
    #print( ymodel )

    bExceed=_bound_check(func, popt)
    if bExceed:
        print( "= = = WARNING, curve fitting in do_LSstyle_fit returns a sum>1.", file=sys.stderr )
        return 9999.99, popt, np.sqrt(np.diag(popv)), ymodel
    else:
        return calc_chi(y, ymodel, dy), popt, np.sqrt(np.diag(popv)), ymodel

