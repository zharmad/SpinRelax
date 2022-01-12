#!/usr/bin/python
import sys
import numpy as np
from math import *
import npufunc
import general_maths as gm
from collections import OrderedDict
import fitting_Ct_functions as fitCt 
"""
Notes: Uses a user-defined numpy universal function in C (npufunc.so)
to compute F(x,y)=x/(x+y) across an array.

A set of functions to compute the NMR relaxation.
Cf.
- Lee, Rance, Chazin, and Palmer. J. Biomol. NMR, 1997
- Ghose, Fushman, and Cowburn. J Magn. Reson., 2001

For early CSA discussions, see
- Fushman, Tjandra, and Cowburn. J. Am. Chem. Soc., 1998
"""

class gyromag:
    """
    This is the base handler for the gyromagnetic properties of a set of nuclei.
    Here, the time units are set to seconds by default and fixed.
    """
    def __init__(self, isotope, csa=None):
        self.num=1
        self.isotope=isotope
        self.timeUnit='s'
        self.time_fact=_return_time_fact(self.timeUnit)
        self.set_gamma(self.isotope)
        if csa is None:
            self.reset_csa(isotope)
        else:
            self.csa=csa

    def reset_csa(self, name):
        if  name=='15N':
            # Large variation possible. See Fushman, Tjandra, and Cowburn, 1998.
            # Also use more commonly accepted value.            
            self.set_csa(-170e-6)
        elif name=='13C':
            # Good average unknown
            self.set_csa(-130e-6)
        else:
            self.set_csa(0.0)
            
    def set_gamma(self, name):
        """
        Sets the gamma value based on the gyromag's isotope and time unit definition.
        Gamma is in units of rad s^-1 T^-1 .
        """
        if name=='1H':
            #In standard freq: 42.576e6
            self.gamma=267.513e6*self.time_fact
        elif name=='13C':
            self.gamma=67.262e6*self.time_fact
        elif name=='15N':
            self.gamma=-27.116e6*self.time_fact
        elif name=='17O':
            self.gamma=-36.264e6*self.time_fact
        elif name=='19F':
            self.gamma=251.662e6*self.time_fact
        elif name=='31P':
            self.gamma=108.291e6*self.time_fact
                
    def set_csa(self, csa, i=None):
        self.csa = csa
        
    def get_csa(self, i=None):
        return self.csa
        
    def set_time_unit(self, tu):
        old = self.time_fact
        self.time_fact = _return_time_fact(tu)
        mult = self.time_fact/old
        self.gamma *= mult

class gyromagMultiCSA(gyromag):
    def __init__(self, isotope, n, csa=None):
        self.num=n
        self.isotope=isotope
        self.timeUnit='s'
        self.time_fact=_return_time_fact(self.timeUnit)
        self.set_gamma(self.isotope)
        if csa is None:
            self.reset_csa(isotope)
        else:
            self.set_csa(csa)

    def reset_csa(self, name):
        if  name=='15N':
            # Large variation possible. See Fushman, Tjandra, and Cowburn, 1998.
            # Also use more commonly accepted value.            
            self.set_csa(np.repeat(-170e-6,self.num))
        elif name=='13C':
            # Good average unknown
            self.set_csa(np.repeat(-130e-6,self.num))
        else:
            self.set_csa(np.repeat(0.0,self.num))
            
    def set_csa(self, csa, i=None):
        """
        Sets whole array if no index argument is given. Does a sanity check against its own number of nuclei.
        """
        if i is None:
            if self.num == len(csa):
                self.csa = np.array(csa)
            else:
                print("= = ERROR: attempting to set CSA array in gyromagMultiCSA, but the lengths to not match!")
                sys.exit(1)
        else:
            self.csa[i] = csa
        
    def get_csa(self, i=None):
        """
        Returns whole array if no index argument is given.
        """        
        if i is None:
            return self.csa
        else:
            return self.csa[i]
        
# = = = omega = = =
def calc_omega_names(nA, nB):
    out=OrderedDict()
    out['0']=0
    out[nA]=1
    out[nB+'-'+nA]=2
    out[nB]=3
    out[nB+'+'+nA]=4
    return out
    
class angularFrequencies:
    """
    This class handles the angular frequencies and their host nuclei.
    It acts to provide the relevant J(omega) frequencies and nuclei properties to downstream relaxation computations.
    Hosts objects:
    self.gA for heavy nuclei, self.gB for 1H for default.
    """
    # = = = Static class variables = = =
    # Default indexing to create the five NMR-component frequencies
    # J(0), J(wX), J(wH-wX), J(wH), J(wH+wX)
    # In ascending order. This is used to obtain relaxation properties from a five-value J(function)
    iOm0   = 0
    iOmA   = 1
    iOmBmA = 2
    iOmB   = 3
    iOmBpA = 4
    
    def __init__(self, nucleiA='15N', nucleiB='1H', fieldStrength=600, fieldUnit='MHz', timeUnit='ps'):
        """
        Standard initiation mode
        """        
        self.timeUnit=timeUnit
        self.time_fact=_return_time_fact(self.timeUnit)
        self.distUnit='nm'
        self.dist_fact=_return_dist_fact(self.distUnit)
        
        self.gA = gyromag( nucleiA )
        self.gB = gyromag( nucleiB )
        self.rAB = 1.02e-1
       
        self.B0 = None
        self.set_magnetic_field( fieldStrength, fieldUnit )
    
        self.nOmega=5
        self.omega=np.zeros(5)
        self.omegaNames=calc_omega_names(nucleiA,nucleiB)
        self.omega[1] = -1.0*self.gA.gamma*self.B0*self.time_fact            
        self.omega[3] = -1.0*self.gB.gamma*self.B0*self.time_fact
        self.omega[2] = (self.omega[3]-self.omega[1])        
        self.omega[4] = (self.omega[3]+self.omega[1])           

    def report(self):
        print("Field: %g T" % self.get_magnetic_field() )
        print("NucleiA: %s @ %s rad^-1s^-1" % ( self.gA.isotope,self.gA.gamma ) )
        print("NucleiB: %s @ %s rad^-1s^-1" % ( self.gB.isotope,self.gB.gamma ) )
        print("Bond length: %g %s" % (self.rAB, self.distUnit) )
        print("Time units: %s (%g)" % (self.timeUnit, self.time_fact) )
        print("Distance units: %s (%g)" % (self.distUnit, self.dist_fact) )
        print("Angular frequencies names: %s" % (str([k for k in self.omegaNames.keys()])) ) 
        print("Angular frequencies (rad %s^-1 T^-1): %s" % (self.timeUnit, str(self.omega)) ) 
        
    def set_magnetic_field(self, inp, unit ):
        if unit == 'Hz':
            self.B0 = 2.0*np.pi*inp / 267.513e6        
        elif unit == 'MHz':
            self.B0 = 2.0*np.pi*inp / 267.513
        elif unit == 'T':
            self.B0 = inp
        else:
            _BAIL( "set_magnetic_field", "incorrect field units given ( %s )" % unit )
        
    def get_magnetic_field(self, unit='T'):
        if unit == 'T':
            return self.B0
        elif unit == 'MHz':
            return self.B0*267.513/(2.0*np.pi)
        elif unit == 'Hz':
            return self.B0*267.513e6/(2.0*np.pi)
        else:
            _BAIL( "set_magnetic_field", "incorrect field units given ( %s )" % unit )        
        
    def set_time_unit(self, tu):
        old=self.time_fact
        self.time_fact = _return_time_fact(tu)
        # Update all time units can measurements.
        self.timeUnit=tu
        mult = self.time_fact / old
        self.omega *= mult

    def print_omega_names(self):
        for key in self.omegaNames:
            print( key )

    def get_frequencies(self):
        return self.omega        

    def get_nuclei_names(self):
        return [ self.gA.isotope, self.gB.isotope ]
        
    def get_factor_DD(self):
        """
        The maths behind the following two aspects are:
        f_DD  = 0.10* (mu_0*hbar/4.0/pi)**2 * gamma_15N**2 * gamma_1H**2 * r_NH**-6.0
        f_CSA = 2.0/15.0 * gamma_15N**2 * B_0 **2 * DeltaSig_15N**2
        mu_0 = 4*pi*1e-7      ; # m   kg s^-2 A-2
        hbar = 1.0545718e-34  ; # m^2 kg s^-1
        pi   = 3.14159265359
        gamma_1H  = 267.513e6  ; # rad s^-1 T^-1
        gamma_15N = -27.116e6  ; # rad s^-1 T^-1
        omega_15N = - gamma_15N * B_0 .
        r_NH = 1.02e-10 ;# m
         (mu_0*hbar/4.0/pi)**2 m^-1 s^2 is the 10^-82 number below. f_DD and f_CSA are maintained in SI units.
        """
        return 0.10 * 1.1121216813552401e-82*self.gA.gamma**2.0*self.gB.gamma**2.0 *(self.rAB*self.dist_fact)**-6.0

    def get_factor_CSA(self,i=None):
        # = = = Some passing arguments in case CSA is an array and we want only one value.
        return 2.0/15.0 * self.gA.get_csa(i)**2.0 * ( self.gA.gamma * self.B0 )**2        

    def initialise_CSA_array(self, numCSAs, CSAvalues=None):
        self.gA = gyromagMultiCSA( self.gA.isotope, numCSAs, CSAvalues)
    
    def update_CSA_array(self, csa, ind=None):
        self.gA.set_csa(self, csa, ind)
    
# = = = = Global rotations. = = =
    
class globalRotationalDiffusion_Base:
    def __init__(self):
        self.name  = 'base'        
        self.D   = None
        # = = = Storing coefficients for later reference during computation, since they do not update regularly.
        self.D_J = None
        self.A_J = None
        self.bVecs      = False
        self.axisAvg    = None
        self.vecNames   = None
        self.vecXH      = None
        self.vecWeights = None
        
    def report(self):
        print( "Type:", self.name )
        print( "Diffusion tensor components:", self.D )
        if self.bVecs:
            print("Principal-axis frame vectors names:", self.vecNames.shape )
            print("Principal-axis frame vectors shape:", self.vecXH.shape )
            if self.vecWeights is None:
                print("Principal-axis frame vectors has no weights.")
            else:
                print("Principal-axis frame vectors weights shape:", self.vecWeights.shape )
        else:                
            print("No principal-axis frame vectors loaded.")   

    def import_frame_vectors_npz(self, fileName):
        """
        Sets up vec (numReplicates, numVectors, 3) and weights (numReplicates, numVectors)
        This eases downstream broadcasting in spin relaxation computations.
        """
        # = = = Treat as a numpy binary file.
        obj = np.load(fileName, allow_pickle=True )
        # = = = Determine data type
        names = obj['names']
        if obj['bHistogram']:
            if obj['dataType'] != 'LambertCylindrical':
                print( "= = = Histogram projection not supported! %s" % obj['dataType'], file=sys.stderr )
                sys.exit(1)
            vecs, weights = convert_LambertCylindricalHist_to_vecs(obj['data'], obj['edges'])
        else:
            if obj['dataType'] != 'PhiTheta':
                print( "= = = Numpy binary datatype not supported! %s" % obj['dataType'], file=sys.stderr )
                sys.exit(1)
            # = = = Pass phi and theta directly to rtp_to_xyz
            vecs = gm.rtp_to_xyz( obj['data'], vaxis=-1, bUnit=True )
        self.bVecs      = True
        self.vecNames   = names
        
        self.vecXH      = np.swapaxes(vecs,0,1)
        self.vecWeights = np.swapaxes(weights,0,1)
        self.axisAvg = 0
        print("Debug import_frame_vectors_npz:", self.vecXH.shape, self.vecWeights.shape )
        
    def import_frame_vectors_pdb(self, pdbFile, trjFile=None, HSelTxt='name H', XSelText='name N and not resname PRO'):
        """
        Altenative functionality to define single vectors from a reference PDB file,
        which should already be in the principal-axis frame.
        Requires mdtraj
        """
        print( "= = = Using vectors as found directly in the coordinate files, via MDTraj module." )
        print( "= = = NOTE: no fitting is conducted." )        
        import mdtraj as md
        import transforms3d_supplement as qs
        
        if not trjFile is None:
            mol = md.load(trjFile, top=pdbFile)
            print( "= = = PDB file %s and trajectory file %s loaded." % (pdbFile, trjFile) )
        else:
            mol = md.load(pdbFile)
            print( "= = = PDB file %s loaded." % pdbFile )

        indH, indX, resXH = confirm_mdtraj_seltxt(mol, HSelTxt, XSelText)
        # Extract submatrix of vector trajectory
        vecs = np.take(mol.xyz, indH, axis=1) - np.take(mol.xyz, indX, axis=1)
        vecs = qs.vecnorm_NDarray(vecs, axis=2)

        # = = = Check shape and reform the number of dimensions for downstream work.
        #       This is based on the mdtraj output.
        if vecs.shape[0] == 1:
            # Shape (num, 3)
            vecs = vecs[0]
            self.axisAvg = None
        else:
            #shape (num, frames, 3)
            vecs = np.swapaxes(vecs,0,1)
            self.axisAvg = 0
        self.bVecs      = True
        self.vecNames   = resXH
        self.vecXH      = vecs
        self.vecWeights = None
        print("Debug import_frame_vectors_pdb:", self.vecXH.shape )
        
    def import_frame_vectors(self, fileName):
        """
        Reads the frame vectors distribution, however that is formatted.
        Returns the vectors, and maybe weights whose dimensions are (nResidue, nSamples, 3).
        Currently supports only phi-theta formats of vector definitions.
        For straight xmgrace data files, this corresponds to the number of plots, then the data-points in each plot.
        """
        weights = None
        if fileName.endswith('.npz'):
            self.import_frame_vectors_npz(fileName)
            return
        elif fileName.endswith('.pdb'):
            # = = = Attempt naive PDB loader with default arguments.
            self.import_frame_vectors_pdb(fileName)
            return
        else:
            # = = = Assuming a generic XYZ-type file structure.
            names, dist_phis, dist_thetas, dum = gs.load_sxydylist(args.distfn, 'legend')
            vecs = gm.rtp_to_xyz( np.stack( (dist_phis,dist_thetas), axis=-1), vaxis=-1, bUnit=True )
            print( vecs.shape )
            sys.exit()
        if not weights is None:
            print( "    ...converted input phi_theta data to vecXH / weights, whose shapes are:", vecs.shape, weights.shape )
        else:
            print( "    ...converted input phi_theta data to vecXH, whose shape is:", vecs.shape )
        self.axisAvg    = 0
        self.bVecs      = True
        self.vecNames   = names
        self.vecXH      = vecs
        self.vecWeights = weights
        print("Debug import_frame_vectors_generic:", self.vecXH.shape, self.vecWeights.shape )        
        print("ERROR: THe generic dfunctionality has noe been tested. Aborting for safety.")
        sys.exit()
        
    def get_names(self):
        # Temporary fix as names in the npz file are ints and not string
        return [ str(x) for x in self.vecNames]
    #return resIDs, vecs, weights
            
class globalRotationalDiffusion_Isotropic(globalRotationalDiffusion_Base):
    """
    Subcopy for isotropic. Does not use the D_J and A_J coefficients since they are trivial.
    """
    def __init__(self, D=None, tau=None):
        if D is None and tau is None:
            print("= = = ERROR: global rotdif models must be initialised with some D or tau argeument!", file=fp)
            return None
        globalRotationalDiffusion_Base.__init__(self)
        self.name='isotropic'
        if not D is None:
            self.D=D
        else:
            self.D=1.0/(6.0*tau)
    
    def set_Diso(self,Diso):
        self.D=Diso
        self.D_J=self.D        
    def get_Diso(self):
        return self.D
        
    def update_A_coefficients(self):
        return
    def get_A_coefficients(self):
        # A_coefficients_symmtop(v, bProlate=(self.D[0]>self.D[1]))
        return 1.0

    def update_D_coefficients(self):
        return
    def get_D_coefficients(self):
        return self.D
    
    def transform_D(self):
        return self.D   
    
    def calc_Jomega_one(self, omega, CtModel, ind=None):
        """
        This calculates the J value for combining an isotropic global tumbling with
        a fitted internal autocorrelation C(t), where
        C(t) = S2 + Sum{ consts[i] * exp ( -t/tau[i] }
        thus this allows fits to multiple time constants in C(t).
        """
        #def J_combine_isotropic_exp_decayN(RObj.omega, 1.0/(6.0*RObj.rotdifModel.D), S2[i], consts[i], taus[i])
        tauGlob=1.0/(6.0*self.D)
        k = 1.0/tauGlob+1.0/CtModel.tau
        Jmat = CtModel.zeta*CtModel.S2*tauGlob/(1.0+(omega*tauGlob)**2.0)
        for i in range(CtModel.nComps):
            Jmat += CtModel.zeta*CtModel.C[i]*k[i]/(k[i]**2.0+omega**2.0)
        return Jmat

    def calc_Jomega(self, omega, Autocorrs):
        """
        This calculates the J value for combining an isotropic global tumbling with
        a fitted internal autocorrelation C(t), where
        C(t) = S2 + Sum{ consts[i] * exp ( -t/tau[i] }
        thus this allows fits to multiple time constants in C(t).
        """
        #def J_combine_isotropic_exp_decayN(RObj.omega, 1.0/(6.0*RObj.rotdifModel.D), S2[i], consts[i], taus[i])
        tauGlob=1.0/(6.0*self.D)
        Jmat = np.zeros( (Autocorrs.nModels,len(omega) ) )
        # Expect only 2 axes for isotropic diffusion, but just in case...
        for i, model in enumerate( Autocorrs.model.values() ):
            Jmat[...,i,:]=self.calc_Jomega_one(omega, model )
        return Jmat
    
    def calc_Jomega_rigid(self, omega):
        return 6.0*self.D/( (6.0*self.D)**2.0 + np.power(omega,2.0) )

class globalRotationalDiffusion_Axisymmetric(globalRotationalDiffusion_Base):
    """
    Subcopy for axi-symmetric.
    Internal storage will be in terms of D_iso and D_aniso, rather than D_parallel, D_perpendicular
    If the latter is given, then bConvert should be set to True during initialisation.
    """
    def __init__(self, D=None, bConvert=False, tau=None, aniso=None):
        if D is None and (tau is None or aniso is None):
            print("= = = ERROR: global rotdif models must be initialised with some D or tau/aniso argeument!", file=sys.stderr)
            return None
        globalRotationalDiffusion_Base.__init__(self)
        self.name='axisymmetric'
        if not D is None:
            if bConvert:
                self.D=np.array([(2.0*D[1]+D[0])/3.0, D[0]/D[1]], dtype=float)
            else:
                self.D=np.array(D, dtype=float)
            #Dperp = 3.*Diso/(2+Daniso)
            #Dpar  = Daniso*Dperp
            # Convention: Diso, Dani --> Dpar, Dperp
        else:
            self.D=np.array( [1.0/(6.0*tau),aniso], dtype=float)
        if self.D[1]>1:
            self.bProlate=True
        else:
            self.bProlate=False
    
    def set_Diso(self, Diso):
        self.D[0]=Diso
        self.update_D_coefficients()
    def set_Daniso(self, Daniso):
        self.D[1]=Daniso
        self.update_D_coefficients()
        
    def get_Diso(self):
        return self.D[0] 
    def get_Daniso(self):
        return self.D[1]
        

    def update_A_coefficients(self):
        """
        Computes the 3 axisymmetric A-coefficients associated with orientation of the vector w.r.t. to the D_rot ellipsoid.
        v can be many dimensions, as long as the X/Y/Z cartesian dimensions is the last. Two examples:
        When a vector distribution is given originally, this should setup the shape (nReplicates, nVecs, 3).
        When a single vector is given originally, this return (nVecs, 3).
        This is designed to handle many vectors at once.
        Also note: The unique axis changes when Daniso > 1 and when Daniso < 1 so as to preserve Dx<Dy<Dz formalism.
        This is governed by bProlate, which changes the unique axis to x when tumbling is oblate.
        """
        if self.bProlate:
            # Use z-dim.
            z2=np.square(self.vecXH.take(-1, axis=-1))
        else:
            # Use x-dim.
            z2=np.square(self.vecXH.take( 0, axis=-1))
        onemz2=1-z2
        A0 = np.multiply( 3.0, np.multiply(z2,onemz2))
        A1 = np.multiply(0.75, np.square(onemz2))
        A2 = np.multiply(0.25, np.square(np.multiply(3.0,z2)-1.0))
        self.A_J = np.stack((A0,A1,A2),axis=-1)
        # = = = This is for broadcasting rules down the track
        #if len(self.A_J.shape)==3:
        #    self.A_J=np.swapaxes(self.A_J, 0, 1)
    
    def get_A_coefficients(self, ind=None):
        if ind is None:
            return self.A_J
        else:
            # Take the second last axis, as this is either of shape (nVecs, 3), or (nReplicates, nVecs, 3).
            return np.take(self.A_J, ind, axis=-2)
    
    def transform_D(self):
        """
        Returns pair of floats (Dpar, Dperp) from the internal Diso, aniso representation.
        """
        tmp=3.0*self.D[0]/(2.0+self.D[1])
        return self.D[1]*tmp, tmp
    
    def update_D_coefficients(self):
        """
        Computes the 3 axisymmetric D-coefficients associated with the D_rot ellipsoid.
        """
        self.D_J = D_coefficients_symmtop( self.transform_D() )
        #return np.array( [5*Dperp+Dpar, 2*Dperp+4*Dpar, 6*Dperp] )
        
    def get_D_coefficients(self):
        return self.D_J
        
    def calc_Jomega_one(self, omega, CtModel, ind):
        D_J=self.get_D_coefficients() ; A_J=self.get_A_coefficients(ind)
        Jmat = _do_Jsum(omega, CtModel.zeta*CtModel.S2*A_J, D_J)
        for j in range(CtModel.nComps):
            Jmat += _do_Jsum(omega, CtModel.zeta*CtModel.C[j]*A_J, D_J+1./CtModel.tau[j])
        return Jmat

    def calc_Jomega_byName(self, omega, CtModel):
        ind = self.get_names().index(CtModel.name)
        return calc_Jomega_one(omega, CtModel, ind)
    
    def calc_Jomega(self, omega, Autocorrs, bSearch=False):
        #def J_combine_symmtop_exp_decayN(om, v, Dpar, Dperp, S2, consts, taus):
        """
        Calculates the J value for combining a symmetric-top anisotropic tumbling with
        a fitted internal autocorrelation C(t), where
        C(t) = S2 + Sum{ consts[i] * exp ( -t/tau[i] }
        thus this allows fits to multiple time constants in C(t).
        This function supports giving multiple vectors at once, of the form v.shape=(L,M,N,...,3)
        
        Assumes that the vectors list in A coefficients is identical to the CtModels list.
        
        If this cannot be assumed, e.g. where the CtModels are a subset of the vectors, give bSearch as an argument.
        """
        #v can be given as an array, with the X/Y/Z cartesian axisin the last position.
        if self.D_J is None:
            self.update_D_coefficients()
        if self.A_J is None:
            self.update_A_coefficients()
        #D_J=self.get_D_coefficients() ; A_J=self.get_A_coefficients()
        if bSearch:
            sh = list(self.vecXH.shape) ; sh[0] = Autocorrs.nModels ; sh[-1] = len(omega)
            Jmat = np.zeros( sh )
            for model in Autocorrs.model.values():
                Jmat[...,i,:] = self.calc_Jomega_byName(omega, model)
        else:
            sh = list(self.vecXH.shape) ; sh[-1] = len(omega)
            Jmat = np.zeros( sh )
            # This is the second last axis
            for i, model in enumerate( Autocorrs.model.values() ):
                Jmat[...,i,:] = self.calc_Jomega_one(omega, model, i)
                #Jmat[i] = _do_Jsum(omega, model.zeta*model.S2*A_J[i], D_J)
                #for j in range(model.nComps):
                #    Jmat[i] += _do_Jsum(omega, model.zeta*model.C[j]*A_J[i], D_J+1./model.tau[j])
        
        return Jmat
        #return _do_Jsum( S2*A_J, D_J) + np.sum([ _do_Jsum(CtModel.C[i]*A_J, D_J+1./CtModel.tau[i]) for i in range(CtModel.nComps)) ])

    def calc_Jomega_rigid(self, omega):
        D_J=self.get_D_coefficients()
        A_J=self.get_A_coefficients()
        return A_J*D_J/(np.power(D_J,2.0)+np.power(omega,2.0))

# = = = spin relaxation = = =

class spinRelaxationBase:
    """
    This meta class handles properties specific to a single NMR experiment, e.g. R1, R2, R1rho, hetNOE.
    Invoking obj.eval(frequencies) should return the computations.
    It is not aware of, per se, of the exact magnetic fields and other properties.
    """    
    def __init__(self, name, timeUnit='ps', angFreq=None, globalRotDif=None, localCtModels=None):
        self.name=name
        self.values = None
        self.errors = None
        self.timeUnit = timeUnit
        self.time_fact = _return_time_fact(self.timeUnit)
        self.angFreq   = angFreq       
        self.globalRotDif  = globalRotDif
        self.localCtModels  = localCtModels
        self.check_consistency()
        
    def check_consistency(self):
        if (not self.angFreq is None) and (not isinstance(self.angFreq, angularFrequencies)):
            print("= = = ERROR in initialisation, angular freq is not of the correct class!", file=sys.stderr)
            sys.exit(1)
        if (not self.globalRotDif is None) and (not isinstance(self.globalRotDif, globalRotationalDiffusion_Base)):
            print("= = = ERROR in initialisation, global rotdif model is not of the correct class!", file=sys.stderr)
            sys.exit(1)
        if (not self.localCtModels is None) and (not isinstance(self.localCtModels, fitCt.autoCorrelations)):
            print("= = = ERROR in initialisation, global rotdif model is not of the correct class!", file=sys.stderr)
            sys.exit(1)

    def set_angular_frequency(self, wObj):
        self.angFreq  = wObj
        
    def set_global_rotdif(self, DObj):
        self.globalRotDif  = DObj
    
    def set_local_Ctmodel(self, CtModel):
        self.localCtModels  = None
    
    def set_magnetic_field(self, fieldStrength, fieldUnit ):
        self.angFreq.set_magnetic_field( fieldStrength, fieldUnit )
    
    def get_magnetic_field(self):
        return self.angFreq.get_magnetic_field()

    def get_name(self):
        return self.name
    
    def get_description(self):
        return "%s Experiment at %sT over %i vectors" % (self.get_name(), self.get_magnetic_field(), self.localCtModels.nModels)

    #def calc_Jomega_one(self, ind):
    #    return self.globalRotDif.calc_Jomega_one(self.angFreq.omega, self.localCtModels.model[i], ind )
    
    def calc_Jomega(self, ind=None):
        if ind is None:
            return self.globalRotDif.calc_Jomega(self.angFreq.omega, self.localCtModels )
        else:
            return self.globalRotDif.calc_Jomega_one(self.angFreq.omega, self.localCtModels.model[ind], ind )

    def report(self):
        print("Name:", self.name)
        if not self.angFreq is None:
            print("Angular frequency information...")
            self.angFreq.report()
        else:
            print("No angular frequency information...")
            
        if not self.globalRotDif is None:
            print("Global tumbling information...")
            self.globalRotDif.report()
        else:
            print("No global tumbling information.")
            
        if not self.localCtModels is None:
            print("Local C(t)-model information...")
            self.localCtModels.report()
        else:
            print("No local C(t)-model information.")
    
    def func(self, f_DD, f_CSA, J):
        """
        Inner function that defines the relationship between experimental observable and J(omega).
        """
        return None
    
    def eval(self, CSAvalue=None):
        """
        The core conversion step from J(omega) to the experimental observable.
        The dimensions of array J implies whether any averaging is done, where:
          1 - Angular frequencies omega is the only dimension involved, of 5 frequencies.
          2 - ( names, omega ) for the isotropic multiple peak computation.
          3 - ( names, vectors, omega ) for the axisymmetric multiple peak computation. Averaging is done here,
              as it nominally represents the contribution of different conformers that share similar enough chemical environments
              to be considered in the same resonance peak.
        
        The base class has no evaluation step. This is replaced in each subclass.
        """
        print("ERROR: spinRelaxationBase.eval() is not meant to be invoked.", file=sys.stderr)
        return None
    
    def set_zeta(self, zeta):
        self.localCtModels.set_zeta(zeta)
            
    def get_zeta(self):
        return self.localCtModels.get_zeta()

    def update_values(self, values, errors=None, ind=None):
        if ind is None:
            self.values = values
            if not errors is None:
                self.errors = errors
        else:
            self.values[i] = values
            if not errors is None:
                self.errors[i] = errors
            
    def check_and_calculate_average(self, arr, ind=None):
        nDim=len(arr.shape)
        e = None
        if self.globalRotDif.axisAvg is None:
            return arr, e
        elif ind is None:
            v = np.average(arr, axis=0, weights=self.globalRotDif.vecWeights)
            e = np.sqrt( np.average( (arr-v)**2.0, axis=0, weights=self.globalRotDif.vecWeights) )
            return v, e
        else:
            v = np.average(arr, weights=self.globalRotDif.vecWeights[:,ind])
            e = np.sqrt( np.average( (arr-v)**2.0, weights=self.globalRotDif.vecWeights[:,ind]) )
            return v, e

    def print_metadata(self, style='stdout', fp=sys.stdout):
        if style=='stdout':
            print("# %s" % self.get_description(), file=fp)
        elif style=='xmgrace':
            print('# Type %s' % self.name, file=fp)
            print('# NucleiA %s' % (self.angFreq.gA.isotope), file=fp)
            print('# NucleiB %s' % (self.angFreq.gB.isotope), file=fp)
            f = self.angFreq.get_magnetic_field(unit='MHz')
            print('# Frequency %g %s' % (f, 'MHz'), file=fp)

    def print_values(self, style='stdout', fp=sys.stdout):
        names = self.localCtModels.get_names()
        if style=='stdout':
            if self.errors is None:
                for x,y in zip(names, self.values):
                    print("%s %g" % (x,y), file=fp)
            else:
                for x,y,dy in zip(names, self.values, self.errors):
                    print("%s %g %g" % (x,y,dy), file=fp)                
        elif style=='xmgrace':

            if self.errors is None:
                for x,y in zip(names, self.values):
                    print("%s %g" % (x,y), file=fp)
            else:
                for x,y,dy in zip(names, self.values, self.errors):
                    print("%s %g %g" % (x,y,dy), file=fp)                
            print("&", file=fp)
            
    def calc_chisq(self, Target, dTarget=None, indices=None):
        v = self.values        
        e = self.errors
        if not indices is None:
            v = v[indices]
            if not e is None:
                e = e[indices]

        if (not e is None) and (not dTarget is None):
            return np.mean(np.square(v-Target)/(np.square(dTarget)+np.square(e)))
        elif e is None:
            return np.mean(np.square(v-Target)/np.square(dTarget))
        elif dTarget is None:
            return np.mean(np.square(v-Target)/np.square(e))
        else:
            return np.mean(np.square(v-Target))

class spinRelaxationR1(spinRelaxationBase):
    """
    Derivative class for R1
    """
    def func(self, f_DD, f_CSA, J):
        # = = = Incorporate Broadcasting rule for multiCSA fitting. Assume all inputs are numpy arrays even if they are single scalars.
        # Note: since J is in the units of inverse time, data needs to be converted back to s^-1
        iOmX = 1 ; iOmH = 3
        return self.time_fact*( f_DD*( J[...,iOmH-iOmX] + 3*J[...,iOmX] + 6*J[...,iOmH+iOmX] ) \
                                + f_CSA*J[...,iOmX] )                       

    def eval(self, ind=None, bVerbose=False):
        """
        The core conversion step from J(omega) to the experimental observable.
        The dimensions of array J implies whether any averaging is done, where:
          1 - Angular frequencies omega is the only dimension involved, of 5 frequencies.
          2 - ( nPeaks/names, omega ) for the isotropic multiple peak computation.
          3 - ( nPeaks/names, nReplicatesVectors, omega ) for the axisymmetric multiple peak computation.
              Averaging over multiple vectors is done here and not at the A_J level,
              as it nominally represents the contribution of different conformers that share similar enough chemical environments
              to be considered in the same resonance peak.
        
        When an index is given, compute only the nth-index of axis-0, corresponding to a particular residue
        """
        J = self.calc_Jomega(ind) 
        # Note: since J is in the units of inverse time, data needs to be converted back to s^-1
        f_DD  = self.angFreq.get_factor_DD()
        f_CSA = np.array(self.angFreq.get_factor_CSA(ind))
        temp = self.func(f_DD, f_CSA, J)
        if bVerbose:
            print("debug R1 shapes pre-average: CSA %s , J %s , R1 %s " % (f_CSA.shape, J.shape, temp.shape))
        v,e = self.check_and_calculate_average(temp, ind)
        self.update_values(v,e, ind=ind)
        return v
        
class spinRelaxationR2(spinRelaxationBase):
    """
    Derivative class for R2
    """
    def func(self, f_DD, f_CSA, J):
        # = = = Incorporate Broadcasting rule for multiCSA fitting. Assume all inputs are numpy arrays even if they are single scalars.
        # Note: since J is in the units of inverse time, data needs to be converted back to s^-1
        iOmX = 1 ; iOmH = 3
        return self.time_fact*( 0.5*f_DD*( 4*J[...,0] + J[...,iOmH-iOmX] + 3*J[...,iOmX] + 6*J[...,iOmH+iOmX] + 6*J[...,iOmH] ) \
                                + 1.0/6.0*f_CSA*(4*J[...,0] + 3*J[...,iOmX]) )
    
    def eval(self, ind=None, bVerbose=False):
        J = self.calc_Jomega(ind) 
        f_DD  = self.angFreq.get_factor_DD()
        f_CSA = np.array(self.angFreq.get_factor_CSA(ind))
        temp = self.func(f_DD, f_CSA, J)
        if bVerbose:
            print("debug R2 shapes pre-average: CSA %s , J %s , R2 %s " % (f_CSA.shape, J.shape, temp.shape))
        v,e = self.check_and_calculate_average(temp, ind)
        self.update_values(v,e, ind=ind)
        return v
        
class spinRelaxationNOE(spinRelaxationBase):
    """
    Derivative class for NOE
    """
    def _eval_R1(self, f_DD, f_CSA, J, ind=None):
        iOmX = 1 ; iOmH = 3
        temp = self.time_fact*( f_DD*( J[...,iOmH-iOmX] + 3*J[...,iOmX] + 6*J[...,iOmH+iOmX] ) \
                                + f_CSA*J[...,iOmX] )    
        v,e = self.check_and_calculate_average(temp, ind)
        return v
        
    def func(self, f_DD, R1, J):
        # = = = Incorporate Broadcasting rule for multiCSA fitting. Assume all inputs are numpy arrays even if they are single scalars.
        # Note: since J is in the units of inverse time, data needs to be converted back to s^-1
        iOmX = 1 ; iOmH = 3
        return 1.0 + self.time_fact * self.angFreq.gB.gamma/(self.angFreq.gA.gamma*R1) * f_DD*(6*J[...,iOmH+iOmX] - J[...,iOmH-iOmX])
    
    def eval(self, ind=None, R1=None, bVerbose=False):
        J = self.calc_Jomega(ind) 
        f_DD  = self.angFreq.get_factor_DD()
        if R1 is None:
            if bVerbose:
                print("DEBUG: ...calculating R1 within NOE.")
            f_CSA = np.array(self.angFreq.get_factor_CSA(ind))
            R1 = self._eval_R1(f_DD, f_CSA, J, ind)
        temp = self.func(f_DD, R1, J)
        if bVerbose:
            print("debug NOE shapes pre-average: R1 %s , J %s , NOE %s " % (R1.shape, J.shape, temp.shape))
        v,e = self.check_and_calculate_average(temp, ind)
        self.update_values(v,e, ind=ind)
        return v
    
class spinRelaxationExperiments:
    """
    This is the handler class to manage multiple independent experiments and link to its associated
    computational components.
    Shapes of arrays:
    - self.data( nExperiments, [x,y,dy] )
    """
    #listDefinedTypes=['R1','R2','NOE']
    
    def __init__(self, globalRotDif=None, localCtModels=None):
        self.num=0
        self.spinrelax=[]
        self.data=[]
        self.globalRotDif=globalRotDif
        self.localCtModels=localCtModels
        self.mapModelNames=[]
        self.bFitInitialised=False
    
    def add_experiment(self, fileName, bIgnoreErrors=False):
        """
        Add one experiment into the list.
        Meta information such as the type of experiment and the nuclei involved are given in the header section.
        NucleiA is the heavy nuclei
        # Type NOE
        # NucleiA   15N
        # NucleiB    1H
        # Frequency 600.133
        1 0.5 0.05
        2 0.9 0.03
        ...
        """
        strType=None ; nucleiA=None; nucleiB=None; freq=None
        names=[] ; values=[] ; errors=[]
        for line in open(fileName,'r'):
            if len(line) == 0:
                continue
            l = line.split()                
            if line[0] == '#' or line[0] == '@':
                # = = = HEADER section
                if l[1] == 'Type':
                    strType=l[2]
                elif l[1] == 'NucleiA':
                    nucleiA=l[2]
                elif l[1] == 'NucleiB':
                    nucleiB=l[2]
                elif l[1] == 'Frequency':
                    freq=float(l[2])
                continue
            # = = = Data section.
            if bIgnoreErrors:
                if len(1)==1:
                    continue
            elif len(l)==1 or len(l)>3:
                print("ERROR in spinRelaxationExperiments.add_experiment(): data line does not obey expected conventions of 2 or 3 space-separated values!", l, file=sts.stderr)
                sys.exit()                
            names.append(l[0])
            values.append(float(l[1]))
            if len(l)>2:
                errors.append(float(l[2]))
            else:
                errors.append(None)

        # = = Sanity Checks
        if strType is None or nucleiA is None or nucleiB is None or freq is None:
            print("ERROR in spinRelaxationExperiments.add_experiment(): not all metadata has been read! "
                  "Require: Type, NucleiA, NucleiB, Frequency", file=sys.stderr)
            return
        names=np.array(names)
        values=np.array(values, dtype=float)
        nErrors = np.sum([ x is None for x in errors])
        if nErrors == len(errors):
            errors = None
        elif nErrors > 0:
            print("ERROR in spinRelaxationExperiments.add_experiment(): either all entries must have uncertainties or none!", file=sys.stderr)
            print("number of values with uncertainties: %i of %i" % (nErrors, len(errors)), file=sys.stderr)
            sys.exit(1)
        
        wObj = angularFrequencies(nucleiA=nucleiA, nucleiB=nucleiB, fieldStrength=freq, fieldUnit='MHz')
        if strType=='R1':
            obj = spinRelaxationR1(strType, angFreq=wObj, globalRotDif=self.globalRotDif, localCtModels=self.localCtModels)
        elif strType=='R2':
            obj = spinRelaxationR2(strType, angFreq=wObj, globalRotDif=self.globalRotDif, localCtModels=self.localCtModels)
        elif strType=='NOE':
            obj = spinRelaxationNOE(strType, angFreq=wObj, globalRotDif=self.globalRotDif, localCtModels=self.localCtModels)
        self.num += 1
        self.spinrelax.append(obj)
        self.data.append(dict(names=names, y=values, dy=errors))
    
    def report(self):
        print("QM vibration correction factor: %g" % self.get_zeta() )
        if not self.globalRotDif is None:
            self.globalRotDif.report()
        else:
            print("No global tumbling information loaded.")
        if not self.localCtModels is None:
            self.localCtModels.report()
        else:
            print("No local C(t) model information loaded.")
        print("Number of experiments:", self.num )
        for i in range(self.num):
            print("...expt ", i)
            self.spinrelax[i].report()
            if not self.data[i]['dy'] is None:
                print("Data array has %i names, %i values, %i errors" % (len(self.data[i]['names']),
                                                                         len(self.data[i]['y']),
                                                                         len(self.data[i]['dy'])))
            else:
                print("Data array has %i names, %i values," % (len(self.data[i]['names']),
                                                               len(self.data[i]['y'])))

    def map_experiment_peaknames_to_models(self):
        """
        This function sets up the numpy index-slicing for mapping experimental and computed peak identities.
        It is planned to support names that have characters and not just integers representing resIDs.
        The intended use is to run CtModels.model[mapnames[i]]
        """
        self.mapModelNames=[]
        if self.localCtModels is None:
            print("ERROR in spinRelaxationExperiments.map_peak_names: need a local C(t) model to map experimental peak names!", file=sys.stderr)
            sys.exit(1)
        namesCt   = self.localCtModels.get_names()
        if not self.globalRotDif is None and self.globalRotDif.bVecs:
            namesRotdif = self.globalRotDif.get_names()
            if not gm.list_identical(namesCt,namesRotdif):
                print("ERROR in spinRelaxationExperiments.map_peak_names: local C(t) model and global rotational-diffusion model do not have matching peak names!", file=sys.stderr)
                print( "...local:",  namesCt, file=sys.stderr )
                print( "...global:", namesRotdif, file=sys.stderr )
                sys.exit(1)
        
        #= = = Use localCt as the authoritive version.
        for i in range(self.num):
            tmp = gm.list_get_map( namesCt, self.data[i]['names'], bValue=False )
            self.mapModelNames.append(tmp)
        
    def initialise_CSA_array(self, CSAValues):
        """
        Searches through all experiments and 
        """
        for sp in self.spinrelax:
            sp.angFreq.initialise_CSA_array(len(CSAValues), CSAValues)

    def update_CSA_array(self, csa, ind=None):
        for sp in self.spinrelax:
            sp.angFreq.update_CSA_array(csa, ind=None)
        
    def eval_all(self, bVerbose=False):
        R1Values=None
        for sp in self.spinrelax:
            if bVerbose:
                print('...evaluating experiment %s at %g T.' % (sp.name,sp.angFreq.B0))
            sp.eval(bVerbose=bVerbose)
            #if sp.name=='R1':
            #    R1Values = sp.eval()
            #elif sp.name=='NOE':
            #    sp.eval(R1=R1Values)
            #else:
            #    sp.eval()
            #print(v.shape, v[0])

    def print_all_values(self, style='stdout',fp=sys.stdout):
        for sp in self.spinrelax:
            sp.print_values(style,fp)
            if style=='stdout':
                print('', file=fp)

    def set_global_Diso(self, Diso):
        self.globalRotDif.set_Diso(Diso)
    def get_global_Diso(self):
        return self.globalRotDif.get_Diso()

    def set_global_Daniso(self, Daniso):
        self.globalRotDif.set_Daniso(Daniso)
    def get_global_Daniso(self):
        return self.globalRotDif.get_Daniso()

    def set_global_zeta(self, zeta):
        self.localCtModels.set_zeta(zeta)
    def get_global_zeta(self):
        return self.localCtModels.get_zeta()

    def set_all_globalCSA(self, csa):
        for sp in self.spinrelax:
            sp.angFreq.gA.set_csa(csa)
    def get_first_globalCSA(self):
        return self.spinrelax[0].angFreq.gA.get_csa()
            
    # = = = = Optimisation related functions and classes.
    listOptimisationVariables=['Diso','Daniso','CSA','zeta','resCSA']
    
    def parse_optimisation_params(self,listOpts):
        self.bFitInitialised=False
        self.listUpdateVariables=[]
        self.listStepSizes=[]
        self.listSetFunctions=[]
        self.listGetFunctions=[]
        for o in listOpts:
            if o not in spinRelaxationExperiments.listOptimisationVariables:
                _BAIL( "parse_optimisation_params",
                        "Optimisation variable %s not found in list!\n"
                        "Possibilities are: %s "% (o, spinRelaxationExperiments.listOptimisationVariables) )
            if o == 'Diso':
                self.listSetFunctions.append(self.set_global_Diso)
                self.listGetFunctions.append(self.get_global_Diso)
                self.listStepSizes.append(1e-5)
            if o == 'Daniso':
                self.listSetFunctions.append(self.set_global_Daniso)
                self.listGetFunctions.append(self.get_global_Daniso)
                self.listStepSizes.append(0.1)
            elif o == 'zeta':
                self.listSetFunctions.append(self.set_global_zeta)
                self.listGetFunctions.append(self.get_global_zeta)
                self.listStepSizes.append(0.1)
            elif o == 'CSA':
                self.listSetFunctions.append(self.set_all_globalCSA)
                self.listGetFunctions.append(self.get_first_globalCSA)
                self.listStepSizes.append(1e-5)
            self.listUpdateVariables.append(o)
        self.bFitInitialised=True
        
    def perform_optimisation(self):
        if not self.bFitInitialised:
            _BAIL("perform_fit","You must first run parse_optimisation_params to tell the script what to optimise.")
        from scipy.optimize import fmin_powell      
        fminOut = fmin_powell(optimisation_loop_inner_function,
                              x0=self.optimisation_loop_get_globals(),
                              direc=self.optimisation_loop_get_direc(),
                              args=( self, '1' ),
                              full_output=True)
        #out = fmin_powell(optfunc_R1R2NOE_Diso, x0=DisoOpt, direc=[0.1*DisoOpt], args=(...), full_output=True )       
        print( "= = = Optimisation complete over variables: %s" % self.optimisation_loop_get_param_names() )
        print( fminOut )
        
    def optimisation_loop_get_param_names(self):
        return self.listUpdateVariables
    
    def optimisation_loop_get_direc(self):
        """
        Should be optimised later as some global variables are semi-direct competitors and so
        it's better to search diagonally (1,1),(1,-1) than (1,0),(0,1).
        Diso and CSA are such competitors.
        """
        n=len(self.listStepSizes)
        out=np.zeros( (n,n) )
        for i in range(n):
            out[i,i]=self.listStepSizes[i]
        return out
    
    def optimisation_loop_get_globals(self):
        out=[]
        for func in self.listGetFunctions:
            out.append( func() )
        return out
            
    def optimisation_loop_set_globals(self, vNew):
        for func, v in zip(self.listSetFunctions, vNew):
            func(v)
        
    def calc_chisq(self):
        chisq=0.0
        for i, sp in enumerate(self.spinrelax):
            chisq += sp.calc_chisq( self.data[i]['y'], self.data[i]['dy'], self.mapModelNames[i] )
        return chisq/self.num
    #    self.eval_all()
    #    chiSq=0.0
    #    for i, sp in enumerate(self.spinrelax):
    #        self.data[i]['y'] - sp.values[self.mapModelNames[i]]

def optimisation_loop_inner_function(params, *args):
    objExpts=args[0]
    objExpts.optimisation_loop_set_globals(params)
    objExpts.eval_all(bVerbose=False)
    chisq = objExpts.calc_chisq()
    print("    ....optimisation step. Params: %s chsqi: %g" % (params, chisq) )
    return chisq
    
# = = = = Old spin relaxation class below = = =
class diffusionModel:
    """
    The base global diffusion model that contains the global rotational diffusion tensor information,
    and potential vector distributions.

    Its main purpose in SpinRelax is to obscure the details of computing D and D coefficients and
    containerise its relevant parameters such as D_iso, D_aniso, and q_orient.

    The rotational diffusion model determines which function should be called when J(omega) is to be computed.
    """
    def __init__(self, model, *args):
        self.vecXH = None
        self.timeUnit=args[0]
        self.time_fact=_return_time_fact(self.timeUnit)
        if   model=='direct_transform':
            # Dummy entry for diffusion model with np.nan to ensure errors.
            self.name='direct_transform'
            self.D=np.nan
        elif model=='rigid_sphere_T':
            self.name='rigid_sphere'
            self.D=1.0/(6.0*float(args[1]))
            self.D_coeff=self.D
            self.D_delta=0.0
            #self.A_coeff=1.0
        elif model=='rigid_sphere_D':
            self.name='rigid_sphere'
            self.D=float(args[1])
            self.D_coeff=self.D
            self.D_delta=0.0
            #self.A_coeff=1.0
        elif model=='rigid_symmtop_Dref':
            #Dperp = 3.*Diso/(2+aniso)
            #Dpar  = aniso*Dperp
            self.name='rigid_symmtop'
            self.D=np.zeros(2)
            self.D[1]=3.0*args[1]/(2.0+args[2])
            self.D[0]=args[2]*self.D[1]
            self.D_coeff=D_coefficients_symmtop(self.D)
            self.D_delta=0.0
            # Convention: Diso, Dani --> Dpar, Dperp
        elif model=='rigid_symmtop_D':
            self.name='rigid_symmtop'
            self.D=np.zeros(2)
            self.D[0]=args[1]
            self.D[1]=args[2]
            self.D_coeff=D_coefficients_symmtop(self.D)
            self.D_delta=0.0
            # Convention: Dpar, Dperp
        elif model=='rigid_symmtop_T':
            self.name='rigid_symmtop'
            self.D=calculate_Dglob( (args[1], args[2], args[3]) )
            self.D_coeff=D_coefficients_symmtop(self.D)
            self.D_delta=0.0
            # Convention:
        elif model=='rigid_ellipsoid_D':
            self.name='rigid_ellipsoid'
            self.D=np.zeros(3)
            self.D[0]=args[1]
            self.D[1]=args[2]
            self.D[2]=args[3]
            # Convention: D0 <= D1 <= D2
            self.D_coeff, self.D_delta = D_coefficients_ellipsoid(D, True)

    def obtain_A_coeff(self, v):
        if model=='rigid_sphere':
            return 1.0
        if model=='rigid_symmtop':
            return A_coefficients_symmtop(v, bProlate=(self.D[0]>self.D[1]))
        if model=='rigid_ellipsoid':
            return A_coefficients_ellipsoid(v, self.D_delta, True)

    def set_time_unit(self, tu):
        old = self.time_fact
        self.time_fact = _return_time_fact(tu)
        mult = self.time_fact/old
        self.D *= mult
        self.D_coeff *= mult
        self.D_delta   *= mult

    def change_Diso(self, Diso):
        if self.name=='rigid_sphere':
            self.D=Diso
            self.D_coeff=self.D
            self.D_delta=0.0
        elif self.name=='rigid_symmtop':
            tmp=self.D[0]/self.D[1]
            self.D[1]=3.0*Diso/(2.0+tmp)
            self.D[0]=tmp*self.D[1]
            self.D_coeff=D_coefficients_symmtop(self.D)
            self.D_delta=0.0
        elif self.name=='rigid_ellipsoid':
            print( "= = ERROR: change_Diso for fully anisotropic models, not implemented.", file=sys.stderr )
            sys.exit(1)

    def import_vecs(self, v):
        self.vecXH = v
        self._sanitise_vecs()

    def _sanitise_vecs(self):
        if type(self.vecXH) != np.ndarray:
            self.vecXH=np.array(self.vecXH)
        sh=self.vecXH.shape
        if sh[-1] != 3:
            print( "= = ERROR in computation of A and D coefficients (spectral_densities.py): input v does not have 3 as its final dimension!", file=sys.stderr )
            sys.exit(2)

    def calc_Jomega(self, omega, autoCorr):
        # = = = The parent class does not apply any global diffusion.
        return J_direct_transform(omega, autoCorr)


class relaxationModel:
    """
    Help for class relaxationModel:
    This is the overall handling class used to compute spin relaxations from trajectories.
    It collects a number of smaller classes that are responsible for functionally distinct sub-components,
    i.e.:
    - the NMR measurements, which handles frequencies, NH types, and spins.
    - the molecule, which handles sequence information, domain definitions, rotational diffusion models, vector data
    
    This overall class contains the following functions:
    - the computing and fitting procedures that need the above sub-classes.
    
    Attributes:
        bond - Name of the intended vector bond, eg 'NH'
        B_0  - Background magnetic field, in Teslas.
        timeUnit - Units of time used for the freqency, such as 'ns', or 'ps'.
        rotdifModel - The Rotational Diffusion model used to define spectral densitiies J(omega)
    """

    # = = = Static class variables = = =
    # Default indexing to create the five NMR-component frequencies
    # J(0), J(wX), J(wH-wX), J(wH), J(wH+wX)
    # In ascending order. This is used to obtain relaxation properties from a five-value J(function)
    iOmX = 1
    iOmH = 3

    def __init__(self, bondType, B_0):
        # Parameters associated with units
        self.omega=None
        self.timeUnit='ns'
        self.time_fact=_return_time_fact(self.timeUnit)
        self.distUnit='nm'
        self.dist_fact=_return_dist_fact(self.distUnit)

        # Atomic parameters.
        self.bondType = bondType
        self.B_0  = B_0
        if   bondType=='NH':
            self.gH  = gyromag('1H')
            self.gX  = gyromag('15N')
            # = = Question use conventional 1.02, or 1.04 to include librations, according to Case, J Biomol. NMR, 1999? = = =
            self.rXH = 1.02e-1
            #self.rXH = 1.04e-1
        elif bondType=='CH':
            self.gH  = gyromag('1H')
            self.gX  = gyromag('13C')
            # Need to update bond length for this atoms
            self.rXH = 1.02e-1
        else:
            print( "= = ERROR in relaxationModel: wrong bondType definition! = =" % bondType, file=sys.stderr )
            sys.exit(1)

        # relaxation model. Note in time units of host object.
        self.set_rotdif_model('rigid_sphere_T', 1.0)
        self.set_freq_relaxation()

    def set_B0(self, B_0):
        self.B_0 = B_0

    def set_time_unit(self, tu):
        old=self.time_fact
        self.time_fact = _return_time_fact(tu)
        # Update all time units can measurements.
        self.timeUnit=tu
        mult = self.time_fact / old
        self.omega *= mult
        self.rotdifModel.set_time_unit(tu)
        #self.gH.set_time_unit(tu) - leave immune to time unit changes!
        #self.gX.set_time_unit(tu)

    def set_freq_relaxation(self):
        """
        This function sets the 5 frequencies of nuclei at the instance's magnetic field,
        in the following order:
            - 0.0 , omega_X, omega_H-omega_X, omega_H, omega_H+omega_X.
        This order will be used for calculating J(w) and relaxaion values.
        """
        self.num_omega=5
        self.omega=np.zeros(self.num_omega)
        iOmH = 3 ; # indexing for the frequencies.
        iOmX = 1 ; #
        # First determine the frequencies omega and J from given inputs.
        self.omega[iOmH] = -1.0*self.gH.gamma*self.B_0*self.time_fact
        self.omega[iOmX] = -1.0*self.gX.gamma*self.B_0*self.time_fact
        self.omega[iOmH-iOmX] = (self.omega[iOmH]-self.omega[iOmX])
        self.omega[iOmH+iOmX] = (self.omega[iOmH]+self.omega[iOmX])

    def print_frequencies(self):
        """
        Report the ferquency order as a debug.
        """
        print("# Order of frequencies for %s - %s relaxation:" % (self.gX.isotope, self.gH.isotope) )
        print("# 0  iOmX    iOmH-iOmX   iOmH    iOmH+iOmX" )
        print( self.omega )

    def set_freq_defined(self, wmin, wmax, wstep):
        self.omega=np.arange(wmin, wmax, wstep)
        self.num_omega=len(self.omega)

    def set_rotdif_model(self, model, *args):
        """
        Define the relaxation model for this object, taking as arguments the global parameters necessary
        to define each model. Available models are:
            - rigid_sphere_T, tau_iso
            - rigid_sphere_D, D_iso
            - rigid_symmtop_D, D_par, D_perp
            - rigid_ellipsoid_D, Dx, Dy, Dz
        """
        self.rotdifModel=diffusionModel(model, self.timeUnit, *args)

    def get_Jomega(self, vNH):
        """
        Calculate the spectral density function for a given set of unit vectors and the current model.
        """
        num_vecs = len(vNH)
        J = np.zeros( (num_vecs, self.num_omega) )
        for i in range(num_vecs):
            J[i] = function_to_be_written(vNH[i])
        return 'Not composed'

    def get_relax_from_J(self, J, CSAvalue=None):
        """
        The maths behind this is:
        f_DD  = 0.10* (mu_0*hbar/4.0/pi)**2 * gamma_15N**2 * gamma_1H**2 * r_NH**-6.0
        f_CSA = 2.0/15.0 * gamma_15N**2 * B_0 **2 * DeltaSig_15N**2
        mu_0 = 4*pi*1e-7      ; # m   kg s^-2 A-2
        hbar = 1.0545718e-34  ; # m^2 kg s^-1
        pi   = 3.14159265359
        gamma_1H  = 267.513e6  ; # rad s^-1 T^-1
        gamma_15N = -27.116e6  ; # rad s^-1 T^-1
        omega_15N = - gamma_15N * B_0 .
        r_NH = 1.02e-10 ;# m
         (mu_0*hbar/4.0/pi)**2 m^-1 s^2 is the 10^-82 number below. f_DD and f_CSA are maintained in SI units.
        """
        iOmX = 1; iOmH = 3

        f_DD = 0.10 * 1.1121216813552401e-82*self.gH.gamma**2.0*self.gX.gamma**2.0 *(self.rXH*self.dist_fact)**-6.0

        if CSAvalue is None:
            f_CSA = 2.0/15.0 * self.gX.csa**2.0 * ( self.gX.gamma * self.B_0 )**2
        else:
            f_CSA = 2.0/15.0 * CSAvalue**2.0 * ( self.gX.gamma * self.B_0 )**2		

        # Note: since J is in the units of inverse time, data needs to be converted back to s^-1
        R1 = self.time_fact*( f_DD*( J[iOmH-iOmX] + 3*J[iOmX] + 6*J[iOmH+iOmX] ) + f_CSA*J[iOmX] )
        R2 = self.time_fact*( 0.5*f_DD*( 4*J[0] + J[iOmH-iOmX] + 3*J[iOmX] + 6*J[iOmH+iOmX] + 6*J[iOmH] ) + 1.0/6.0*f_CSA*(4*J[0] + 3*J[iOmX]) )
        NOE = 1.0 + self.time_fact * self.gH.gamma/(self.gX.gamma*R1) * f_DD*(6*J[iOmH+iOmX] - J[iOmH-iOmX])

        return R1, R2, NOE

    def get_relax_from_J_simd(self, J, axis=-1, CSAvalue=None):
        iOmX = 1; iOmH = 3

        f_DD = 0.10 * 1.1121216813552401e-82*self.gH.gamma**2.0*self.gX.gamma**2.0 *(self.rXH*self.dist_fact)**-6.0        
        if CSAvalue is None:
            f_CSA = 2.0/15.0 * self.gX.csa**2.0 * ( self.gX.gamma * self.B_0 )**2
        else:
            f_CSA = 2.0/15.0 * CSAvalue**2.0 * ( self.gX.gamma * self.B_0 )**2
                    
        if axis==-1:
            R1 = self.time_fact*( f_DD*( J[...,iOmH-iOmX] + 3*J[...,iOmX] + 6*J[...,iOmH+iOmX] ) + f_CSA*J[...,iOmX] )
            R2 = self.time_fact*( 0.5*f_DD*( 4*J[...,0] + J[...,iOmH-iOmX] + 3*J[...,iOmX] + 6*J[...,iOmH+iOmX] + 6*J[...,iOmH] ) + 1.0/6.0*f_CSA*(4*J[...,0] + 3*J[...,iOmX]) )            
            NOE = 1.0 + self.time_fact * self.gH.gamma/(self.gX.gamma*R1) * f_DD*(6*J[...,iOmH+iOmX] - J[...,iOmH-iOmX])
        elif axis==0:
            R1 = self.time_fact*( f_DD*( J[iOmH-iOmX,...] + 3*J[iOmX,...] + 6*J[iOmH+iOmX,...] ) + f_CSA*J[iOmX,...] )
            R2 = self.time_fact*( 0.5*f_DD*( 4*J[0,...] + J[iOmH-iOmX,...] + 3*J[iOmX,...] + 6*J[iOmH+iOmX,...] + 6*J[iOmH,...] ) + 1.0/6.0*f_CSA*(4*J[0,...] + 3*J[iOmX,...]) )
            NOE = 1.0 + self.time_fact * self.gH.gamma/(self.gX.gamma*R1) * f_DD*(6*J[iOmH+iOmX,...] - J[iOmH-iOmX,...])

        if False:
            # HASHTAG
            print( f_DD, f_CSA )
            print( self.time_fact )
            print( R2.shape, J.shape )
            print( np.mean(R2) )
            print( J[0] )
            sys.exit()            
            
        return R1, R2, NOE


    def _get_f_DD(self):
        return 0.10 * 1.1121216813552401e-82*self.gH.gamma**2.0*self.gX.gamma**2.0 *(self.rXH*self.dist_fact)**-6.0

    def _get_f_CSA(self):
        return 2.0/15.0 * self.gX.csa**2.0 * ( self.gX.gamma * self.B_0 )**2

    def get_R1(self, J):
        f_DD = _get_f_DD() ; f_CSA = _get_f_CSA()
        return self.time_fact*( f_DD*( J[iOmH-iOmX] + 3*J[iOmX] + 6*J[iOmH+iOmX] ) + f_CSA*J[iOmX] )

    def get_R2(self, J):
        f_DD = _get_f_DD() ; f_CSA = _get_f_CSA()
        return self.time_fact*( 0.5*f_DD*( 4*J[0] + J[iOmH-iOmX] + 3*J[iOmX] + 6*J[iOmH+iOmX] + 6*J[iOmH] ) + 1.0/6.0*f_CSA*(4*J[0] + 3*J[iOmX] ) )

    def get_NOE(self, J):
        f_DD = _get_f_DD() ; f_CSA = _get_f_CSA()
        return 1.0 + self.time_fact * self.gH.gamma/(self.gX.gamma*R1) * f_DD*(6*J[iOmH+iOmX] - J[iOmH-iOmX])

    def get_etaZ(self, J, beta=0.0):
        """
        Notation following that of Kroenke et al., JACS 1998. Eq. 2
        Here, beta is the angle (in radians) between the symmetry axis of the CSA tensor and the N-H bond.
        """
        # mu_0 hbar / 4_pi = hbar* 10^-7
        fact = -1.0545718e-41*self.gH.gamma*self.gX.gamma**2.0*(self.rXH*self.dist_fact)**-3.0 * self.B_0*self.gX.csa* 0.4
        return fact*(1.5*cos(beta)-0.5)*J[iOmX]

    def get_etaXY(self, J, beta=0.0):
        """
        Notation following that of Kroenke et al., JACS 1998.
        Here, beta is the angle (in radians) between the symmetry axis of the CSA tensor and the N-H bond.
        """
        fact = -1.0545718e-41*self.gH.gamma*self.gX.gamma**2.0*(self.rXH*self.dist_fact)**-3.0 * self.B_0*self.gX.csa* 0.4
        return fact/6.0*(1.5*cos(beta)-0.5)*( 4.0*J[0] + 3.0*J[iOmX] )

    def get_rho_from_J(self, J):
        """
        Taking Eq. 4 of Ghose, Fushman and Cowburn (2001), and define rho as a ratio of modified R1' and R2'
        that have high frequency components removed.
        """
        return J[self.iOmX]/J[0]

    def get_rho_from_J_simd(self, J, axis=-1):
        if axis == -1:
            return J[...,self.iOmX]/J[...,0]
        elif axis == 0:
            return J[self.iOmX,...]/J[0,...]

    def calculate_rho_from_relaxation(self, rvec, drvec=[] ):
        """
        Taking Eq. 4 of Ghose, Fushman and Cowburn (2001), calculate rho from R1, R2, and NOE directly,
        rather than from the spectral density J(omega). This is used to convert experimental measurements to rho.
        rvec is the triple of (R1, R2, NOE)
        Error is known to be bad.
        """
        if  drvec==[]:
            R1=rvec[0] ; R2=rvec[1] ; NOE=rvec[2]
            HF  = -0.2*(self.gX.gamma/self.gH.gamma)*(1-NOE)*R1
            R1p = R1 - 7.0*(0.921/0.87)**2.0*HF
            R2p = R2 - 6.5*(0.955/0.87)**2.0*HF
            return 4.0/3.0*R1p/(2.0*R2p-R1p)
        else:
            R1=rvec[0]  ;  R2=rvec[1]  ;  NOE=rvec[2]
            dR1=drvec[0] ; dR2=drvec[1] ; dNOE=drvec[2]
            HF  = -0.2*(self.gX.gamma/self.gH.gamma)*(1-NOE)*R1
            R1p = R1 - 7.0*(0.921/0.87)**2.0*HF
            R2p = R2 - 6.5*(0.955/0.87)**2.0*HF
            rho  = 4.0/3.0*R1p/(2.0*R2p-R1p)
            drho = 0
            print( "= = ERROR: drho calculation is not implemented!" )
            sys.exit(1)
            return (rho, drho)

# = = = = = = =
# End class definitions.
# Begin function definitions.
# = = = = = = =

def _BAIL( functionName, message ):
    """
    Universal failure mode message.
    """
    print( "= = ERROR in function %s : %s" % ( functionName, message), file=sys.stderr )
    sys.exit(1)    

def _return_time_fact(tu):
    if tu=='ps':
        return 1.0e-12
    elif tu=='ns':
        return 1.0e-9
    elif tu=='us':
        return 1.0e-6
    elif tu=='ms':
        return 1.0e-3
    elif tu=='s':
        return 1.0e-0
    else:
        print( "= = ERROR in object definition: invalid time unit definition!", file=sys.stderr )
        return

def _return_dist_fact(du):
    if du=='pm':
        return 1.0e-12
    elif du== 'A':
        return 1.0e-10
    elif du=='nm':
        return 1.0e-9
    elif du=='um':
        return 1.0e-6
    elif du=='mm':
        return 1.0e-3
    elif du=='m':
        return 1.0e-0
    else:
        print( "= = ERROR in relaxationModel: invalid distance unit definition!", file=sys.stderr )
        return

#Associated functions to assist
def _sanitise_v(v):
    if type(v) != np.ndarray:
        v=np.array(v)
    sh=v.shape
    if sh[-1] != 3:
        print( "= = ERROR in computation of A and D coefficients (spectral_densities.py): input v does not have 3 as its final dimension!", file=sys.stderr )
        sys.exit(2)
    return v

#Functions to calculate A and D coefficients:
def D_coefficients_symmtop(D):
    """
    Computes the 3 axisymmetric D-coefficients associated with the D_rot ellipsoid.
    """
    Dpar=D[0]
    Dperp=D[1]
    D_J=np.zeros(3)
    D_J[0]= 5*Dperp +   Dpar
    D_J[1]= 2*Dperp + 4*Dpar
    D_J[2]= 6*Dperp
    return D_J

def A_coefficients_symmtop(v, bProlate=True):
    """
    Computes the 3 axisymmetric A-coefficients associated with orientation of the vector w.r.t. to the D_rot ellipsoid.
    v can be many dimensions, as lons as the X/Y/Z cartesian dimensions is the last. e.g. v.shape = (M,N,3)
    Note the current implementation is probably a bit slower on small sizes comapred to the trivial.
    This is designed to handle many vectors at once.
    Also note: The unique axis changes when Daniso > 1 and when Daniso < 1 so as to preserve Dx<Dy<Dz formalism.
    This is governed by bProlate, which changes the unique axis to x when tumbling is oblate.
    """
    v=_sanitise_v(v)
    if bProlate:
        # Use z-dim.
        z2=np.square(v.take(-1,axis=-1))
    else:
        # Use x-dim.
        z2=np.square(v.take(0,axis=-1))
    onemz2=1-z2
    A0 = np.multiply( 3.0, np.multiply(z2,onemz2))
    A1 = np.multiply(0.75, np.square(onemz2))
    A2 = np.multiply(0.25, np.square(np.multiply(3,z2)-1))
    return np.stack((A0,A1,A2),axis=-1)

def Ddelta_ellipsoid(D):
    Diso= ( D[0] + D[1] + D[2] )/3.0
    D2  = ( D[0]*D[1] + D[0]*D[2] + D[1]*D[2] )/3.0
    delta =[ (D[i]-Diso)/sqrt(Diso**2 - D2**2) for i in range(3) ]
    return delta

def D_coefficients_ellipsoid(D, bDoDelta=False):
    """
    Computes the 5 full-anisotropic D-coefficients associated with the D_rot ellipsoid.
    Also returns 'delta' required for the corresponding A_coefficient computation, if required.
    """
    Diso= ( D[0] + D[1] + D[2] )/3.0
    D2  = ( D[0]*D[1] + D[0]*D[2] + D[1]*D[2] )/3.0
    fact1= sqrt(Diso**2 - D2**2)
    D_J=np.zeros(5)
    D_J[0]= 4*D[0] +   D[1] +   D[2]
    D_J[1]=   D[0] + 4*D[1] +   D[2]
    D_J[2]=   D[0] +   D[1] + 4*D[2]
    D_J[3]= 6*Diso + 6*fact1
    D_J[4]= 6*Diso - 6*fact1
    if bDoDelta:
        delta =[ (D[i]-Diso)/fact1 for i in range(3) ]
        return D_J, delta
    else:
        return D_J

def A_coefficients_ellipsoid(v, DD, bDDisDelta=False):
    """
    Computes the 5 sull-anisotropic A-coefficients associated with orientation of the vector w.r.t. to the D_rot ellipsoid.
    DD is given either as the D-Rot elements or its 'delta' transformation for direct use.
    """
    #v can be given as an array with X/Y/Z cartesian dimensions being the last.
    #"""
    if bDDisDelta:
        delta=DD
    else:
        delta=Ddelta_ellipsoid(dd)
    #v=_sanitise_v(v)
    #v2=np.square(v)
    #v4=np.square(v2)
    #fact2=np.multiply(0.75,np.sum(v4))-0.25
    v2 = [ v[i]*v[i] for i in range(3) ]
    v4 = [ v2[i]*v2[i] for i in range(3) ]
    fact2 = 0.25*( 3.0*(v4[0]+v4[1]+v4[2])-1.0)
    fact3 = 1.0/12.0*(delta[0]*(3*v4[0]+6*v2[1]*v2[2]-1) + delta[1]*(3*v4[1]+6*v2[0]*v2[2]-1) + delta[2]*(3*v4[2]+6*v2[0]*v2[1]-1))
    A=np.zeros(5)
    A[0]= 3*v2[1]*v2[2]
    A[1]= 3*v2[0]*v2[2]
    A[2]= 3*v2[0]*v2[1]
    A[3]= fact2-fact3
    A[4]= fact2+fact3
    return A

def _do_Jsum(om, A_J, D_J):
    """
    Lowest level operation. J = Sum_i components for each om. Return dimensions (N_om) for one vector A_j, and (N_Aj,N_om) otherwise.
    Equivalent to the old implementation:
    return np.array([ np.sum([A_J[i]*D_J[i]/(D_J[i]**2 + om[j]**2) for i in range(len(D_J))]) for j in range(len(om)) ])

    Output J has MxN, where M is the remaining dimensions of A, and N is the number of frequencies.
    J = A_ij*D_j/(D_j^2+om_k^2) = A_ij T_jk, in Einstein summation form.
    A can have many dimensions as long as the last dimension is the matching one.
    """
    Dmat=npufunc.Jomega.outer(D_J,om)
    return np.einsum('...j,jk',A_J,Dmat)

#model 0
#Rigid, isotropic tumbling
#J = tau_c / ( 1 + (w*tau_c)**2 )
def _Jglobal_sphere_D(w, Diso):
    J = 6*Diso / ( (6*Diso)**2 + w**2)
    return J

def _Jglobal_sphere_t(w, tau_c):
    J = tau_c / ( 1 + (w*tau_c)**2 )
    return J

# Rigid, symmetric top tumbling. Uses Fushman's notation in D.,
# but folds the 2/5 prefactor in J(omega) into the physical constants
# ofollowing Palmer.
def _Jglobal_symmtop_D(w, v, Dpar, Dperp):
    D_J=D_coefficients_symmtop((Dpar, Dperp))
    A_J=A_coefficients_symmtop(v, bProlate=(Dpar>Dperp) )
    J = [A_J[i]*D_J[i]/(D_J[i]**2 + w**2) for i in range(3)]
    return J

#Rigid, asymmetric tumbling
#See Ghose, Fushman, and Cowburn, J Magn. Reson., 2001
def _Jglobal_ellipsoid_D(w, v, D):
    D_J, delta = D_coefficients_ellipsoid(D, True)
    A_J=A_coefficients_ellipsoid(v, delta, True)
    J = [A_J[i]*D_J[i]/(D_J[i]**2 + w**2) for i in range(5)]
    return J

# = = =
# Classical Lipari-Szabo, isotropic
def _J_LipariSzabo_m2(w, tau_glob, S2, tau_int):
    tau_eff = tau_int * tau_glob / (tau_int + tau_glob)
    J = S2 * tau_glob / (1 + (w*tau_glob)**2 ) + (1-S2) * tau_eff / (1 + (w*tau_eff)**2)
    return J
    #J = S2 * Diso / (Diso**2 + w**2 ) + (1-S2) * tau_eff / (1 + (w*tau_eff)**2)

# Classical Lipari-Szabo + Effective anisotropic diffusion.
# See d'Auvergne's thesis in 2006.
def _J_combine_LS_anisotropic(w, S2, tau_int, A_J, D_J):
    """
    The internal loop for spectral density calcualtions.
    Keeps w as an internal 1D-vector throughout.
    """
    dims=len(A_J)
    J=np.zeros( (dims,len(w)) )
    for k in range(dims):
        D_eff=D_J[k]+1.0/tau_int
        J[i] = S2[i]*A_J[k]*D_J[k]/(D_J[k]**2+w**2) + (1-S2) * D_eff/(D_eff**2 + w**2)
    return J.sum(axis=0)

def J_direct_transform(om, consts, taus):
    """
    This calculates the direct fourier transform of C(t) without a global tumbling factor.
    In this case the order parameter makes no contributions whatsoever?
    """
    ndecay=len(consts) ; noms=len(om)
    Jmat = np.zeros( (ndecay, noms ) )
    for i in range(ndecay):
        Jmat[i] = consts[i]*taus[i] /(1 + (taus[i]*om)**2.)
    return Jmat.sum(axis=0)

#tau_eff = tau_int * tau_iso / (tau_int + tau_iso)
#J = S2 * Diso / (Diso**2 + w**2 ) + (1-S2) * tau_eff / (1 + (w*tau_eff)**2)

def J_combine_isotropic_exp_decayN(om, tau_iso, S2, consts, taus):
    """
    This calculats the J value for combining an isotropic global tumbling with
    a fitted internal autocorrelation C(t), where
    C(t) = S2 + Sum{ consts[i] * exp ( -t/tau[i] }
    thus this allows fits to multiple time constants in C(t).
    """
    k = (1.0/tau_iso)+(1.0/np.array(taus))
    ndecay=len(consts) ; noms  =len(om)
    Jmat = S2*tau_iso/(1.+(om*tau_iso)**2.)
    for i in range(ndecay):
        Jmat += consts[i]*k[i] /(k[i]**2.+om**2.)
    return Jmat
    #Jmat = np.zeros( (ndecay+1, noms ) )
    #Jmat[0]= S2*tau_iso/(1.+(om*tau_iso)**2.)
    #for i in range(ndecay):
    #    Jmat[i+1] = consts[i]*k[i] /(k[i]**2.+om**2.)
    #return Jmat.sum(axis=0)

def J_combine_symmtop_exp_decayN(om, v, Dpar, Dperp, S2, consts, taus):
    """
    Calculates the J value for combining a symmetric-top anisotropic tumbling with
    a fitted internal autocorrelation C(t), where
    C(t) = S2 + Sum{ consts[i] * exp ( -t/tau[i] }
    thus this allows fits to multiple time constants in C(t).
    Note that v needs to be in the frame of the rotational diffusion tensor D, i.e. PAF.

    This function supports giving multiple vectors at once, of the form v.shape=(L,M,N,...,3)
    """
    #v can be given as an array, with the X/Y/Z cartesian axisin the last position.
    #"""
    D_J=D_coefficients_symmtop((Dpar, Dperp))
    # A_J is the same shape as v, so 3 in this case.
    A_J=A_coefficients_symmtop(v, bProlate=(Dpar>Dperp) )
    ndecay=len(consts) ; noms  =len(om)

    Jmat = _do_Jsum(om, S2*A_J, D_J)
    for i in range(ndecay):
        Jmat += _do_Jsum(om, consts[i]*A_J, D_J+1./taus[i])
    return Jmat
#    if len(v.shape) > 1:
#        Jmat0 = _do_Jsum(om, S2*A_J, D_J)
#        sh_J = Jmat0.shape ; sh_out=list(sh_J) ; sh_out.insert(0, ndecay+1)
#        Jmat = np.zeros(sh_out)
#        Jmat[0] = Jmat0
#        for i in range(ndecay):
#            Jmat[i+1] = _do_Jsum(om, consts[i]*A_J, D_J+1./taus[i])
#        return Jmat.sum(axis=0)
#    else:
#        Jmat = np.zeros( (ndecay+1, noms ) )
#        Jmat[0]= _do_Jsum(om, S2*A_J, D_J)
#        for i in range(ndecay):
#            Jmat[i+1] = _do_Jsum(om, consts[i]*A_J, D_J+1./taus[i])
#        return Jmat.sum(axis=0)
    #return _do_Jsum( S2*A_J, D_J) + np.sum([ _do_Jsum(consts[i]*A_J, D_J+1./taus[i]) for i in range(len(consts)) ])

def J_combine_ellipsoid_exp_decayN(om, v, D, S2, consts, taus):
    """
    This calculats the J value for combining an ellipsoid, fully anisotropic tumbling with
    a fitted internal autocorrelation C(t), where
    C(t) = S2 + Sum{ consts[i] * exp ( -t/tau[i] }
    thus this allows fits to multiple time constants in C(t).
    Note that v needs to be in the frame of the rotational diffusion tensor D.
    D = (Dx, Dy, Dz) where Dx <= Dy <= Dz.
    """
    D_J, delta = D_coefficients_ellipsoid(D, True)
    A_J=A_coefficients_ellipsoid(v, delta, True)
    return _do_Jsum(om, S2*A_J, D_J) + np.sum([ _do_Jsum(om, consts[i]*A_J, D_J+1./taus[i]) for i in range(len(consts)) ])

def calculate_spectral_density(model, w, *args):
    """
    The models adopted in this spectral density J(w) calculation are as follows:
    - rigid_sphere_T (tau_glob)
      Returns a single J reflecting spherical tumbling, given one constant.
    - rigid_sphere_D (D_iso)
      Returns a single J reflecting spherical tumbling, given one constant.
    - rigid_symmtop_D (D , [v] )
      D=(Dpar, Dperp)
    - rigid_ellipsoid_D (D , [v] )
      D=(D[0], D[1], D[2])
    - LS_classic_T (tau_glob, [S2], [tau_int])
      Classic Lipari-Szabo formulation.
      Uses one global tumbling parameter, plus a list of internal parameters S2 and tau_int
      for each vector being included. Being model-free, the vectors themselves are not required.
    - LS_classic_D (Diso, [S2], [tau_int] )
    - LS_symmtop_D (D, [v], [S2], [tau_int] )
    """

    if model=='rigid_sphere_T':
        return _Jglobal_sphere_t(w, args[0])
    if model=='rigid_sphere_D':
        return _Jglobal_sphere_D(w, args[0])
    if model=='rigid_symmtop_D':
        D=args[0]; v=args[1]
        D_J=D_coefficients_symmtop(D)
        J=np.zeros((len(v),len(w)))
        for i in range(len(v)):
            A_J=A_coefficients_symmtop(v[i], bProlate=(D[0]>D[1]))
            J[i] = [ np.sum([A_J[k]*D_J[k]/(D_J[k]**2 + w[j]**2) for k in range(3)]) for j in range(len(w)) ]
        return J
    if model=='rigid_ellipsoid_D':
        D=args[0]; v=args[1]
        D_J, delta =D_coefficients_ellipsoid(D, True)
        J=np.zeros((len(v),len(w)))
        for i in range(len(v)):
            A_J=A_coefficients_ellipsoid(v[i], delta, True)
            J[i] = [ np.sum([A_J[k]*D_J[k]/(D_J[k]**2 + w[j]**2) for k in range(5)]) for j in range(len(w)) ]
        return J
    if model=='LS_classic_D':
        tau_glob=args[0] ;  S2=args[1] ; tau_int=args[2]
        J=np.zeros( (len(S2),len(w)) )
        J = [ _J_LipariSzabo_m2(w, tau_glob, S2[i], tau_int[i]) for i in range(len(S2)) ]
        return J
    if model=='LS_symmtop_D':
        # See eq. 8.66 in d'Auvergne's Thesis (2006) @ University of Melbourne, Australia
        # We apply the Lipari-Szabo to each time coefficient.
        D=args[0] ; v=args[1] ; S2=args[2] ; tau_int=args[3]
        D_J=D_coefficients_symmtop(D)
        J=np.zeros((len(v),len(w)))
        for i in range(len(v)):
            A_J=A_coefficients_symmtop(v[i], bProlate=(D[0]>D[1]))
            J[i]=_J_combine_LS_anisotropic(w, S2[i], tau_int[i], A_J, D_J)
        return J
    if model=='LS_ellipsoid_D':
        # See eq. 8.66 in d'Auvergne's Thesis (2006) @ University of Melbourne, Australia
        # We apply the Lipari-Szabo to each time coefficient.
        D=args[0] ; v=args[1] ; S2=args[2] ; tau_int=args[3]
        D_J, delta=D_coefficients_ellipsoid(D, True)
        J=np.zeros((len(v),len(w)))
        for i in range(len(v)):
            A_J=A_coefficients_ellipsoid(v[i], delta, True)
            J[i]=_J_combine_LS_anisotropic(w, S2[i], tau_int[i], A_J, D_J)
        return J

    else:
        print( "= = ERROR: unknown model given to calculate_spectral_density!" )
        return -1

def obtain_HX_frequencies(gamma_X=-27.116e6, DeltaSigma_X=-160e-6, r_XH=1.02e-10, gamma_1H=267.513e6):
    iOmH = 3 ; # indexing for the frequencies.
    iOmX = 1 ; #
    # First determine the frequencies omega and J from given inputs.
    omega = np.zeros(5)
    omega[iOmH] = -1.0*gamma_1H*B_0
    omega[iOmX] = -1.0*gamma_X*B_0
    omega[iOmH-iOmX] = omega[iOmH]-omega[iOmX]
    omega[iOmH+iOmX] = omega[iOmH]+omega[iOmX]

    return omega, iOmH, iOmX

def calculate_relaxation_from_J(J):
    # f_DD = 519627720.1974593 , if r_NH is at default values
    f_DD  = 7.958699205571828e-67 * r_XH**-6.0 * gamma_X**2
    # f_CSA = 498637299.69233465, if B_0 = 600.13, and DeltaSigma=-160e-6
    f_CSA = 2.0/15.0 * DeltaSigma_X**2 * ( gamma_X * B_0 )**2

    R1 = f_DD*( J[iOmH-iOmX] + 3*J[iOmX] + 6*J[iOmH+iOmX] ) + f_CSA*J[iOmX]
    R2 = 0.5*f_DD*( 4*J[0] + J[iOmH-iOmX] + 3*J[iOmX] + 6*J[iOmH+iOmX] + 6*J[iOmH] ) + 1.0/6.0*f_CSA*(4*J[0] + 3*J[iOmX])
    NOE = 1.0 + gamma_1H/gamma_X/R1 * f_DD*(6*J[iOmH+iOmX] - J[iOmH-iOmX])

    return R1, R2, NOE

def calculate_relaxation(model, B_0, gamma_X=-27.116e6, DeltaSigma_X=-160e-6, r_XH=1.02e-10, *args):
    """
    To calculate relaxation, we require values of J at five different frequencies:
    - J[0], J[om_N], J[iOmH-iOmX], J[iOmX], J[iOmH + iOmX]

    Define constants using Palmer's standard so as to leave J(omega) with no prefactors.
    factor_dipole-dipole and factor_CSA.
    f_DD  = 0.10* (mu_0*hbar/4.0/pi)**2 * gamma_15N**2 * gamma_1H**2 * r_NH**-6.0
    f_CSA = 2.0/15.0 * gamma_15N**2 * B_0 **2 * DeltaSig_15N**2
    mu_0 = 8.85418782e-12 ; # m^-3 kg^-1 s^4 A^2
    mu_0 = 4*pi*1e-7
    hbar = 1.0545718e-34  ; # m^2 kg s^-1
    pi   = 3.14159265359
    gamma_1H  = 267.513e6  ; # rad s^-1 T^-1
    gamma_15N = -27.116e6  ; # rad s^-1 T^-1
    omega_15N = - gamma_15N * B_0 .
    r_NH = 1.02e-10 ;# m
    DeltaSigma_15N is generally dependent on the local structure,
    the literature average is -160 ppm

    R1 = f_DD*( J[om_N-iOmH] + 3*J[om_N] + 6*J[om_N+iOmH] ) + f_CSA*J[om_N]
    R2 = 0.5*f_DD*( 4*J[0] + J[om_N-iOmH] + 3*J[om_N] + 6*J[om_N+iOmH] + 6*J[iOmH] ) + 1.0/6.0*f_CSA*(4*J[0] + 3*J[om_N])
    NOE = 1.0 + gamma_H/gamma_N/R1 * f_DD*(6*J[om_N+iOmH] - J[om_N-iOmH])
    """
    omega, iOmH, iOmX = obtain_HX_frequencies(gamma_X, DeltaSigma_X, r_XH)
    gamma_1H  = 267.513e6  ; # rad s^-1 T^-1
    # First determine the frequencies omega and J from given inputs.

    J = calculate_spectral_density(model, omega, args)

    # f_DD = 519627720.1974593 , if r_NH is at default values
    f_DD  = 7.958699205571828e-67 * r_XH**-6.0 * gamma_X**2
    # f_CSA = 498637299.69233465, if B_0 = 600.13, and DeltaSigma=-160e-6
    f_CSA = 2.0/15.0 * DeltaSigma_X**2 * ( gamma_X * B_0 )**2

    R1 = f_DD*( J[iOmH-iOmX] + 3*J[iOmX] + 6*J[iOmH+iOmX] ) + f_CSA*J[iOmX]
    R2 = 0.5*f_DD*( 4*J[0] + J[iOmH-iOmX] + 3*J[iOmX] + 6*J[iOmH+iOmX] + 6*J[iOmH] ) + 1.0/6.0*f_CSA*(4*J[0] + 3*J[iOmX])
    NOE = 1.0 + gamma_1H/gamma_X/R1 * f_DD*(6*J[iOmH+iOmX] - J[iOmH-iOmX])

    return R1, R2, NOE

def calculate_rho_from_relaxation(gamma_H, gamma_N, R1, R2, NOE):
    """
    Taking Eq. 4 of Ghose, Fushman and Cowburn (2001), calculate rho from R1, R2, and NOE directly,
    rather than from the spectral density J(omega). This is used to convert experimental measurements to rho.
    """
    HF  = -0.2*(gamma_N/gamma_H)*(1-NOE)*R1
    R1p = R1 - 7.0*(0.921/0.87)**2.0*HF
    R2p = R2 - 6.5*(0.955/0.87)**2.0*HF

    return 4.0/3.0*R1p/(2.0*R2p-R1p)

def calculate_NH_relaxation_from_Ct(bondtype, B_0, t, Ct):
    """
    Assume NH for now.
    """
    gamma_1H  = 267.513e6  ; # rad s^-1 T^-1
    gamma_X   = -27.116e6
    DeltaSigma_X=-160e-6
    r_XH=1.02e-10

    om, G = do_dft(t, Ct)
    J = G.real

    omega, iOmH, iOmX = obtain_HX_frequencies()

    Jw=np.zeros(5)
    for i in range(5):
        w  = omega[i]
        Jw[i] = interpolate_point(w, om, J)

    # f_DD = 519627720.1974593 , if r_NH is at default values
    f_DD  = 7.958699205571828e-67 * r_XH**-6.0 * gamma_X**2
    # f_CSA = 498637299.69233465, if B_0 = 600.13, and DeltaSigma=-160e-6
    f_CSA = 2.0/15.0 * DeltaSigma_X**2 * ( gamma_X * B_0 )**2

    R1 = f_DD*( J[iOmH-iOmX] + 3*J[iOmX] + 6*J[iOmH+iOmX] ) + f_CSA*J[iOmX]
    R2 = 0.5*f_DD*( 4*J[0] + J[iOmH-iOmX] + 3*J[iOmX] + 6*J[iOmH+iOmX] + 6*J[iOmH] ) + 1.0/6.0*f_CSA*(4*J[0] + 3*J[iOmX])
    NOE = 1.0 + gamma_1H/gamma_X/R1 * f_DD*(6*J[iOmH+iOmX] - J[iOmH-iOmX])

    return R1, R2, NOE



# = = = Section on direct fourier transform from a C(t) signal
# Uses Palmer's definition, so no 2/5 out the front.
#
# J(omega) = Real{ Integral_inf^inf C(tau)*exp(-i*omega*tau) dtau  }
#
# Boils down to a discrete fourier transform, and taking its real part
#
# J = Real { np.fft.rfft( C(tau) }
#
# The time sequence is given: [0, dt, 2*dt, ... (N-1)*dt] (N total points)
# And returns the omega: [  0, 2*pi/(N*dt), 4*pi/(N*dt), ... , pi*/dt] (N/2 total points)

def interpolate_point(xi, x, y):
    """
    find y(xi) by splitting x in half repeatedly until the x is found.
    """
    num_pts = len(x)
    if num_pts%2==0:
        #Even
        i_h2 = num_pts/2
        i_h1 = num_pts/2 - 1
        if x[i_h2] < xi:
            return interpolate_point(xi, x[i_h2:], y[0:i_h2:])
        elif x[i_h1] > xi:
            return interpolate_point(xi, x[0:i_h1], y[0:i_h1])
        else:
            return ((xi-x[i_h1])*y[i_h2]+(x[i_h2]-xi)*y[i_h1])/(x[i_h2]-x[i_h1])
    else:
        #Odd
        i_half = num_pts/2
        if   x[i_half] < xi:
            return interpolate_point(xi, x[i_half:], y[i_half:])
        elif x[i_half] > xi:
            return interpolate_point(xi, x[0:i_half+1], y[0:i_half+1])
        else:
            return y[i_half]


def do_dft(t, f):
    dt = t[1]-t[0]
    N = len(t)
    if N != len(f):
        print( "= = ERROR: lengths of time and function to be transformed are not the same!", file=sys.stderr )
        return -1

    om = np.linspace(0.0, np.pi/dt, 1+N/2)
    G = np.fft.rfft(f)
    return om, G

# = = = = Imported functions from calculate-relaxations
def convert_LambertCylindricalHist_to_vecs(hist, edges):
    print( "= = = Reading histogram in Lambert-Cylindral projection, and returning distribution of non-zero vectors." )
    # = = = Expect histograms as a list of 2D entries: (nResidues, phi, cosTheta)
    nResidues   = hist.shape[0]
    phis   = 0.5*(edges[0][:-1]+edges[0][1:])
    thetas = np.arccos( 0.5*(edges[1][:-1]+edges[1][1:]) )
    pt = np.moveaxis( np.array( np.meshgrid( phis, thetas, indexing='ij') ), 0, -1)
    binVecs = gm.rtp_to_xyz( pt, vaxis=-1, bUnit=True )
    del pt, phis, thetas
    print( "    ...shapes of first histogram and average-vector array:", hist[0].shape, binVecs.shape )
    nPoints = hist[0].shape[0]*hist[0].shape[1]
    # = = = just in case this is a list of histograms..
    # = = = Keep all of the zero-weight entries vecause it keeps the broadcasting speed.
    #vecs    = np.zeros( (nResidues, nPoints, 3 ), dtpye=binVecs.dtype )
    #weights = np.zeros_like( vecs )
    return np.repeat( binVecs.reshape(nPoints,3)[np.newaxis,...], nResidues, axis=0), \
           np.reshape( hist, ( nResidues, nPoints) )
    # return vecs, weights
    
# There may be minor mistakes in the selection text. Try to identify what is wrong.
def confirm_mdtraj_seltxt(mol, Hseltxt, Xseltxt):
    bError=False
    indH = mol.topology.select(Hseltxt)
    indX = mol.topology.select(Xseltxt)
    numH = len(indH) ; numX = len(indX)
    if numH == 0:
        bError=True
        t1 = mol.topology.select('name H')
        t2 = mol.topology.select('name HN')
        t3 = mol.topology.select('name HA')
        print( "    .... ERROR: The 'name H' selects %i atoms, 'name HN' selects %i atoms, and 'name HA' selects %i atoms." % (t1, t2, t3) )
    if numX == 0:
        bError=True
        t1 = mol.topology.select('name N')
        t2 = mol.topology.select('name NT')
        print( "    .... ERROR: The 'name N' selects %i atoms, and 'name NT' selects %i atoms." % (t1, t2) )

    resH = [ mol.topology.atom(x).residue.resSeq for x in indH ]
    resX = [ mol.topology.atom(x).residue.resSeq for x in indX ]
    if resX != resH:
        bError=True
        print( "    .... ERROR: The residue lists are not the name between the two selections:" )
        print( "    .... Count for X (%i)" % numX, resX )
        print( "    .... Count for H (%i)" % numH, resH )
    if bError:
        print("    ... Errors encountered, please fix up the selection syntax and rerun using import_frame_vectors_pdb() with these syntaxes.")
        sys.exit(1)

    return indH, indX, resX