#!/usr/bin/python

import sys
import numpy as np
from math import *
import npufunc

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
    def __init__(self, isotope):
        self.name=isotope
        self.csa=0.0
        self.time_unit='s'
        self.time_fact=_return_time_fact(self.time_unit)
        self.set_gamma(self.name)

    def set_gamma(self, name):
        """
        Sets the gamma value based on the gyromag's isotope and time unit definition.
        Gamma is in units of rad s^-1 T^-1 .
        """
        if name=='1H':
            self.gamma=267.513e6*self.time_fact
            #In standard freq: 42.576e6
        elif name=='13C':
            self.gamma=67.262e6*self.time_fact
            # Good average unknown
            self.csa=-130e-6
        elif name=='15N':
            self.gamma=-27.116e6*self.time_fact
            # Large variation possible. See Fushman, Tjandra, and Cowburn, 1998.
            # Also use more commonly accepted value.
            self.csa=-170e-6
        elif name=='17O':
            self.gamma=-36.264e6*self.time_fact
        elif name=='19F':
            self.gamma=251.662e6*self.time_fact
        elif name=='31P':
            self.gamma=108.291e6*self.time_fact

    def set_time_unit(self, tu):
        old = self.time_fact
        self.time_fact = _return_time_fact(tu)
        mult = self.time_fact/old
        self.gamma *= mult

class diffusion_model:
    def __init__(self, model, *args):
        self.time_unit=args[0]
        self.time_fact=_return_time_fact(self.time_unit)
        if   model=='rigid_sphere_T':
            self.name='rigid_sphere'
            self.D=1.0/6.0*float(args[1])
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
            print >> sys.stderr, "= = ERROR: change_Diso for fully anisotropic models, not implemented."
            sys.exit(1)

class relaxObject:
    """
    Help for class relaxObject:
    This class handles objects and functions related to calculating NMR spin relaxation.
    Attributes:
        bond - Name of the intended vector bond, eg 'NH'
        B_0  - Background magnetic field, in Teslas.
        time_unit - Units of time used for the freqency, such as 'ns', or 'ps'.
        rotdif_model - The Rotational Diffusion model used to define spectral densitiies J(omega)
    """

    def __init__(self, bond, B_0):
        # Parameters associated with units
        self.omega=np.array(0.0)
        self.time_unit='ns'
        self.time_fact=_return_time_fact(self.time_unit)
        self.dist_unit='nm'
        self.dist_fact=_return_dist_fact(self.dist_unit)

        # Atomic parameters.
        self.bond = bond
        self.B_0  = B_0
        if   bond=='NH':
            self.gH  = gyromag('1H')
            self.gX  = gyromag('15N')
            # = = Question use conventional 1.02, or 1.04 to include librations, according to Case, J Biomol. NMR, 1999? = = =
            self.rXH = 1.02e-1
            #self.rXH = 1.04e-1
        elif bond=='CH':
            self.gH  = gyromag('1H')
            self.gX  = gyromag('13C')
            # Need to update bond length for this atoms
            self.rXH = 1.02e-1
        else:
            print >> sys.stderr, "= = ERROR in relaxObject: wrong bond definition! = ="
            sys.exit(1)

        # relaxation model. Note in time units of host object.
        self.set_rotdif_model('rigid_sphere_T', 1.0)

    def set_B0(self, B_0):
        self.B_0 = B_0

    def set_time_unit(self, tu):
        old=self.time_fact
        self.time_fact = _return_time_fact(tu)
        # Update all time units can measurements.
        self.time_unit=tu
        mult = self.time_fact / old
        self.omega *= mult
        self.rotdif_model.set_time_unit(tu)
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
        om_H = 3 ; # indexing for the frequencies.
        om_X = 1 ; #
        # First determine the frequencies omega and J from given inputs.
        self.omega[om_H] = -1.0*self.gH.gamma*self.B_0*self.time_fact
        self.omega[om_X] = -1.0*self.gX.gamma*self.B_0*self.time_fact
        self.omega[om_H-om_X] = (self.omega[om_H]-self.omega[om_X])
        self.omega[om_H+om_X] = (self.omega[om_H]+self.omega[om_X])

    def print_freq_order(self):
        print "omega=(0, om_X, om_H-om_X, om_H, om_H+om_X )"

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
        self.rotdif_model=diffusion_model(model, self.time_unit, *args)

    def get_Jomega(self, vNH):
        """
        Calculate the spectral density function for a given set of unit vectors and the current model.
        """
        num_vecs = len(vNH)
        J = np.zeros( (num_vecs, self.num_omega) )
        for i in range(num_vecs):
            J[i] = function_to_be_written(vNH[i])
        return 'Not composed'

    def get_relax_from_J(self, J):
        #f_DD  = 0.10* (mu_0*hbar/4.0/pi)**2 * gamma_15N**2 * gamma_1H**2 * r_NH**-6.0
        #f_CSA = 2.0/15.0 * gamma_15N**2 * B_0 **2 * DeltaSig_15N**2
        #mu_0 = 4*pi*1e-7      ; # m   kg s^-2 A-2
        #hbar = 1.0545718e-34  ; # m^2 kg s^-1
        #pi   = 3.14159265359
        #gamma_1H  = 267.513e6  ; # rad s^-1 T^-1
        #gamma_15N = -27.116e6  ; # rad s^-1 T^-1
        #omega_15N = - gamma_15N * B_0 .
        #r_NH = 1.02e-10 ;# m
        # (mu_0*hbar/4.0/pi)**2 m^-1 s^2 is the 10^-82 number below. f_DD and f_CSA are maintained in SI units.
        f_DD = 0.10 * 1.1121216813552401e-82*self.gH.gamma**2.0*self.gX.gamma**2.0 *(self.rXH*self.dist_fact)**-6.0
        f_CSA = 2.0/15.0 * self.gX.csa**2.0 * ( self.gX.gamma * self.B_0 )**2
        om_H = 3 ; # indexing for the frequencies.
        om_X = 1 ; #
        # Note: since J is in the units of inverse time, data needs to be converted back to s^-1
        R1 = self.time_fact*( f_DD*( J[om_H-om_X] + 3*J[om_X] + 6*J[om_H+om_X] ) + f_CSA*J[om_X] )
        R2 = self.time_fact*( 0.5*f_DD*( 4*J[0] + J[om_H-om_X] + 3*J[om_X] + 6*J[om_H+om_X] + 6*J[om_H] ) + 1.0/6.0*f_CSA*(4*J[0] + 3*J[om_X]) )
        NOE = 1.0 + self.time_fact * self.gH.gamma/(self.gX.gamma*R1) * f_DD*(6*J[om_H+om_X] - J[om_H-om_X])

        return R1, R2, NOE

    def get_rho_from_J(self, J):
        """
        Taking Eq. 4 of Ghose, Fushman and Cowburn (2001), and define rho as a ratio of modified R1' and R2'
        that have high frequency components removed.
        """
        om_X=1
        return J[om_X]/J[0]

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
            print "= = ERROR: drho calculation is not implemented!"
            sys.exit(1)
            return (rho, drho)


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
        print >> sys.stderr, "= = ERROR in object definition: invalid time unit definition!"
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
        print >> sys.stderr, "= = ERROR in relaxObject: invalid distance unit definition!"
        return

#Associated functions to assist
def _sanitise_v(v):
    if type(v) != np.ndarray:
        v=np.array(v)
    sh=v.shape
    if sh[-1] != 3:
        print >> sys.stderr, "= = ERROR in computation of A and D coefficients (spectral_densities.py): input v does not have 3 as its final dimension!"
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
    #z2=v[2]*v[2]
    #A=np.zeros(3)
    #A[0]= 3.00*z2*(1-z2)
    #A[1]= 0.75*(1-z2)**2
    #A[2]= 0.25*(3*z2-1)**2
    #return A

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
    A can have many dimensions as long as th elast dimension is the matching one.
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
    Jmat = np.zeros( (ndecay+1, noms ) )
    Jmat[0]= S2*tau_iso/(1.+(om*tau_iso)**2.)
    for i in range(ndecay):
        Jmat[i+1] = consts[i]*k[i] /(k[i]**2.+om**2.)
    return Jmat.sum(axis=0)

def J_combine_symmtop_exp_decayN(om, v, Dpar, Dperp, S2, consts, taus):
    """
    Calculats the J value for combining a symmetric-top anisotropic tumbling with
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

    if len(v.shape) > 1:
        Jmat0 = _do_Jsum(om, S2*A_J, D_J)
        sh_J = Jmat0.shape ; sh_out=list(sh_J) ; sh_out.insert(0, ndecay+1)
        Jmat = np.zeros(sh_out)
        Jmat[0] = Jmat0
        for i in range(ndecay):
            Jmat[i+1] = _do_Jsum(om, consts[i]*A_J, D_J+1./taus[i])
        return Jmat.sum(axis=0)
    else:
        Jmat = np.zeros( (ndecay+1, noms ) )
        Jmat[0]= _do_Jsum(om, S2*A_J, D_J)
        for i in range(ndecay):
            Jmat[i+1] = _do_Jsum(om, consts[i]*A_J, D_J+1./taus[i])
        return Jmat.sum(axis=0)
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
        print "= = ERROR: unknown model given to calculate_spectral_density!"
        return -1

def obtain_HX_frequencies(gamma_X=-27.116e6, DeltaSigma_X=-160e-6, r_XH=1.02e-10, gamma_1H=267.513e6):
    om_H = 3 ; # indexing for the frequencies.
    om_X = 1 ; #
    # First determine the frequencies omega and J from given inputs.
    omega = np.zeros(5)
    omega[om_H] = -1.0*gamma_1H*B_0
    omega[om_X] = -1.0*gamma_X*B_0
    omega[om_H-om_X] = omega[om_H]-omega[om_X]
    omega[om_H+om_X] = omega[om_H]+omega[om_X]

    return omega, om_H, om_X

def calculate_relaxation_from_J(J):
    # f_DD = 519627720.1974593 , if r_NH is at default values
    f_DD  = 7.958699205571828e-67 * r_XH**-6.0 * gamma_X**2
    # f_CSA = 498637299.69233465, if B_0 = 600.13, and DeltaSigma=-160e-6
    f_CSA = 2.0/15.0 * DeltaSigma_X**2 * ( gamma_X * B_0 )**2

    R1 = f_DD*( J[om_H-om_X] + 3*J[om_X] + 6*J[om_H+om_X] ) + f_CSA*J[om_X]
    R2 = 0.5*f_DD*( 4*J[0] + J[om_H-om_X] + 3*J[om_X] + 6*J[om_H+om_X] + 6*J[om_H] ) + 1.0/6.0*f_CSA*(4*J[0] + 3*J[om_X])
    NOE = 1.0 + gamma_1H/gamma_X/R1 * f_DD*(6*J[om_H+om_X] - J[om_H-om_X])

    return R1, R2, NOE

def calculate_relaxation(model, B_0, gamma_X=-27.116e6, DeltaSigma_X=-160e-6, r_XH=1.02e-10, *args):
    """
    To calculate relaxation, we require values of J at five different frequencies:
    - J[0], J[om_N], J[om_H-om_X], J[om_X], J[om_H + om_X]

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

    R1 = f_DD*( J[om_N-om_H] + 3*J[om_N] + 6*J[om_N+om_H] ) + f_CSA*J[om_N]
    R2 = 0.5*f_DD*( 4*J[0] + J[om_N-om_H] + 3*J[om_N] + 6*J[om_N+om_H] + 6*J[om_H] ) + 1.0/6.0*f_CSA*(4*J[0] + 3*J[om_N])
    NOE = 1.0 + gamma_H/gamma_N/R1 * f_DD*(6*J[om_N+om_H] - J[om_N-om_H])
    """
    omega, om_H, om_X = obtain_HX_frequencies(gamma_X, DeltaSigma_X, r_XH)
    gamma_1H  = 267.513e6  ; # rad s^-1 T^-1
    # First determine the frequencies omega and J from given inputs.

    J = calculate_spectral_density(model, omega, args)

    # f_DD = 519627720.1974593 , if r_NH is at default values
    f_DD  = 7.958699205571828e-67 * r_XH**-6.0 * gamma_X**2
    # f_CSA = 498637299.69233465, if B_0 = 600.13, and DeltaSigma=-160e-6
    f_CSA = 2.0/15.0 * DeltaSigma_X**2 * ( gamma_X * B_0 )**2

    R1 = f_DD*( J[om_H-om_X] + 3*J[om_X] + 6*J[om_H+om_X] ) + f_CSA*J[om_X]
    R2 = 0.5*f_DD*( 4*J[0] + J[om_H-om_X] + 3*J[om_X] + 6*J[om_H+om_X] + 6*J[om_H] ) + 1.0/6.0*f_CSA*(4*J[0] + 3*J[om_X])
    NOE = 1.0 + gamma_1H/gamma_X/R1 * f_DD*(6*J[om_H+om_X] - J[om_H-om_X])

    return R1, R2, NOE

def calculate_rho_from_relaxation(gamma_H, gamma_N, R1, R2, NOE):
    """
    Taking Eq. 4 of Ghose, Fushman and Cowburn (2001), calculate rho from R1, R2, and NOE directly,
    rather than from the spectral density J(omega). THis is used to convert experimental measurements to rho.
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

    omega, om_H, om_X = obtain_HX_frequencies()

    Jw=np.zeros(5)
    for i in range(5):
        w  = omega[i]
        Jw[i] = interpolate_point(w, om, J)

    # f_DD = 519627720.1974593 , if r_NH is at default values
    f_DD  = 7.958699205571828e-67 * r_XH**-6.0 * gamma_X**2
    # f_CSA = 498637299.69233465, if B_0 = 600.13, and DeltaSigma=-160e-6
    f_CSA = 2.0/15.0 * DeltaSigma_X**2 * ( gamma_X * B_0 )**2

    R1 = f_DD*( J[om_H-om_X] + 3*J[om_X] + 6*J[om_H+om_X] ) + f_CSA*J[om_X]
    R2 = 0.5*f_DD*( 4*J[0] + J[om_H-om_X] + 3*J[om_X] + 6*J[om_H+om_X] + 6*J[om_H] ) + 1.0/6.0*f_CSA*(4*J[0] + 3*J[om_X])
    NOE = 1.0 + gamma_1H/gamma_X/R1 * f_DD*(6*J[om_H+om_X] - J[om_H-om_X])

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
        print >> sys.stderr, "= = ERROR: lengths of time and function to be transformed are not the same!"
        return -1

    om = np.linspace(0.0, np.pi/dt, 1+N/2)
    G = np.fft.rfft(f)
    return om, G

