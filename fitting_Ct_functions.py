import numpy as np
from scipy.optimize import curve_fit
import sys
#from scipy.optimize import powell_min

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

# This is a series of exponential functions that are a simple sum of exponentials.
# The odd-numbered  set allows the
def func_exp_decay1(t, tau_a):
    return np.exp(-t/tau_a)
def func_exp_decay2(t, A, tau_a):
    return (1-A) + A*np.exp(-t/tau_a)
def func_exp_decay3(t, S2, A, tau_a):
    return S2 + A*np.exp(-t/tau_a)
def func_exp_decay4(t, A, tau_a, B, tau_b):
    return (1-A-B) + A*np.exp(-t/tau_a) + B*np.exp(-t/tau_b)
def func_exp_decay5(t, S2, A, tau_a, B, tau_b ):
    return S2 + A*np.exp(-t/tau_a) + B*np.exp(-t/tau_b)
def func_exp_decay6(t, A, tau_a, B, tau_b, G, tau_g ):
    return (1-A-B-G) + A*np.exp(-t/tau_a) + B*np.exp(-t/tau_b) + G*np.exp(-t/tau_g)
def func_exp_decay7(t, S2, A, tau_a, B, tau_b, G, tau_g ):
    return S2 + A*np.exp(-t/tau_a) + B*np.exp(-t/tau_b) + G*np.exp(-t/tau_g)
def func_exp_decay8(t, A, tau_a, B, tau_b, G, tau_g, D, tau_d):
    return (1-A-B-G-D) + A*np.exp(-t/tau_a) + B*np.exp(-t/tau_b) + G*np.exp(-t/tau_g) + D*np.exp(-t/tau_d)
def func_exp_decay9(t, S2, A, tau_a, B, tau_b, G, tau_g, D, tau_d):
    return S2 + A*np.exp(-t/tau_a) + B*np.exp(-t/tau_b) + G*np.exp(-t/tau_g) + D*np.exp(-t/tau_d)
def func_exp_decay10(t, A, tau_a, B, tau_b, G, tau_g, D, tau_d, E, tau_e):
    return (1-A-B-G-D-E) + A*np.exp(-t/tau_a) + B*np.exp(-t/tau_b) + G*np.exp(-t/tau_g) + D*np.exp(-t/tau_d) + E*np.exp(-t/tau_e)
def func_exp_decay11(t, S2, A, tau_a, B, tau_b, G, tau_g, D, tau_d, E, tau_e):
    return S2 + A*np.exp(-t/tau_a) + B*np.exp(-t/tau_b) + G*np.exp(-t/tau_g) + D*np.exp(-t/tau_d) + E*np.exp(-t/tau_e)

def _return_parameter_names(num_pars):
    if num_pars==1:
        return ['tau_a']
    elif num_pars==2:
         return ['C_a', 'tau_a']
    elif num_pars==3:
         return ['S2_0', 'C_a', 'tau_a']
    elif num_pars==4:
         return ['C_a', 'tau_a', 'C_b', 'tau_b']
    elif num_pars==5:
         return ['S2_0', 'C_a', 'tau_a', 'C_b', 'tau_b']
    elif num_pars==6:
         return ['C_a', 'tau_a', 'C_b', 'tau_b', 'C_g', 'tau_g']
    elif num_pars==7:
         return ['S2_0', 'C_a', 'tau_a', 'C_b', 'tau_b', 'C_g', 'tau_g']
    elif num_pars==8:
         return ['C_a', 'tau_a', 'C_b', 'tau_b', 'C_g', 'tau_g', 'C_d', 'tau_d']
    elif num_pars==9:
         return ['S2_0', 'C_a', 'tau_a', 'C_b', 'tau_b', 'C_g', 'tau_g', 'C_d', 'tau_d']
    elif num_pars==10:
         return ['C_a', 'tau_a', 'C_b', 'tau_b', 'C_g', 'tau_g', 'C_d', 'tau_d', 'C_e', 'tau_e']
    elif num_pars==11:
         return ['S2_0', 'C_a', 'tau_a', 'C_b', 'tau_b', 'C_g', 'tau_g', 'C_d', 'tau_d', 'C_e', 'tau_e']

    return []



def sort_parameters(num_pars, params):
    if np.fmod( num_pars, 2 ) == 1:
        S2     = params[0]
        consts = [ params[k] for k in range(1,num_pars,2) ]
        taus   = [ params[k] for k in range(2,num_pars,2) ]
        Sf     = 1-params[0]-np.sum(consts)
    else:
        consts = [ params[k] for k in range(0,num_pars,2) ]
        taus   = [ params[k] for k in range(1,num_pars,2) ]
        S2     = 1.0 - np.sum( consts )
        Sf     = 0.0
    return S2, consts, taus, Sf

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
    #print ymodel

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
    #print ymodel

    return calc_chi(y, ymodel, dy), popt, np.sqrt(np.diag(popv)), ymodel

def scan_LSstyle_fits(x, y, dy=[]):
    chi_list=[]
    par_list=[]
    err_list=[]
    mod_list=[]
    name_list=[]
    for npars in range(1,10):
        chi, params, errors, ymodel = do_LSstyle_fit(npars, x, y, dy)
        names = _return_parameter_names(npars)
        chi_list.append(chi)
        par_list.append(params)
        err_list.append(errors)
        mod_list.append(ymodel)
        name_list.append(names)

    return chi_list, name_list, par_list, err_list, mod_list

def run_Expstyle_fits(x, y, dy, npars):
    names = _return_parameter_names(npars)
    try:
        chi, params, errors, ymodel = do_Expstyle_fit(npars, x, y, dy)
    except:
        print " ...fit returns an error! Continuing."

    return chi, names, params, errors, ymodel

#def findbest_LSstyle_fits(x, y, dy=[], bPrint=True):
def findbest_Expstyle_fits(x, y, dy=[], bPrint=True, par_list=[2,3,5,7,9], threshold=0.5):
    chi_min=np.inf
    # Search forwards
    for npars in par_list:
        names = _return_parameter_names(npars)
        try:
            chi, params, errors, ymodel = do_Expstyle_fit(npars, x, y, dy)
        except:
            print " ...fit returns an error! Continuing."
            break
        bBadFit=False
        for i in range(npars):
            if errors[i]>params[i]:
                print  " --- fit shows overfitting with %d parameters." % npars
                print  "  --- Occurred with parameter %s: %g +- %g " % (names[i], params[i], errors[i])
                bBadFit=True
                break
        if (not bBadFit) and chi/chi_min < threshold:
            chi_min=chi ; par_min=params ; err_min=errors ; npar_min=npars ; ymod_min=ymodel
        else:
            break

    if bPrint:
        names = _return_parameter_names(npar_min)
        print "= = Found %d parameters to be the minimum necessary to describe curve: chi(%d) = %g vs. chi(%d) = %g)" % (npar_min, npar_min, chi_min,  npars, chi)
        S2_all=1.0
        for i in range(npar_min):
            print "Parameter %d %s: %g +- %g " % (i, names[i], par_min[i], err_min[i])
            if 'S2' in names[i]:
                S2_all=S2_all*par_min[i]
        #print "Overall S2: %g" % S2_all
        # Special case for 2:
        if npar_min == 2:
            S2_all= 1.0 - par_min[0]

    return chi_min, names, par_min, err_min, ymod_min

