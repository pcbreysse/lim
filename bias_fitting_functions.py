"""
Calculate the halo bias depending on the mass for different fitting functions

All functions return b(M)

Bias model parameter values are given by a dictionary called bias_par.  Each
function takes a value of dc, nu
"""

import numpy as np

def Tinker10(self,dc,nu):
    # Parameters of bias fit
    if len(self.bias_par.keys()) == 0:
        y = np.log10(200.)
        A = 1. + 0.24*y*np.exp(-(4./y)**4.)
        a = 0.44*y - 0.88
        B = 0.183
        b = 1.5
        C = 0.019 + 0.107*y + 0.19*np.exp(-(4./y)**4.)
        c = 2.4
    else:
        y = self.bias_par['y']
        B = self.bias_par['B']
        b = self.bias_par['b']
        c = self.bias_par['c']
        A = 1. + 0.24*y*np.exp(-(4./y)**4.)
        C = 0.019 + 0.107*y + 0.19*np.exp(-(4./y)**4.)
        a = 0.44*y - 0.88
    
    return 1.- A*nu**a/(nu**a+dc**a) + B*nu**b + C*nu**c
    #return 1.+(nu**2.-1.)/dc
            
def Mo96(self,dc,nu):
    """
    Peak-background split bias correspdonding to PS HMF.
    
    Taken from Mo and White (1996)
    """
    return 1. + (nu**2.-1.)/dc
            
def Jing98(self,dc,nu):
    """
    Empirical bias of Jing (1998): http://adsabs.harvard.edu/abs/1998ApJ...503L...9J
    """
    Mh = (self.M.to(self.Msunh)).value
    nu_star = (Mh/self.mass_non_linear)**(self.cosmo_input['ns']+3.)/6.
    if len(self.bias_par.keys()) == 0:
        a = 0.5
        b = 0.06
        c = 0.02
    else:
        a = self.bias_par['a']
        b = self.bias_par['b']
        c = self.bias_par['c']
    return (a/nu_star**4. + 1.)**(b-c*self.cosmo_input['ns']) * \
                (1.+(nu_star**2. -1.)/dc)
                                
def ST99(self,dc,nu):
    """
    Peak-background split bias corresponding to ST99 HMF.
    
    Taken from Sheth & Tormen (1999).
    """
    if len(self.bias_par.keys()) == 0:
        q = 0.707
        p = 0.3
    else:
        q = self.bias_par['q']
        p = self.bias_par['p']
    return 1. + (q*nu**2-1.)/dc + (2.*p/dc)/(1.+(q*nu**2)**p)
            
def SMT01(self,dc,nu):
    """
    Extended Press-Schechter-derived bias function corresponding to SMT01 HMF
    
    Taken from Sheth, Mo & Tormen (2001)
    """
    if len(self.bias_par.keys()) == 0:
        a = 0.707
        b = 0.5
        c = 0.6
    else: 
        a = self.bias_par['a']
        b = self.bias_par['b']
        c = self.bias_par['c']
    sa = a**0.5
    return 1.+(sa*(a*nu**2.) + sa*b*(a*nu**2.)**(1.-c) - \
                    (a*nu**2.)**c/((a*nu**2.)**c + \
                    b*(1.-c)*(1.-c/2.)))/(dc*sa)
                         
def Seljak04(self,dc,nu):
    """
    Empirical bias relation from Seljak & Warren (2004), without cosmological dependence.
    """
    Mh = (self.M.to(self.Msunh)).value
    x = Mh/self.mass_non_linear
    if len(self.bias_par.keys()) == 0:
        a = 0.53
        b = 0.39
        c = 0.45
        d = 0.13
        e = 40.
        f = 5e-4
        g = 1.5
    else:
        a = self.bias_par['a']
        b = self.bias_par['b']
        c = self.bias_par['c']
        d = self.bias_par['d']
        e = self.bias_par['e']
        f = self.bias_par['f']
        g = self.bias_par['g']
    return a + b*x**c + d/(e*x+1.) + f*x**g
            
def Seljak04_Cosmo(self,dc,nu):
    """
    Empirical bias relation from Seljak & Warren (2004), with cosmological dependence.
    Doesn't include the running of the spectral index alpha_s.
    """
    Mh = (self.M.to(self.Msunh)).value
    x = Mh/self.mass_non_linear
    if len(self.bias_par.keys()) == 0:
        a = 0.53
        b = 0.39
        c = 0.45
        d = 0.13
        e = 40.
        f = 5e-4
        g = 1.5
        a1 = 0.4
        a2 = 0.3
        a3 = 0.8
    else:
        a = self.bias_par['a']
        b = self.bias_par['b']
        c = self.bias_par['c']
        d = self.bias_par['d']
        e = self.bias_par['e']
        f = self.bias_par['f']
        g = self.bias_par['g']
        a1 = self.bias_par['a1']
        a2 = self.bias_par['a2']
        a3 = self.bias_par['a3']
    return a + b*x**c + d/(e*x+1.) + f*x**g + np.log10(x)*        \
            (a1*(self.camb_pars.omegam - 0.3 + self.cosmo_input['ns'] - 1.) +   \
            a2*(transfer.sigma_8[0]-0.9 + self.hubble - 0.7) + a4*self.cosmo_input['nrun'])
                   
def Tinker05(self,dc,nu):
    """
    Empirical bias, same as SMT01 but modified parameters.
    """
    if len(self.bias_par.keys()) == 0:
        a = 0.707
        b = 0.35
        c = 0.8
    else:
        a = self.bias_par['a']
        b = self.bias_par['b']
        c = self.bias_par['c']
    sa = a**0.5
    return 1.+(sa*(a*nu**2) + sa*b*(a*nu**2)**(1.-c) - (a*nu**2)**c/((a*nu**2)**c +    \
                    b*(1.-c)*(1.-c/2.)))/(dc*sa)
            
def Mandelbaum05(self,dc,nu):
    """
    Empirical bias, same as ST99 but changed parameters
    """
    if len(self.bias_par.keys()) == 0:
        q = 0.73
        p = 0.15
    else:
        q = self.bias_par['q']
        p = self.bias_par['p']
    return 1. + (q*nu**2.-1.)/dc + (2.*p/dc)/(1.+(q*nu**2.)**p)
            
def Manera10(self,dc,nu):
    """
    Empirical bias, same as ST99 but changed parameters
    """
    if len(self.bias_par.keys()) == 0:
        q = 0.709
        p = 0.248
    else:
        q = self.bias_par['q']
        p = self.bias_par['p']
    return 1. + (q*nu**2.-1.)/dc + (2.*p/dc)/(1.+(q*nu**2.)**p)
