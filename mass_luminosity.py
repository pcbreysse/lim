"""
Calculate Mass-Luminosity relations for different models of line
emission.

All functions take a vector of masses in M_sun and return luminosities
in L_sun.

Model parameter values are given by a dictionary called MLpar.  Each
function also takes a value of the redshift z even if the L(M) model is not
redshift dependent.  This allows the functions to be called consistently by
LineModel()

TODO:
Add in models from Matlab code
"""

import numpy as np
import astropy.units as u
import astropy.constants as cu
from scipy.interpolate import interp2d,interp1d

def MassPow(Mvec, MLpar, z):
    """
    Power law L(M)/L_sun = A*(M/M_sun)^b (See Breysse et al. 2015)

    Parameters:
    A	      Overall amplitude, dimensionless
    b         Power law slope, dimensionless
    Mcut_min  Minimum mass below which L=0 (in M_sun)
    Mcut_max  Maximum mass above which L=0 (in M_sun)
    
    Assumed to be redshift independent

    >>> Mvec = np.array([1.e8,1.e10,1.e16]) * u.Msun
    >>> MLpar = {'A':2e-6, 'b':1., \
        'Mcut_min':1e9*u.Msun, 'Mcut_max':1e15*u.Msun}
    >>> z = 3.0
    >>> print MassPow(Mvec,MLpar,z)
    [     0.  20000.      0.] solLum
    """

    A = MLpar['A']
    b = MLpar['b']
    Mcut_min = MLpar['Mcut_min']
    Mcut_max = MLpar['Mcut_max']

    L = A * np.array(Mvec)**b*u.Lsun

    L[Mvec < Mcut_min] = 0.*u.Lsun
    L[Mvec > Mcut_max] = 0.*u.Lsun

    return L
    
def DblPwr(Mvec, MLpar, z):
    """
    Double power law with redshift dependence 
    L(M)/Lsun = A * 10^(b1*z) * (M/1e8 Msun)^b2 * (1+M/M_*)^b3
    
    Parameters:
    A	      Overall amplitude, dimensionless
    b1        Redshift slope, dimensionless
    b2        Low mass power law, dimensionless
    b3        High mass power law, dimensionless
    Mstar     Power law turnover mass, in M_sun
    Mcut_min  Minimum mass below which L=0 (in M_sun)
    Mcut_max  Maximum mass above which L=0 (in M_sun)
    
    >>> Mvec = np.array([1.e8,1.e10,1.e16]) * u.Msun
    >>> MLpar = {'A':5.8e-3, 'b1':0.35, 'b2':1.97, 'b3':-2.92, \
        'Mstar':8.e11*u.Msun, 'Mcut_min':1e9*u.Msun, 'Mcut_max':1e15*u.Msun}
    >>> z = 3.0
    >>> print DblPwr(Mvec,MLpar,z)
    [     0.  546.60...      0.        ] solLum
    """
    
    A = MLpar['A']
    b1 = MLpar['b1']
    b2 = MLpar['b2']
    b3 = MLpar['b3']
    Mstar = MLpar['Mstar']
    Mcut_min = MLpar['Mcut_min']
    Mcut_max = MLpar['Mcut_max']
    
    L = A * 10.**(b1*z) * (Mvec/(1.e8*u.Msun))**b2 * (1.+Mvec/Mstar)**b3
    L = L*u.Lsun
    
    L[Mvec < Mcut_min] = 0.*u.Lsun
    L[Mvec > Mcut_max] = 0.*u.Lsun
    
    return L
    
def TonyLi(Mvec, MLpar, z):
    '''
    CO emission model from Li et al. (2016).  Uses Behroozi et al. SFR(M)
    results.
    
    NOTE ON THIS MODEL: The Li et al. model has two types of scatter: one on
    SFR(M) and one on LCO(SFR), denoted as sigma_SFR and sigma_LCO.  The LCO
    scatter should be entered into LineModel() as the usual sigma_scatter
    input.  However, the SFR scatter behaves differently in that it does not
    preserve mean(LCO), but preserves mean(SFR) instead.  Thus it should be
    given as part of MLpar, there are specific hacks added to LineModel() to
    account for this.
    
    Parameters:
    alpha         Slope of logLIR/logLCO relation, dimensionless
    beta          Intercept of logLIR/logLCO relation, dimensionless
    dMF           10^10 times SFR/LIR normalization (See Li et al. Eq 1), 
                    dimensionless
    BehrooziFile  Filename where Behroozi et al. data is stored, default
                    'sfr_release.dat'. File can be downloaded from
                    peterbehroozi.com/data, (string)
    Mcut_min  Minimum mass below which L=0 (in M_sun)
    Mcut_max  Maximum mass above which L=0 (in M_sun)
    
    >>> Mvec = np.array([1.e8,1.e10,1.e16]) * u.Msun
    >>> MLpar = {'alpha':1.17, 'beta':0.21, 'dMF':1.0,\
        'BehrooziFile':'sfr_release.dat', 'Mcut_min':1e9*u.Msun,\
        'Mcut_max':1e15*u.Msun}
    >>> z = 3.0
    >>> print TonyLi(Mvec,MLpar,z)
    [     0.  205.14...      0.        ] solLum
    '''
    
    alpha = MLpar['alpha']
    beta = MLpar['beta']
    dMF = MLpar['dMF']
    BehrooziFile = MLpar['BehrooziFile']
    Mcut_min = MLpar['Mcut_min']
    Mcut_max = MLpar['Mcut_max']
    
    # Read and interpolate Behroozi SFR(M) data
    x = np.loadtxt(BehrooziFile)
    zb = np.unique(x[:,0])-1.
    logMb = np.unique(x[:,1])
    logSFRb = x[:,2].reshape(137,122,order='F')
    
    logSFR_interp = interp2d(logMb,zb,logSFRb,bounds_error=False,fill_value=0.)
    
    # Compute SFR(M) in Msun/yr
    logM = np.log10((Mvec.to(u.Msun)).value)
    SFR = 10.**logSFR_interp(logM,z)
    
    # Compute IR luminosity in Lsun
    LIR = SFR/(dMF*1e-10)
    
    # Compute L'_CO in K km/s pc^2
    Lprime = (10.**-beta * LIR)**(1./alpha)
    
    # Compute LCO
    L = (4.9e-5*u.Lsun)*Lprime
    
    L[Mvec < Mcut_min] = 0.*u.Lsun
    L[Mvec > Mcut_max] = 0.*u.Lsun
    
    
    return L
    
def SilvaCII(Mvec, MLpar, z):
    '''
    Silva et al. (2015) CII model, relates CII luminosity and SFR by
    log10(L_CII/Lsun) = a_LCII*log10(SFR/(Msun/yr)) + b_LCII
    
    SFR(M) computed from the double power law fit in their Eq. (8), with
    parameters interpolated from their Table 2.
    
    Note that the L(M) relations derived from this model are a variant on the
    DblPwr model above, but with the input parameters changed to match the
    Silva et al. numbers
    
    Parameters:
    a   a_LCII parameter in L(SFR), dimensionless
    b   b_LCII parameter in L(SFR)
    Mcut_min  Minimum mass below which L=0 (in M_sun)
    Mcut_max  Maximum mass above which L=0 (in M_sun)
    
    >>> Mvec = np.array([1.e8,1.e10,1.e16]) * u.Msun
    >>> MLpar = {'a':0.8475, 'b':7.2203, 'Mcut_min':1e9*u.Msun, \
                    'Mcut_max':1e15*u.Msun}
    >>> z = 7.5
    >>> print SilvaCII(Mvec,MLpar,z)
    [     0.  4587712...      0.        ] solLum
    '''
    
    aLCII = MLpar['a']
    bLCII = MLpar['b']
    Mcut_min = MLpar['Mcut_min']
    Mcut_max = MLpar['Mcut_max']
    
    # Interpolate SFR from Table 2 of Silva et al. 2015
    SFR = Silva_SFR(Mvec,z)
    
    # LCII relation
    L = 10**(aLCII*np.log10(SFR/(1*u.Msun/u.yr))+bLCII)*u.Lsun
    
    # Mmin, Mmax cuts
    L[Mvec < Mcut_min] = 0.*u.Lsun
    L[Mvec > Mcut_max] = 0.*u.Lsun
    
    return L
    

###################
# Other functions #
###################

def Silva_SFR(M,z):
    '''
    Returns SFR(M,z) interpolated from values in Table 2 of Silva et al. 2015
    '''
    x = np.loadtxt('Silva15_SFR_params.dat')
    
    z0 = x[0,:]
    M0 = interp1d(z0,x[1,:])(z)*u.Msun/u.yr
    Ma = interp1d(z0,x[2,:])(z)*u.Msun
    Mb = interp1d(z0,x[3,:])(z)*u.Msun
    a = interp1d(z0,x[4,:])(z)
    b = interp1d(z0,x[5,:])(z)
    
    return M0*(M/Ma)**a*(1+M/Mb)**b
    

if __name__ == "__main__":
    import doctest

    doctest.testmod(optionflags=doctest.ELLIPSIS |
                    doctest.NORMALIZE_WHITESPACE)
