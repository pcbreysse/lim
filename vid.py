'''
Module for computing one-point statistics of line intensity maps.

VID classes are subclasses of LineObs, as they require knowledge of a voxel
volume, which in turn requires an instrument definition.
'''

import numpy as np
import astropy.units as u
from scipy.stats import poisson
from scipy.interpolate import interp1d
from scipy.misc import factorial as fact

from line_obs import LineObs
from _utils import cached_property,ulogspace,get_default_params,check_params
import luminosity_functions as lf
import _vid_tools as vt


class VID(LineObs):
    '''
    An object which computes the VID and related quantities for a given
    LineModel and instrument.  Includes the caching and updating functionality
    of LineModel.  Keep in mind that cached properties will not update if
    inputs are changed using anything other than the update() method.
    
    This module uses the lognormal VID model from Breysse et al. 2017
    
    Currently only available for 'LF' model types
    
    This Class outputs the VID in two forms, a probability distribution PT
    computed at nT points, and a binned histogram Bi computed at Nbin_hist
    points.  Each of these can have either linear or log spaced bins.  For the
    PDF, bins are linear if do_fast=True and logarithmic if do_fast=False, as
    the linearly-binned VID can be computed using FFT's, while the log-binned
    version must use brute force integration.  The interpolated histogram Bi
    will have bin types set by the value of linear_bin.
    
    In general, the linearly-binned FFT method should be used whenever it is
    possible to set nT high enough to capture all of the dynamic range of 
    dn/dL.  For models spanning a high range of L, the brute-force method will
    allow the VID to be computed with far fewer T bins, effectively improving
    memory usage at the cost of computation time.
    
    There are many ways for numerical errors to creep into the VID
    convolutions, especially when using FFT's.  To combat this, this Class
    includes a method DrawTest, which produces a sample histogram manually
    drawn from the lognormal number count PDF and given luminosity function.
    Comparing the output of this method to self.Bi provides a check on these
    numerical errors.
    
    TODO:
    Add function to compute sigma_G from power spectrum
    Add routine for computing dn/dL from non-monotonic ML models
    Add function dealing with scatter in such models
    
    INPUT PARAMETERS:
    Tmin:           Minimum voxel intensity for computation in temperature
                    units if do_Jysr=False, intensity units if do_Jysr=True.
                    (Default = 1e-2 uK)
    
    Tmax:           Maximum voxel intensity for computation (Default = 
                    1000*u.uK)
                    
    nT:             Number of T bins for PDF computation (Default = 1e5, 
                    should be set much lower if do_fast=False)
    
    do_fast:        Bool, if True, PDF is computed with linearly spaced bins
                    using FFT convolutions (Default = True)
                    
    sigma_G:        Width parameter of lognormal number count PDF, in the
                    future this will be computed automatically from the power
                    spectrum (Default = 1.6)
                    
    Nmax:           Maximum number of sources/voxel, sets the maximum number
                    of convolutions.  Should be set high enough so that the
                    VID converges over the desired temperature range (Default
                    = 100)
                    
    Nbin_hist:      Number of bins in predicted histogram (Default = 101)
                    
    subtract_mean:  Bool, if True PDF's and histograms are reported for a map
                    with the mean subtracted (Default = False)

    linear_bin:     Bool, if True the predicted histogram is computed for
                    linear intensity bins (Default = False)
                    
    DOCTESTS:
    Test parameters chosen to make a VID model which can be computed quickly
    and accurately, most notably increasing Lmin from the default
    
    >>> model_par = {'phistar':8.7e-11*u.Lsun**-1*u.Mpc**-3,\
                    'Lstar':2.1e6*u.Lsun,'alpha':-1.87,'Lmin':5000.*u.Lsun}
    >>> m = VID(model_par=model_par)
    >>> m.Nbar
    <Quantity 0.791...>
    >>> m.PT[10:12]
    <Quantity [ 0.311..., 0.318...] 1 / uK>
    >>> m.normalization
    <Quantity 0.99994...>
    >>> m.Bi[10:12]
    array([  89.03...,  171.7...])
    '''
    
    def __init__(self,Tmin=1.0e-2*u.uK, Tmax=1000.*u.uK, nT=10**5,
                 do_fast=True, sigma_G=1.6, Nmax=100, Nbin_hist=101,
                 subtract_mean=False,linear_bin=False,**obs_kwargs):
                 
        LineObs.__init__(self,**obs_kwargs)
        
        # Get input parameters and check that they are valid
        self._vid_params = locals()
        self._vid_params.pop('self')
        self._vid_params.pop('obs_kwargs')
        self._default_vid_params = get_default_params(VID.__init__)
        check_params(self._vid_params,self._default_vid_params)
        
        if self.model_type!='LF':
            raise ValueError('VID computations only available for LF models')
        
        # Set VID parameters
        for key in self._vid_params:
            setattr(self,key,self._vid_params[key])
            
        # Combine vid_params with obs_params
        self._params.update(self._vid_params)
        self._default_params.update(self._default_vid_params)
        

    ##################
    # Intensity bins #
    ##################
    @cached_property
    def Tedge(self):
        '''
        Edges of intensity bins. Uses linearly spaced bins if do_fast=True,
        logarithmically spaced if do_fast=False
        '''
        if self.do_fast:
            Te = np.linspace(self.Tmin,self.Tmax,self.nT+1)
        else:
            Te = ulogspace(self.Tmin,self.Tmax,self.nT+1)
            
        if self.subtract_mean:
            return Te-self.Tmean
        else:
            return Te
            
        
    @cached_property
    def T(self):
        '''
        Centers of intensity bins
        '''
        return vt.binedge_to_binctr(self.Tedge)
        
    @cached_property
    def dT(self):
        '''
        Widths of intensity bins
        '''
        return np.diff(self.Tedge)
        
    ######################################### 
    # Number count probability distribution #
    #########################################
    @cached_property
    def Nbar(self):
        '''
        Mean number of galaxies per voxel
        '''
        return self.nbar*self.Vvox
        
    @cached_property
    def Ngal(self):
        '''
        Vector of galaxy number counts, from 0 to self.Nmax
        '''
        return np.array(range(0,self.Nmax+1))
        
    
    @cached_property
    def PofN(self):
        '''
        Probability of a voxel containing N galaxies.  Uses the lognormal +
        Poisson model from Breysse et al. 2017
        '''
        # PDF of galaxy density field mu
        logMuMin = np.log10(self.Nbar)-20*self.sigma_G
        logMuMax = np.log10(self.Nbar)+5*self.sigma_G
        mu = np.logspace(logMuMin,logMuMax,10**4.)
        mu2,Ngal2 = np.meshgrid(mu,self.Ngal) # Keep arrays for fast integrals
        Pln = vt.lognormal_Pmu(mu2,self.Nbar,self.sigma_G)

        P_poiss = poisson.pmf(Ngal2,mu2)
                
        return np.trapz(P_poiss*Pln,mu)
        
    ###################
    # Intensity PDF's #
    ###################
    @cached_property
    def XLT(self):
        '''
        Constant relating total luminosity in a voxel to its observed
        intensity.  Equal to CLT/Vvox
        '''
        return self.CLT/self.Vvox
        
    
    @cached_property
    def P1(self):
        '''
        Probability of observing a given intensity in a voxel which contains
        exactly one emitter
        '''
        # Compute dndL at L's equivalent to T bins
        dndL_T = lambda L: getattr(lf,self.model_name)(L, self.model_par)
        if self.subtract_mean:
            return dndL_T((self.T+self.Tmean)/self.XLT)/(self.nbar*self.XLT)
        else:
            return dndL_T(self.T/self.XLT)/(self.nbar*self.XLT)
        
    @cached_property
    def PT(self):
        '''
        Probability of intensity between T and T+dT in a given voxel.  Uses
        fft's and linearly spaced T points if do_fast=1.  Uses brute-force
        convolutions and logarithmically spaced T points if do_fast=0
        
        Does NOT include the delta function at T=0 from voxels containing zero
        sources.  That is handled by self.PT_zero, which is later taken into
        account when computing B_i. This means that PT will not integrate to
        unity, but rather 1-PT_zero.
        '''
        if self.do_fast:
            fP1 = np.fft.fft(self.P1)*self.dT
            # FT of PDF should be dimensionless, but the fft function removes
            # the unit from P1
            fP1 = ((fP1*self.P1.unit).decompose()).value 
            
            fPT_N = np.zeros((self.Nmax,self.T.size),dtype=complex)
            
            for ii in range(1,self.Nmax):
                fPT_N[ii,:] = fP1**(ii)*self.PofN[ii]
            
            fPT = fPT_N.sum(axis=0)
            
            # Errors in fft's leave a small imaginary part, remove for output
            return (np.fft.ifft(fPT)/self.dT).real
            
        else:
            P_N = np.zeros([self.Nmax,self.T.size])*self.P1.unit
            P_N[0,:] = self.P1
            
            for ii in range(1,self.Nmax):
                PN[ii,:] = vt.conv_parallel(self.T,P_N[ii-1],
                                            self.T,self.P1,self.T)
            
            PT = np.zeros(self.T.size)

            for ii in range(0,self.Nmax):
                PT = PT+PN[ii,:]*self.PofN[ii+1]
                
            return PT
            
    @cached_property
    def PT_zero(self):
        '''
        P(T) contains a delta function at T=0 from voxels which contain zero 
        sources.  Delta functions are difficult to include naturally in
        arrays, so we model it separately here.  This quantity will need to be
        taken into account for any integrals over P(T) which cover T=0. (See
        the self.normalization function below)
        '''
        return self.PofN[0]
        
    @cached_property
    def normalization(self):
        '''
        Outputs the value of integral(P(T)dT) including the spike at T=0.
        Used as a numerical check, should come out quite close to 1.0
        '''
        return np.trapz(self.PT,self.T)+self.PT_zero
        
    ########################
    # Predicted histograms #
    ########################
                                            
    @cached_property
    def Tedge_i(self):
        '''
        Edges of histogram bins
        '''
        if self.linear_bin:
            Te = np.linspace(-self.Tmax,self.Tmax,self.Nbin_hist+1)
        else:
            Te = ulogspace(self.Tmin,self.Tmax,self.Nbin_hist+1)
        
        if self.subtract_mean:
            return Te-self.Tmean
        else:
            return Te
        
    @cached_property
    def Ti(self):
        '''
        Centers of histogram bins
        '''
        return vt.binedge_to_binctr(self.Tedge_i)
        
    @cached_property
    def Bi(self):
        '''
        Predicted number of sources in each bin
        '''
        if self.subtract_mean:
            return vt.pdf_to_histogram(self.T,self.PT,self.Tedge_i,self.Nvox,
                                        self.Tmean,self.PT_zero)
        else:
            return vt.pdf_to_histogram(self.T,self.PT,self.Tedge_i,self.Nvox,
                                        0.*self.Tmean.unit,self.PT_zero)
                                        
    ######################################
    # Draw galaxies to test convolutions #    
    ######################################
    
    def DrawTest(self,Ndraw):
        '''
        Function which draws sample galaxy populations from input number count
        and luminosity distributions.  Outputs Ndraw histograms which can be
        compared to self.Bi
        '''
        h = np.zeros([Ndraw,self.Ti.size])
        
        # PofN must be exactly normalized
        PofN = self.PofN/self.PofN.sum()
        
        for ii in range(0,Ndraw):
            # Draw number of galaxies in each voxel
            N = np.random.choice(self.Ngal,p=PofN,size=self.Nvox.astype(int))
            
            # Draw galaxy luminosities
            Ledge = np.logspace(0,10,10**4+1)*u.Lsun
            Lgal = vt.binedge_to_binctr(Ledge)
            dL = np.diff(Ledge)
            PL = getattr(lf,self.model_name)(Lgal,self.model_par)*dL
            PL = PL/PL.sum() # Must be exactly normalized
            
            T = np.zeros(self.Nvox.astype(int))*u.uK
            
            for jj in range(0,self.Nvox):
                if N[jj]==0:
                    L = 0.*u.Lsun
                else:
                    L = np.random.choice(Lgal,p=PL,size=N[jj])*u.Lsun
                    
                T[jj] = self.XLT*L.sum()
            
            h[ii,:] = np.histogram(T,bins=self.Tedge_i)[0]
        
        if Ndraw==1:
            # For simplicity of later use, returns a 1-D array if Ndraw=1
            return h[0,:]
        else:
            return h
            
############
# Doctests #
############

if __name__ == "__main__":
    import doctest

    doctest.testmod(optionflags=doctest.ELLIPSIS |
                    doctest.NORMALIZE_WHITESPACE)
