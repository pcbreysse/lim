'''
Module for computing details of line intensity mapping observations
'''

import numpy as np
import astropy.units as u
import astropy.constants as cu
from scipy.special import erf

from lim import LineModel
from _utils import cached_property,get_default_params,check_params,check_model

class LineObs(LineModel):
    '''
    An object containing a line intensity model as well as tools to compute
    aspects of an experimental observation of said model.
    
    The base class of LineObs is lim.LineModel, so it has all of the
    attributes of a LineModel, including the caching and update functionality.
    Keep in mind that cached_properties will not update if inputs are changed
    using anything but the update() method.
    
    In all cases, "pixel" refers to a two-dimensional area and "voxel" refers
    to a three-dimensional volume.
    
    Default input parameter values are for the COMAP1 CO intensity mapping
    instrument.
    
    INPUT PARAMETERS:
    Tsys:           Instrument system temperature (Default = 40 K)
    
    Nfeeds:         Number of feeds (Default = 19)
    
    beam_FWHM:      Beam full width at half maximum (Default = 4.1")
    
    Delta_nu:      Total frequency range covered by instrument (Default = 8 GHz)
    
    dnu:            Width of a single frequency channel (Default = 15.6 MHz)
    
    tobs:           Observing time on a single field (Default = 6000 hr)
    
    Omega_field:    Solid angle covered by a single field
                    (Default = 2.25 deg^2)    
    
    Nfield:         Number of fields observed (Default = 1)
    
    **line_kwargs:  Input parameters of LineModel()
    
    DOCTESTS:
    >>> m = LineObs()
    >>> m.Pk[0:2]
    <Quantity [ 98477..., 98741...] Mpc3 uK2>
    >>> m.Nvox
    <Quantity 1387152.0>
    >>> m.sk[0:2]
    <Quantity [ 472891..., 427400... ] Mpc3 uK2>
    >>> m.SNR
    <Quantity 16.9...>
    '''
    
    def __init__(self, Tsys=40*u.K, Nfeeds=19, beam_FWHM=4.1*u.arcmin, 
                    Delta_nu=8*u.GHz, dnu=15.6*u.MHz, tobs=6000*u.hr, 
                    Omega_field=2.25*u.deg**2, Nfield=1,**line_kwargs):
                    
        # Initiate LineModel() parameters
        #super(LineObs, self).__init__(**line_kwargs) # PROBLEM WITH autoreload
        LineModel.__init__(self,**line_kwargs)
        
        self._obs_params = locals()
        self._obs_params.pop('self')
        self._obs_params.pop('line_kwargs')
        self._default_obs_params = get_default_params(LineObs.__init__)
        check_params(self._obs_params,self._default_obs_params)
        
        # Set instrument parameters
        for key in self._obs_params:
            setattr(self,key,self._obs_params[key])
        
        # Combine lim_params with obs_params
        self._params.update(self._obs_params)
        self._default_params.update(self._default_obs_params)
        
    ##############
    # Field Size #
    ##############
    
    @cached_property
    def Nch(self):
        '''
        Number of frequency channels, rounded if dnu does not divide evenly
        into Delta_nu
        '''
        return np.round((self.Delta_nu/self.dnu).decompose())
        
    @cached_property
    def beam_width(self):
        '''
        Beam width defined as 1-sigma width of Gaussian beam profile
        '''
        return self.beam_FWHM*0.4247
        
    @cached_property
    def Nside(self):
        '''
        Number of pixels on a side of a map.  Pixel size is assumed to be one
        beam FWHM on a side.  Rounded if FWHM does not divide evenly into
        sqrt(Omega_field)
        '''
        theta_side = np.sqrt(self.Omega_field)
        return np.round((theta_side/self.beam_width).decompose())
    
    @cached_property
    def Npix(self):
        '''
        Number of pixels in a map
        '''
        return self.Nside**2
        
    @cached_property
    def Nvox(self):
        '''
        Number of voxels in a map
        '''
        return self.Npix*self.Nch
        
    @cached_property
    def fsky(self):
        '''
        Fraction of sky covered by a field
        '''
        fsky = (self.Omega_field/(4*np.pi*u.rad)).decompose()
    
    @cached_property
    def r0(self):
        '''
        Comoving distance to central redshift of field
        '''
        return self.h.cosmo.comoving_distance(self.z)
    
    @cached_property    
    def Vfield(self):
        '''
        Comoving volume of a single field
        '''
        return (self.r0**2*(self.Omega_field/(1.*u.rad**2))*cu.c*self.Delta_nu
                *(1+self.z)**2/(self.H*self.nu)).to(u.Mpc**3)
    
    @cached_property            
    def Vvox(self):
        '''
        Comoving volume of a single voxel
        '''
        return self.Vfield/self.Nvox
        
    ##########################
    # Instrument noise power #
    ##########################
            
    @cached_property
    def tpix(self):
        '''
        Time spent observing each pixel
        '''
        return self.tobs/self.Npix
    
    @cached_property
    def sigma_N(self):
        '''
        Instrumental noise per voxel
        '''
        sig = self.Tsys/np.sqrt(self.Nfeeds*self.dnu*self.tpix)
        return sig.to(u.uK)
    
    @cached_property    
    def Pnoise(self):
        '''
        Noise power spectrum amplitude
        '''
        return self.sigma_N**2*self.Vvox
        
    @cached_property
    def sigma_par(self):
        '''
        High-resolution cutoff for line-of-sight modes
        '''
        return (cu.c*self.dnu*(1+self.z)/(self.H*self.nuObs)).to(u.Mpc)
    
    @cached_property
    def sigma_perp(self):
        '''
        High-resolution cutoff for transverse modes
        '''
        return (self.r0*(self.beam_width/(1*u.rad))).to(u.Mpc)
        
    @cached_property
    def Wk(self):
        '''
        Resolution cutoff in power spectrum
        '''
        mu = np.linspace(0,1,1000)
        ki,mui = np.meshgrid(self.k,mu)
        exparg1 = -(self.k**2*self.sigma_perp**2).decompose()
        exparg2 = -((ki**2*(self.sigma_par**2-self.sigma_perp**2)*mui**2)
                    .decompose())
        return np.exp(exparg1)*np.trapz(np.exp(exparg2),mu,axis=0)
        
    @cached_property
    def Nmodes(self):
        '''
        Number of modes between k and k+dk
        '''
        return self.k**2.*self.dk*self.Vfield*self.Nfield/(4*np.pi**2)
        
    @cached_property
    def sk_CV(self):
        '''
        Error at k due to sample variance
        '''
        return self.Pk/(np.sqrt(self.Nmodes)*self.Wk)
        
    @cached_property
    def sk_N(self):
        '''
        Error at k due to instrumental noise
        '''
        return self.Pnoise/(np.sqrt(self.Nmodes)*self.Wk)
        
    @cached_property
    def sk(self):
        '''
        Total error at k
        '''
        return self.sk_CV+self.sk_N
        
    @cached_property
    def kmin_field(self):
        '''
        Minimum k accessible in a single field, set by the minimum side length
        '''
        # Line-of-sight side length
        z_min = self.nu/(self.nuObs+self.Delta_nu/2.)-1
        z_max = self.nu/(self.nuObs-self.Delta_nu/2.)-1
        dr_los = (self.h.cosmo.comoving_distance(z_max)-
                    self.h.cosmo.comoving_distance(z_min))
        kmin_los = 2*np.pi/dr_los
        # Transverse side length
        dr_sky = (np.sqrt((self.Omega_field/(1*u.rad**2)).decompose())
                    *self.h.cosmo.angular_diameter_distance(self.z))
        kmin_sky = 2*np.pi/dr_sky
        return min([kmin_los,kmin_sky])
        
    @cached_property
    def SNR(self):
        '''
        Signal to noise ratio for given model and experiment
        '''
        SNR_k = (self.Pk**2/self.sk**2).decompose()
        return np.sqrt(SNR_k[self.k>=self.kmin_field].sum())
        
############
# Doctests #
############

if __name__ == "__main__":
    import doctest

    doctest.testmod(optionflags=doctest.ELLIPSIS |
                    doctest.NORMALIZE_WHITESPACE)
