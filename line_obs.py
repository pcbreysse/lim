'''
Module for computing details of line intensity mapping observations
'''

import numpy as np
import astropy.units as u
import astropy.constants as cu
from scipy.special import erf
from scipy.stats import poisson
from scipy.interpolate import interp1d
from scipy.misc import factorial as fact

from line_model import LineModel
from _utils import cached_property,get_default_params,check_params,check_model
from _utils import ulogspace
import _vid_tools as vt
import luminosity_functions as lf      


class LineObs(LineModel):
    '''
    An object containing a line intensity model as well as tools to compute
    aspects of an experimental observation of said model. This version is used
    for low-frequency instruments which measure intensity in brightness
    temperature units.
    
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
    >>> m.Pk[0:2]/1e5
    <Quantity [ 1.08..., 1.09...] Mpc3 uK2>
    >>> m.Nvox
    <Quantity 1387152.0>
    >>> m.sk[0:2]/1e5
    <Quantity [ 5.16..., 4.67...] Mpc3 uK2>
    >>> m.SNR
    <Quantity 17.8...>
    >>> m.PofN[0:2]
    <Quantity [ 0.305..., 0.181...]>
    >>> m.Bi[3:5]/1000
    array([ 3.97...,  9.54...])
    '''
    
    def __init__(self, 
                 Tsys_NEFD=40*u.K,
                 Nfeeds=19,
                 beam_FWHM=4.1*u.arcmin,
                 Delta_nu=8*u.GHz,
                 dnu=15.6*u.MHz,
                 tobs=6000*u.hr, 
                 Omega_field=2.25*u.deg**2,
                 Nfield=1,
                 Tmin_VID=1.0e-2*u.uK,
                 Tmax_VID=1000.*u.uK,
                 nT=10**5,
                 do_fast_VID=True,
                 sigma_G=1.6,
                 Ngal_max=100,
                 Nbin_hist=101,
                 subtract_VID_mean=False,
                 linear_VID_bin=False,
                 **line_kwargs):
                    
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
        self._input_params.update(self._obs_params)
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
        return (self.Omega_field/(4*np.pi*u.rad**2)).decompose()
    
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
        Time spent observing each pixel with a single detector
        '''
        return self.tobs/self.Npix
    
    @cached_property
    def sigma_N(self):
        '''
        Instrumental noise per voxel. Defined slightly differently depending
        on doJysr.
        '''
        if self.do_Jysr:
            return ((self.Tsys_NEFD/self.beam_width**2)
                    .to(u.Jy*u.s**(1./2)/u.sr))
        else:
            return ((self.Tsys_NEFD/np.sqrt(self.Nfeeds*self.dnu*self.tpix))
                    .to(u.uK))
    
    @cached_property    
    def Pnoise(self):
        '''
        Noise power spectrum amplitude
        '''
        if self.do_Jysr:
            return self.sigma_N**2*self.Vvox/(self.tpix*self.Nfeeds)
        else:
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

    #############################################
    #############################################
    ### Voxel Intensity Distribution Functions ##
    #############################################
    #############################################
    
    ##################
    # Intensity bins #
    ##################
    @cached_property
    def Tedge(self):
        '''
        Edges of intensity bins. Uses linearly spaced bins if do_fast_VID=True,
        logarithmically spaced if do_fast=False
        '''
        if self.do_fast_VID:
            Te = np.linspace(self.Tmin_VID,self.Tmax_VID,self.nT+1)
        else:
            Te = ulogspace(self.Tmin_VID,self.Tmax,self.nT+1)
            
        if self.subtract_VID_mean:
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
        Vector of galaxy number counts, from 0 to self.Ngal_max
        '''
        return np.array(range(0,self.Ngal_max+1))
        
    
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
        if self.subtract_VID_mean:
            return dndL_T((self.T+self.Tmean)/self.XLT)/(self.nbar*self.XLT)
        else:
            return dndL_T(self.T/self.XLT)/(self.nbar*self.XLT)
        
    @cached_property
    def PT(self):
        '''
        Probability of intensity between T and T+dT in a given voxel.  Uses
        fft's and linearly spaced T points if do_fast_VID=1.  Uses brute-force
        convolutions and logarithmically spaced T points if do_fast_VID=0
        
        Does NOT include the delta function at T=0 from voxels containing zero
        sources.  That is handled by self.PT_zero, which is later taken into
        account when computing B_i. This means that PT will not integrate to
        unity, but rather 1-PT_zero.
        '''
        if self.do_fast_VID:
            fP1 = np.fft.fft(self.P1)*self.dT
            # FT of PDF should be dimensionless, but the fft function removes
            # the unit from P1
            fP1 = ((fP1*self.P1.unit).decompose()).value 
            
            fPT_N = np.zeros((self.Ngal_max,self.T.size),dtype=complex)
            
            for ii in range(1,self.Ngal_max):
                fPT_N[ii,:] = fP1**(ii)*self.PofN[ii]
            
            fPT = fPT_N.sum(axis=0)
            
            # Errors in fft's leave a small imaginary part, remove for output
            return (np.fft.ifft(fPT)/self.dT).real
            
        else:
            P_N = np.zeros([self.Ngal_max,self.T.size])*self.P1.unit
            P_N[0,:] = self.P1
            
            for ii in range(1,self.Ngal_max):
                PN[ii,:] = vt.conv_parallel(self.T,P_N[ii-1],
                                            self.T,self.P1,self.T)
            
            PT = np.zeros(self.T.size)

            for ii in range(0,self.Ngal_max):
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
        if self.linear_VID_bin:
            Te = np.linspace(-self.Tmax_VID,self.Tmax_VID,self.Nbin_hist+1)
        else:
            Te = ulogspace(self.Tmin_VID,self.Tmax_VID,self.Nbin_hist+1)
        
        if self.subtract_VID_mean:
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
        if self.subtract_VID_mean:
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
