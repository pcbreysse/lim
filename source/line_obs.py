'''
Module for computing details of line intensity mapping observations
'''

import numpy as np
import astropy.units as u
import astropy.constants as cu
from scipy.interpolate import interp1d
from scipy.special import legendre

from source.line_model import LineModel
from source.tools._utils import cached_obs_property,cached_vid_property,get_default_params
from source.tools._utils import ulogspace, ulinspace,check_params,log_interp1d

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
    
    N_FG_par:       Multiplicative factor in the volume window for kmin_par
                    to account for foregrounds. Default = 1, No foregrounds
                    (only volume effects)
                    
    N_FG_perp:      Multiplicative factor in the volume window for kmin_perp
                    to account for foregrounds. Default = 1, No foregrounds
                    (only volume effects)
                    
    do_FG_wedge:    Apply foreground wedge removal. Default = False
    
    a_FG:           Constant superhorizon buffer for foregrounds. Default = 0
    
    b_FG:           Foreground parameter accounting for antenna chromaticity. Default = 0 
    
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
                 N_FG_par = 1,
                 N_FG_perp = 1,
                 do_FG_wedge = False,
                 a_FG = 0,
                 b_FG = 0,
                 **line_kwargs):
                    
        # Initiate LineModel() parameters
        LineModel.__init__(self,**line_kwargs)
        
        self._update_cosmo_list = self._update_cosmo_list
        
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
    
    @cached_obs_property
    def Nch(self):
        '''
        Number of frequency channels, rounded if dnu does not divide evenly
        into Delta_nu
        '''
        return np.round((self.Delta_nu/self.dnu).decompose())
        
        
    @cached_obs_property
    def beam_width(self):
        '''
        Beam width defined as 1-sigma width of Gaussian beam profile
        '''
        return self.beam_FWHM*0.4247
        
        
    @cached_obs_property
    def Nside(self):
        '''
        Number of pixels on a side of a map.  Pixel size is assumed to be one
        beam FWHM on a side.  Rounded if FWHM does not divide evenly into
        sqrt(Omega_field)
        '''
        theta_side = np.sqrt(self.Omega_field)
        return np.round((theta_side/self.beam_width).decompose())
    
    
    @cached_obs_property
    def Npix(self):
        '''
        Number of pixels in a map
        '''
        return self.Nside**2
        
        
    @cached_obs_property
    def Nvox(self):
        '''
        Number of voxels in a map
        '''
        return self.Npix*self.Nch
        
        
    @cached_obs_property
    def fsky(self):
        '''
        Fraction of sky covered by a field
        '''
        return (self.Omega_field/(4*np.pi*u.rad**2)).decompose()
    
    
    @cached_obs_property
    def r0(self):
        '''
        Comoving distance to central redshift of field
        '''
        if self.cosmo_code == 'camb':
            return self.cosmo.comoving_radial_distance(self.z)*u.Mpc
        else:
            return self.cosmo.angular_distance(self.z)*(1.+self.z)*u.Mpc
    
    
    @cached_obs_property
    def Sfield(self):
        '''
        Area of single field in the sky in Mpc**2
        '''
        return (self.r0**2*(self.Omega_field/(1.*u.rad**2))).to(u.Mpc**2)
        
        
    @cached_obs_property
    def Lfield(self):
        '''
        Depth of a single field
        '''
        z_min = (self.nu/(self.nuObs+self.Delta_nu/2.)-1).value
        z_max = (self.nu/(self.nuObs-self.Delta_nu/2.)-1).value
        if self.cosmo_code == 'camb':
            dr_los = (self.cosmo.comoving_radial_distance(z_max)-
                      self.cosmo.comoving_radial_distance(z_min))
        else:
            dr_los = (self.cosmo.angular_distance(z_max)*(1.+z_max)-
                      self.cosmo.angular_distance(z_min)*(1.+z_min))
        return dr_los*u.Mpc
                
                
    @cached_obs_property    
    def Vfield(self):
        '''
        Comoving volume of a single field
        '''
        return self.Sfield*self.Lfield
    
    
    @cached_obs_property            
    def Vvox(self):
        '''
        Comoving volume of a single voxel
        '''
        return self.Vfield/self.Nvox
        
    
    ##########################
    # Instrument noise power #
    ##########################
            
    @cached_obs_property
    def tpix(self):
        '''
        Time spent observing each pixel with a single detector
        '''
        return self.tobs/self.Npix
    
    
    @cached_obs_property
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
    
    
    @cached_obs_property    
    def Pnoise(self):
        '''
        Noise power spectrum amplitude
        '''
        if self.do_Jysr:
            return (self.sigma_N**2*self.Vvox/(self.tpix*self.Nfeeds)).to(u.Mpc**3*u.Jy**2/u.sr**2)
        else:
            return self.sigma_N**2*self.Vvox
            
        
    @cached_obs_property
    def sigma_par(self):
        '''
        High-resolution cutoff for line-of-sight modes
        '''
        return (cu.c*self.dnu*(1+self.z)/(self.H*self.nuObs)).to(u.Mpc)
    
    
    @cached_obs_property
    def sigma_perp(self):
        '''
        High-resolution cutoff for transverse modes
        '''
        return (self.r0*(self.beam_width/(1*u.rad))).to(u.Mpc)
                
                
    @cached_obs_property
    def kmax_los(self):
        '''
        Maximum k in line of sight direction
        '''
        return 2.*np.pi/self.sigma_par
    
    
    @cached_obs_property
    def kmax_sky(self):
        '''
        Maximum k in the transverse direction
        '''
        return 2.*np.pi/self.sigma_perp
        
        
    @cached_obs_property
    def kmin_los(self):
        '''
        Minimum k in the line of sight direction
        '''
        return 2*np.pi/self.Lfield
        
        
    @cached_obs_property
    def kmin_sky(self):
        '''
        Minimum k in the transverse direction
        '''
        return 2*np.pi/self.Sfield**0.5
    
    
    @cached_obs_property
    def kmin_field(self):
        '''
        Minimum k accessible in a single field, set by the maximum side length
        '''
        return min([self.kmin_los,self.kmin_sky])
        
        
    @cached_obs_property
    def kmax_field(self):
        '''
        Maximum k accesible in a given survey, set by the best resolution
        '''
        return max([self.kmax_los,self.kmax_sky])
        
        
    @cached_obs_property
    def Wkmax_par(self):
        '''
        Resolution cutoff in power spectrum in the los direction
        '''
        exparg = -((self.k_par*self.sigma_par)**2).decompose()
        return np.exp(exparg)
        
        
    @cached_obs_property
    def Wkmax_perp(self):
        '''
        Resolution cutoff in power spectrum in the transverse direction
        '''
        exparg = -((self.k_perp*self.sigma_perp)**2).decompose()
        return np.exp(exparg)
        
        
    @cached_obs_property
    def Wkmax(self):
        '''
        Resolution cutoff in power spectrum
        '''
        return self.Wkmax_par*self.Wkmax_perp
        
    @cached_obs_property
    def Wkmin_par(self):
        '''
        Precision cutoff in power spectrum due to volume observed in los direction
        '''
        exparg = -((self.k_par/(self.N_FG_par*self.kmin_los))**2).decompose()
        return 1.-np.exp(exparg)
        
        
    @cached_obs_property
    def Wkmin_perp(self):
        '''
        Precision cutoff in power spectrum due to volume observed in transverse direction
        '''
        exparg = -((self.k_perp/(self.N_FG_perp*self.kmin_sky))**2).decompose()
        return 1.-np.exp(exparg)
        
        
    @cached_obs_property
    def Wkmin(self):
        '''
        Precision cutoff in power spectrum due to volume observed
        '''
        return self.Wkmin_par*self.Wkmin_perp
        
    
    @cached_obs_property
    def Wk_FGwedge(self):
        '''
        Applies foreground wedge removal
        '''
        W = np.ones(self.ki_grid.shape)
        if self.do_FG_wedge:
            #k_par_min = a + b*k_perp
            kpar_min_wedge = self.a_FG.to(self.k.unit) + self.b_FG*np.abs(self.k_perp)
            ind = np.where(np.abs(self.k_par)<kpar_min_wedge)
            W[ind] = 0
            return W
        else:
            return W
        
        
    @cached_obs_property
    def Wk(self):
        '''
        Resolution cutoff in power spectrum
        '''
        return self.Wkmin*self.Wkmax


    @cached_obs_property
    def Nmodes(self):
        '''
        Number of modes between k and k+dk.        
        Multiply by dmu/2 to get the number of modes between k and k+dk and mu and mu+dmu
        '''
        return self.ki_grid**2*self.dk*self.Vfield*self.Nfield/4./np.pi**2.
        
        
    @cached_obs_property
    def sk_CV(self):
        '''
        Error at k and mu due to sample variance
        '''
        return self.Pk/np.sqrt(self.Nmodes*self.dmu[0])
        
        
    @cached_obs_property
    def covmat_CV_00(self):
        '''
        00 term of the covariance matrix from CV
        '''
        return 0.5*np.trapz(self.Pk**2/self.Nmodes,self.mu,axis=0)
        
        
    @cached_obs_property
    def covmat_CV_02(self):
        '''
        02 term of the covariance matrix from CV
        (equal to the 20)
        '''
        L2 = legendre(2)(self.mui_grid)
        return 5./2.*np.trapz(self.Pk**2*L2/self.Nmodes,self.mu,axis=0)
        
        
    @cached_obs_property
    def covmat_CV_04(self):
        '''
        04 term of the covariance matrix from CV
        (equal to the 40)
        '''
        L4 = legendre(4)(self.mui_grid)
        return 9./2.*np.trapz(self.Pk**2*L4/self.Nmodes,self.mu,axis=0)
        
        
    @cached_obs_property
    def covmat_CV_22(self):
        '''
        22 term of the covariance matrix from CV
        '''
        L2 = legendre(2)(self.mui_grid)
        return 25./2.*np.trapz(self.Pk**2*L2*L2/self.Nmodes,self.mu,axis=0)
        
        
    @cached_obs_property
    def covmat_CV_24(self):
        '''
        24 term of the covariance matrix from CV
        (equal to the 42)
        '''
        L2 = legendre(2)(self.mui_grid)
        L4 = legendre(4)(self.mui_grid)
        return 45./2.*np.trapz(self.Pk**2*L2*L4/self.Nmodes,self.mu,axis=0)
        
        
    @cached_obs_property
    def covmat_CV_44(self):
        '''
        44 term of the covariance matrix from CV
        '''
        L4 = legendre(4)(self.mui_grid)
        return 81./2.*np.trapz(self.Pk**2*L4*L4/self.Nmodes,self.mu,axis=0)
        
        
    def covmat_CV_l1l2(self,l1,l2):
        '''
        l1l2 term of the covariance matrix from CV
        '''
        if l1 == 0 and l2 == 0:
            return self.covmat_CV_00
        elif l1 == 0 and l2 == 2:
            return self.covmat_CV_02
        elif l1 == 0 and l2 == 4:
            return self.covmat_CV_04
        elif l1 == 2 and l2 == 2:
            return self.covmat_CV_22
        elif l1 == 2 and l2 == 4:
            return self.covmat_CV_24
        elif l1 == 4 and l2 == 4:
            return self.covmat_CV_44
        else:
            Ll1 = legendre(l1)(self.mui_grid)
            Ll2 = legendre(l2)(self.mui_grid)
            return (2.*l1+1.)*(2.*l2+1.)/2.*np.trapz(self.Pk**2*L1*L2/self.Nmodes,self.mu,axis=0)
        
        
    @cached_obs_property
    def sk_N(self):
        '''
        Error at k and mu due to instrumental noise
        '''
        return self.Pnoise/(np.sqrt(self.Nmodes*self.dmu[0]/2.))
            
            
    @cached_obs_property
    def covmat_N_00(self):
        '''
        00 term of the covariance matrix from instrumental noise
        '''
        return 1./2.*np.trapz(self.Pnoise**2./(self.Nmodes),self.mu,axis=0)
        
        
    @cached_obs_property
    def covmat_N_02(self):
        '''
        02 term of the covariance matrix from instrumental noise
        (equal to the 02)        
        '''
        L2 = legendre(2)(self.mui_grid)
        return 5./2.*np.trapz(self.Pnoise**2.*L2/(self.Nmodes),self.mu,axis=0)


    @cached_obs_property
    def covmat_N_04(self):
        '''
        04 term of the covariance matrix from instrumental noise
        (equal to the 04)        
        '''
        L4 = legendre(4)(self.mui_grid)
        return 9./2.*np.trapz(self.Pnoise**2.*L4/(self.Nmodes),self.mu,axis=0)
        
        
    @cached_obs_property
    def covmat_N_22(self):
        '''
        22 term of the covariance matrix from instrumental noise
        '''
        L2 = legendre(2)(self.mui_grid)
        return 25./2.*np.trapz(self.Pnoise**2.*L2*L2/(self.Nmodes),self.mu,axis=0)


    @cached_obs_property
    def covmat_N_24(self):
        '''
        24 term of the covariance matrix from instrumental noise
        (equal to the 42)
        '''
        L2 = legendre(2)(self.mui_grid)
        L4 = legendre(4)(self.mui_grid)
        return 45./2.*np.trapz(self.Pnoise**2.*L2*L4/(self.Nmodes),self.mu,axis=0)
        
        
    @cached_obs_property
    def covmat_N_44(self):
        '''
        44 term of the covariance matrix from instrumental noise
        '''
        L4 = legendre(4)(self.mui_grid)
        return 81./2.*np.trapz(self.Pnoise**2.*L4*L4/(self.Nmodes),self.mu,axis=0)
        
        
    def covmat_N_l1l2(self,l1,l2):
        '''
        l1l2 term of the covariance matrix from N
        '''
        if l1 == 0 and l2 == 0:
            return self.covmat_N_00
        elif l1 == 0 and l2 == 2:
            return self.covmat_N_02
        elif l1 == 0 and l2 == 4:
            return self.covmat_N_04
        elif l1 == 2 and l2 == 2:
            return self.covmat_N_22
        elif l1 == 2 and l2 == 4:
            return self.covmat_N_24
        elif l1 == 4 and l2 == 4:
            return self.covmat_N_44
        else:
            Ll1 = legendre(l1)(self.mui_grid)
            Ll2 = legendre(l2)(self.mui_grid)
            return (2.*l1+1.)*(2.*l2+1.)*np.trapz(self.Pnoise**2.*l1l2/(self.Nmodes),self.mu,axis=0)
        
        
    @cached_obs_property
    def sk(self):
        '''
        Total error at k and mu
        '''
        return self.sk_CV+self.sk_N
        
        
    @cached_obs_property
    def covmat_00(self):
        '''
        00 term of the total covariance matrix
        '''
        integrand = (self.Pk+self.Pnoise)/self.Nmodes**0.5
        return 0.5*np.trapz(integrand**2,self.mu,axis=0)
        
        
    @cached_obs_property
    def covmat_02(self):
        '''
        02 term of the total covariance matrix
        '''
        L2 = legendre(2)(self.mui_grid)
        integrand = (self.Pk+self.Pnoise)/self.Nmodes**0.5
        return 5./2.*np.trapz(integrand**2*L2,self.mu,axis=0)
        
        
    @cached_obs_property
    def covmat_04(self):
        '''
        04 term of the total covariance matrix
        '''
        L4 = legendre(4)(self.mui_grid)
        integrand = (self.Pk+self.Pnoise)/self.Nmodes**0.5
        return 9./2.*np.trapz(integrand**2*L4,self.mu,axis=0)
        
        
    @cached_obs_property
    def covmat_22(self):
        '''
        22 term of the total covariance matrix
        '''
        L2 = legendre(2)(self.mui_grid)
        integrand = (self.Pk+self.Pnoise)/self.Nmodes**0.5
        return 25./2.*np.trapz(integrand**2*L2*L2,self.mu,axis=0)
        
        
    @cached_obs_property
    def covmat_24(self):
        '''
        24 term of the total covariance matrix
        '''
        L2 = legendre(2)(self.mui_grid)
        L4 = legendre(4)(self.mui_grid)
        integrand = (self.Pk+self.Pnoise)/self.Nmodes**0.5
        return 45./2.*np.trapz(integrand**2*L2*L4,self.mu,axis=0)
        
        
    @cached_obs_property
    def covmat_44(self):
        '''
        44 term of the total covariance matrix
        '''
        L4 = legendre(4)(self.mui_grid)
        integrand = (self.Pk+self.Pnoise)/self.Nmodes**0.5
        return 81./2.*np.trapz(integrand**2*L4*L4,self.mu,axis=0)


    def covmat_l1l2(self,l1,l2):
        '''
        l1l2 term of the total covariance matrix
        '''
        if l1 == 0 and l2 == 0:
            return self.covmat_00
        elif l1 == 0 and l2 == 2:
            return self.covmat_02
        elif l1 == 0 and l2 == 4:
            return self.covmat_04
        elif l1 == 2 and l2 == 2:
            return self.covmat_22
        elif l1 == 2 and l2 == 4:
            return self.covmat_24
        elif l1 == 4 and l2 == 4:
            return self.covmat_44
        else:
            l1 = legendre(l1)(self.mui_grid)
            l2 = legendre(l2)(self.mui_grid)
            integrand = (self.Pk+self.Pnoise)/self.Nmodes**0.5
            return (2.*l1+1.)*(2.*l2+1.)*np.trapz(integrand**2*l1*l2,self.mu,axis=0)
        
        
    @cached_obs_property
    def nk_field(self):
        '''
        Number of k bins for a given survey, based on kmax, kmin and 
        delta_k (=kmin)
        
        Only works if k_kind=linear
        '''
        if not self.k_kind == 'linear':
            raise ValueError('nk_field can only be computed for linear spacing')
        kmax = self.kmax_field
        delta_k = self.kmin_field
        
        return (kmax-delta_k)/delta_k
        
        
    @cached_obs_property
    def SNR(self):
        '''
        Signal to noise ratio for given model and experiment
        '''
        SNR_k = (self.Pk**2/self.sk**2).decompose()
        ind = np.logical_and(self.k>=self.kmin_field,self.k<=self.kmax_field)
        return np.sqrt(SNR_k[ind].sum())
        
        
    @cached_obs_property
    def SNR_0(self):
        '''
        Signal to noise ratio in the monopole for given model and experiment
        '''
        SNR_k = (self.Pk_0**2/self.covmat_00).decompose()
        ind = np.logical_and(self.k>=self.kmin_field,self.k<=self.kmax_field)
        return np.sqrt(SNR_k[ind].sum())
        
        
    @cached_obs_property
    def SNR_2(self):
        '''
        Signal to noise ratio in the quadrupole for given model and experiment
        '''
        SNR_k = (self.Pk_2**2/self.covmat_22).decompose()
        ind = np.logical_and(self.k>=self.kmin_field,self.k<=self.kmax_field)
        return np.sqrt(SNR_k[ind].sum())
        
        
    @cached_obs_property
    def SNR_4(self):
        '''
        Signal to noise ratio in the hexadecapole for given model and experiment
        '''
        SNR_k = (self.Pk_4**2/self.covmat_44).decompose()
        ind = np.logical_and(self.k>=self.kmin_field,self.k<=self.kmax_field)
        return np.sqrt(SNR_k[ind].sum())
        
        
    @cached_obs_property
    def SNR_multipoles(self):
        '''
        Signal to noise ratio in the monopole, quadrupole and hexadecapole
        for given model and experiment
        '''
        ind = np.where(np.logical_and(self.k>=self.kmin_field,
                                      self.k<=self.kmax_field))[0]
        Nkseen = len(ind)
        Pkvec = np.zeros(Nkseen*3)
        covmat = np.zeros((Nkseen*3,Nkseen*3))
        
        Pkvec[:Nkseen] = self.Pk_0[ind]
        Pkvec[Nkseen:Nkseen*2] = self.Pk_2[ind]
        Pkvec[Nkseen*2:Nkseen*3] = self.Pk_4[ind]
        
        covmat[:Nkseen,:Nkseen] = np.diag(self.covmat_00[ind])
        covmat[:Nkseen,Nkseen:Nkseen*2] = np.diag(self.covmat_02[ind])
        covmat[:Nkseen,Nkseen*2:Nkseen*3] = np.diag(self.covmat_04[ind])
        covmat[Nkseen:Nkseen*2,:Nkseen] = np.diag(self.covmat_02[ind])
        covmat[Nkseen:Nkseen*2,Nkseen:Nkseen*2] = np.diag(self.covmat_22[ind])
        covmat[Nkseen:Nkseen*2,Nkseen*2:Nkseen*3] = np.diag(self.covmat_24[ind])
        covmat[Nkseen*2:Nkseen*3,:Nkseen] = np.diag(self.covmat_04[ind])
        covmat[Nkseen*2:Nkseen*3,Nkseen:Nkseen*2] = np.diag(self.covmat_24[ind])
        covmat[Nkseen*2:Nkseen*3,Nkseen*2:Nkseen*3] = np.diag(self.covmat_44[ind])
        
        return np.sqrt(np.dot(Pkvec,np.dot(np.linalg.inv(covmat),Pkvec)))
        
        
    def get_covmat(self,Nmul):
        '''
        Get the covariance matrix for a given number of multipoles 
        (starting always from the monopole and without skipping any)
        '''
        if Nmul > 3:
            raise ValueError('Not implemented yet!\
            Implement covmat_66 and expand this function')
            
        covmat = np.zeros((self.nk*Nmul,self.nk*Nmul))
        covmat[:self.nk,:self.nk] = np.diag(self.covmat_00)
        
        if Nmul > 1:
            covmat[:self.nk,self.nk:self.nk*2] = np.diag(self.covmat_02)
            covmat[self.nk:self.nk*2,:self.nk] = np.diag(self.covmat_02)
            covmat[self.nk:self.nk*2,self.nk:self.nk*2] = np.diag(self.covmat_22)
            covmat[:self.nk,self.nk:self.nk*2] = np.diag(self.covmat_02)
        if Nmul > 2:
            covmat[:self.nk,self.nk*2:self.nk*3] = np.diag(self.covmat_04)
            covmat[self.nk:self.nk*2,self.nk*2:self.nk*3] = np.diag(self.covmat_24)
            covmat[self.nk*2:self.nk*3,:self.nk] = np.diag(self.covmat_04)
            covmat[self.nk*2:self.nk*3,self.nk:self.nk*2] = np.diag(self.covmat_24)
            covmat[self.nk*2:self.nk*3,self.nk*2:self.nk*3] = np.diag(self.covmat_44)

        return covmat
        
        
    @cached_vid_property
    def PDFnoise(self):
        '''
        PDF of the noise, to include noise in the total signal of the VID
        We multiply the distribution by 2 in order to have a normalized PDF
        (this is only half of a Gaussian)
        '''
        if self.do_Jysr:
            sigN2 = self.Pnoise/self.Vvox
            exparg = -0.5*self.T**2/sigN2
            norm = (2.*np.pi*sigN2)**0.5
        else:
            exparg = -0.5*(self.T/self.sigma_N)**2.
            norm = (2.*np.pi)**0.5*self.sigma_N
        return (2.*np.exp(exparg)/norm).to((self.T**-1).unit)
        
        
############
# Doctests #
############

if __name__ == "__main__":
    import doctest

    doctest.testmod(optionflags=doctest.ELLIPSIS |
                    doctest.NORMALIZE_WHITESPACE)
