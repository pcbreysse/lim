'''
Module for computing statistics of cross-correlations between intensity maps
of different lines (CrossLine class) or between an intensity map and a galaxy
survey (CrossGal class, TODO)
'''


import numpy as np
import astropy.units as u
import astropy.constants as cu
from scipy.interpolate import interp1d
from scipy.special import legendre
from lim1 import lim

from source.line_model import LineModel
from source.tools._utils import cached_property,get_default_params
from source.tools._utils import ulogspace, ulinspace,check_params,log_interp1d

class CrossLine(object):
    '''
    Cross-correlation between two lines
    '''
    def __init__(self,
                 # Parameters of line 1
                 model1_type = 'ML',
                 model1_name = 'MassPow',
                 model1_par = dict(A=2e-6,b=1),
                 sigma_scatter1 = 0.,
                 nu1 = 115*u.GHz,
                 nuObs1 = 30*u.GHz,
                 Tsys_NEFD1 = 40*u.K,
                 Nfeeds1 = 19,
                 beam_FWHM1 = 4.1*u.arcmin,
                 Delta_nu1 = 8*u.GHz,
                 dnu1 = 15.6*u.MHz,
                 tobs1 = 6000*u.hr,
                 Omega_field1 = 2.25*u.deg**2,
                 beam_efficiency1 = 1.,
                 do_Jysr1 = False,
                 interloper_params_1 = None,
                 # Parameters of line 2
                 model2_type = 'ML',
                 model2_name = 'MassPow',
                 model2_par = dict(A=2e-6,b=1),
                 sigma_scatter2 = 0.,
                 nu2 = 230*u.GHz,
                 nuObs2 = 60*u.GHz,
                 Tsys_NEFD2 = 40*u.K,
                 Nfeeds2 = 19,
                 beam_FWHM2 = 4.1*u.arcmin,
                 Delta_nu2 = 8*u.GHz,
                 dnu2 = 15.6*u.MHz,
                 tobs2 = 6000*u.hr,
                 Omega_field2 = 2.25*u.deg**2,
                 beam_efficiency2 = 1.,
                 do_Jysr2 = False,
                 interloper_params_2 = None,
                 # Parameters common to both lines
                 corr12 = 1., # Correlation coefficient between the two scatters
                 cosmo_code = 'camb',
                 cosmo_input_camb=dict(f_NL=0,H0=67.36,cosmomc_theta=None,ombh2=0.02237, omch2=0.12, 
                               omk=0.0, neutrino_hierarchy='degenerate', 
                               num_massive_neutrinos=3, mnu=0.06, nnu=3.046, 
                               YHe=None, meffsterile=0.0, standard_neutrino_neff=3.046, 
                               TCMB=2.7255, tau=None, deltazrei=None, bbn_predictor=None, 
                               theta_H0_range=[10, 100],w=-1.0, wa=0., cs2=1.0, 
                               dark_energy_model='ppf',As=2.1e-09, ns=0.9649, nrun=0, 
                               nrunrun=0.0, r=0.0, nt=None, ntrun=0.0, 
                               pivot_scalar=0.05, pivot_tensor=0.05,
                               parameterization=2,halofit_version='mead'),
                 cosmo_input_class=dict(f_NL=0,H0=67.36,omega_b=0.02237, omega_cdm=0.12, 
                               A_s=2.1e-9,n_s=0.9649,
                               N_ncdm=3, m_ncdm='0.02,0.02,0.02', N_ur = 0.00641,
                               output='mPk,mTk'),
                 Px_shot_manual = None, # Cross-shot if specified manually
                 hmf_model = 'ST',
                 bias_model = 'ST99',
                 bias_par = {},
                 Mmin = 1e9*u.Msun,
                 Mmax = 1e15*u.Msun,
                 nM = 5000,
                 Lmin = 10*u.Lsun,
                 Lmax = 1e8*u.Lsun,
                 nL = 5000,
                 kmin = 1e-2/u.Mpc,
                 kmax = 10/u.Mpc,
                 nk = 100,
                 k_kind = 'log',
                 Nfield = 1,
                 fduty = 1.,
                 do_onehalo = False,
                 do_RSD = True,
                 sigma_NL = 7*u.Mpc,
                 nmu = 1000,
                 FoG_damp = 'Lorentzian',
                 smooth = False,
                 nonlinear = False,
                 Nmul = 3):
        
        # Get list of input values to check type and units
        self._x_params = locals()
        self._x_params.pop('self')
        
        # Get list of input names and default values
        self._default_x_params = get_default_params(CrossLine.__init__)
        # Check that input values have the correct type and units
        check_params(self._x_params,self._default_x_params)
        
        # Set all given parameters
        [setattr(self,key,self._x_params[key]) for key in self._x_params]
        
        # Cross-correlation requires mass-luminosity model
        if self.model1_type=='LF' or self.model2_type=='LF':
            raise ValueError("model_type must be 'ML' or 'TOY'.")
        #self.model1_type = 'ML'
        #self.model2_type = 'ML'
        
        # TODO: Add one-halo cross-correlation behavior
        if self.do_onehalo:
            print('One-halo cross-correlations not currently available')
            self.do_onehalo = False
        
        # Create overall lists of parameters (Only used if using one of 
        # CrossLine's subclasses
        self._input_params = {} # Don't want .update to change _x_params
        self._default_params = {}
        self._input_params.update(self._x_params)
        self._default_params.update(self._default_x_params)
        
        # Create list of cached properties
        self._update_list = []
    
    ###########################################
    # Create lim objects for autocorrelations #
    ###########################################
    
    @cached_property
    def par1(self):
        '''
        Parameter dictionary for line 1
        '''
        return dict(cosmo_code=self.cosmo_code,
                    cosmo_input_camb=self.cosmo_input_camb,
                    cosmo_input_class=self.cosmo_input_class,
                    model_type=self.model1_type,
                    model_name=self.model1_name,
                    model_par=self.model1_par,
                    nu=self.nu1,
                    nuObs=self.nuObs1,
                    Tsys_NEFD=self.Tsys_NEFD1,
                    Nfeeds=self.Nfeeds1,
                    beam_FWHM=self.beam_FWHM1,
                    Delta_nu=self.Delta_nu1,
                    dnu=self.dnu1,
                    tobs=self.tobs1,
                    Omega_field=self.Omega_field1,
                    beam_efficiency = self.beam_efficiency1,
                    do_Jysr=self.do_Jysr1,
                    hmf_model=self.hmf_model,
                    bias_model=self.bias_model,
                    bias_par=self.bias_par,
                    Mmin=self.Mmin,
                    Mmax=self.Mmax,
                    nM=self.nM,
                    Lmin=self.Lmin,
                    Lmax=self.Lmax,
                    nL=self.nL,
                    kmin=self.kmin,
                    kmax=self.kmax,
                    nk=self.nk,
                    k_kind=self.k_kind,
                    sigma_scatter=self.sigma_scatter1,
                    fduty=self.fduty,
                    do_onehalo=self.do_onehalo,
                    do_RSD=self.do_RSD,
                    sigma_NL=self.sigma_NL,
                    nmu=self.nmu,
                    FoG_damp=self.FoG_damp,
                    smooth=self.smooth,
                    nonlinear=self.nonlinear,
                    Nmul=self.Nmul,
                    interloper_params=self.interloper_params_1,
                    Nfield = self.Nfield)                 

    @cached_property
    def par2(self):
        '''
        Parameter dictionary for line 2
        '''
        return dict(cosmo_code=self.cosmo_code,
                    cosmo_input_camb=self.cosmo_input_camb,
                    cosmo_input_class=self.cosmo_input_class,
                    model_type=self.model2_type,
                    model_name=self.model2_name,
                    model_par=self.model2_par,
                    nu=self.nu2,
                    nuObs=self.nuObs2,
                    Tsys_NEFD=self.Tsys_NEFD2,
                    Nfeeds=self.Nfeeds2,
                    beam_FWHM=self.beam_FWHM2,
                    Delta_nu=self.Delta_nu2,
                    dnu=self.dnu2,
                    tobs=self.tobs2,
                    Omega_field=self.Omega_field2,
                    beam_efficiency = self.beam_efficiency2,
                    do_Jysr=self.do_Jysr2,
                    hmf_model=self.hmf_model,
                    bias_model=self.bias_model,
                    bias_par=self.bias_par,
                    Mmin=self.Mmin,
                    Mmax=self.Mmax,
                    nM=self.nM,
                    Lmin=self.Lmin,
                    Lmax=self.Lmax,
                    nL=self.nL,
                    kmin=self.kmin,
                    kmax=self.kmax,
                    nk=self.nk,
                    k_kind=self.k_kind,
                    sigma_scatter=self.sigma_scatter2,
                    fduty=self.fduty,
                    do_onehalo=self.do_onehalo,
                    do_RSD=self.do_RSD,
                    sigma_NL=self.sigma_NL,
                    nmu=self.nmu,
                    FoG_damp=self.FoG_damp,
                    smooth=self.smooth,
                    nonlinear=self.nonlinear,
                    Nmul=self.Nmul,
                    interloper_params=self.interloper_params_2,
                    Nfield = self.Nfield)
    
    @cached_property
    def m1(self):
        '''
        lim object for Line 1 autocorrelation
        '''
        return lim(self.par1)
    
    @cached_property
    def m2(self):
        '''
        lim object for line 2 autocorrelation
        '''
        return lim(self.par2)
    
    ###########################################
    # Compute overlapping area/redshift range #
    ###########################################
    
    @cached_property
    def z_min(self):
        '''
        Minimum redshift of overlapping surveys
        '''
        return np.array([self.m1.z_min,self.m2.z_min]).max()
    
    @cached_property
    def z_max(self):
        '''
        Minimum redshift of overlapping surveys
        '''
        return np.array([self.m1.z_max,self.m2.z_max]).min()
        
    @cached_property
    def Omega_field(self):
        '''
        Solid angle subtended by overlapping surveys, assumes all of the
        smaller field is contained within the larger
        '''
        return min([self.m1.Omega_field,self.m2.Omega_field])
    
    def check_overlap(self):
        '''
        Checks if redshift ranges overlap at all, raises error if surveys do
        not have any redshifts in common
        '''
        if self.z_max<self.z_min:
            raise ValueError('No overlapping redshifts')
    
    @cached_property
    def m1x(self):
        '''
        m1 object centered at overlap redshift
        '''
        self.check_overlap()
        par = self.par1
        #par['nuObs'] = self.m1.nu/(1.+self.z)
        nuMin = self.m1.nu/(1.+self.z_max)
        nuMax = self.m1.nu/(1.+self.z_min)
        par['nuObs'] = (nuMin+nuMax)/2.
        par['Delta_nu'] = nuMax-nuMin
        par['Omega_field'] = self.Omega_field
        par['tobs'] = self.m1.tobs*self.Omega_field/self.m1.Omega_field
        return lim(par)
        
    @cached_property
    def m2x(self):
        '''
        m2 object centered at overlap redshift
        '''
        self.check_overlap()
        par = self.par2
        #par['nuObs'] = self.m2.nu/(1.+self.z)
        nuMin = self.m2.nu/(1.+self.z_max)
        nuMax = self.m2.nu/(1.+self.z_min)
        par['nuObs'] = (nuMin+nuMax)/2.
        par['Delta_nu'] = nuMax-nuMin
        par['Omega_field'] = self.Omega_field
        par['tobs'] = self.m2.tobs*self.Omega_field/self.m2.Omega_field
        return lim(par)
    
    @cached_property
    def z(self):
        if self.m1x.z != self.m2x.z:
            raise ValueError("Something went wrong with mx's")
        return self.m1x.z
    
    #######################################
    # Power spectra in overlapping region #
    #######################################
    
    @cached_property
    def k(self):
        '''
        Power spectrum magnitude bin centers
        '''
        return self.m1x.k
    
    @cached_property
    def mu(self):
        '''
        Power spectrum direction bin centers
        '''
        return self.m1x.mu
    
    @cached_property
    def ki_grid(self):
        '''
        Grid of k bins for anisotropic calculations
        '''
        return self.m1x.ki_grid
    
    @cached_property
    def mui_grid(self):
        '''
        Grid of mu bins for anisotropic calculations
        '''
        return self.m1x.mui_grid
    
    @cached_property
    def Pm(self):
        '''
        Matter power spectrum in overlapping region
        '''
        return self.m1x.Pm
    
    @cached_property
    def Px_clust(self):
        '''
        Clustering component of the cross-spectrum
        '''
        wt1 = self.m1x.Tmean*self.m1x.bavg*np.sqrt(self.m1x.RSD)
        wt2 = self.m2x.Tmean*self.m2x.bavg*np.sqrt(self.m2x.RSD)
        return wt1*wt2*self.Pm
    
    @cached_property
    def Px_shot(self):
        '''
        Shot noise in cross-spectrum
        '''
        if self.Px_shot_manual is not None:
            print('Using manually input cross-shot power')
            try:
                p = self.Px_shot_manual.to(self.Px_clust.unit)
            except u.UnitConversionError:
                raise u.UnitConversionError(
                       'Manually specified cross-shot has incorrect units')
            return p
        else:
            exparg = self.corr12*self.sigma_scatter1*self.sigma_scatter2*np.log(10)**2
            itgrnd = self.m1x.dndM*self.m1x.LofM*self.m2x.LofM*np.exp(exparg)
            itgrl = np.trapz(itgrnd,self.m1x.M)
            return self.m1x.CLT*self.m2x.CLT*itgrl*self.fduty
    
    @cached_property
    def Px(self):
        '''
        Total 2D cross-spectrum
        '''
        P = self.Px_clust+self.Px_shot
        if self.smooth:
            return np.sqrt(self.m1x.Wk*self.m2x.Wk)*P
        else:
            return P
    
    def get_Px_l(self,l):
        '''
        Get multipole l of cross-spectrum
        '''
        Ll = legendre(l)(self.mui_grid)
        nrm = 0.5*(2.*l+1)
        return nrm*np.trapz(self.Px*Ll,self.mu,axis=0)
    
    @cached_property
    def Px_0(self):
        '''
        Monopole of cross-power spectrum
        '''
        return self.get_Px_l(0)
    
    @cached_property
    def Px_2(self):
        '''
        Quadrupole of cross-spectrum
        '''
        return self.get_Px_l(2)
    
    @cached_property
    def Px_4(self):
        '''
        Hexadecapole of cross-spectrum
        '''
        return self.get_Px_l(4)
    
    
    
    ##########################################
    # Error and covariance on cross-spectrum #
    ##########################################
    
    @cached_property
    def sx_N(self):
        '''
        Error on cross-spectrum at k and mu due to noise alone
        '''
        Nm = self.m1x.Nmodes
        return np.sqrt(self.m1x.Pnoise*self.m2x.Pnoise)/np.sqrt(2.*Nm*self.m1x.dmu[0])
    
    def get_covmat_x_l1l2_N(self,l1,l2):
        '''
        l1l2 term of the covariance matrix of the cross-spectrum due to noise
        alone
        '''
        itgrnd = self.sx_N**2*self.m1x.dmu[0]
        Ll1 = legendre(l1)(self.mui_grid)
        Ll2 = legendre(l2)(self.mui_grid)
        return 0.5*(2.*l1+1.)*(2.*l2+1.)*np.trapz(itgrnd*Ll1*Ll2,
                    self.mu,axis=0)
    
    @cached_property
    def covmat_x_N_00(self):
        '''
        00 term of the covariance matrix of the cross-spectrum
        '''
        return self.get_covmat_x_l1l2_N(0,0)
    
    @cached_property
    def covmat_x_N_22(self):
        '''
        22 term of the covariance matrix of the cross-spectrum
        '''
        return self.get_covmat_x_l1l2_N(2,2)
    
    @cached_property
    def covmat_x_N_44(self):
        '''
        44 term of the covariance matrix of the cross-spectrum
        '''
        return self.get_covmat_x_l1l2_N(4,4)
    
    @cached_property
    def sx(self):
        '''
        Error on cross-spectrum at k and mu
        '''
        s1 = self.m1x.Pk+self.m1x.Pnoise
        s2 = self.m1x.Pk+self.m2x.Pnoise
        Nm = self.m1x.Nmodes
        return np.sqrt(self.Px**2+s1*s2)/np.sqrt(2.*Nm*self.m1x.dmu[0])
    
    def get_covmat_x_l1l2(self,l1,l2):
        '''
        l1l2 term of the covariance matrix of the cross-spectrum
        '''
        itgrnd = self.sx**2*self.m1x.dmu[0]
        Ll1 = legendre(l1)(self.mui_grid)
        Ll2 = legendre(l2)(self.mui_grid)
        return 0.5*(2.*l1+1.)*(2.*l2+1.)*np.trapz(itgrnd*Ll1*Ll2,
                    self.mu,axis=0)
    
    @cached_property
    def covmat_x_00(self):
        '''
        00 term of the covariance matrix of the cross-spectrum
        '''
        return self.get_covmat_x_l1l2(0,0)
    
    @cached_property
    def covmat_x_22(self):
        '''
        22 term of the covariance matrix of the cross-spectrum
        '''
        return self.get_covmat_x_l1l2(2,2)
    
    @cached_property
    def covmat_x_44(self):
        '''
        44 term of the covariance matrix of the cross-spectrum
        '''
        return self.get_covmat_x_l1l2(4,4)
        
    @cached_property
    def SNR_x_0(self):
        '''
        Signal-to-noise of the monopole of the cross-spectrum
        '''
        if not self.smooth:
            print('Warning: Set smooth=True to take into account\
                   field size and beam smoothing')
        SNR_k = (self.Px_0**2/self.covmat_x_00).decompose()
        return np.sqrt(SNR_k.sum())
    
    @cached_property
    def SNR_x_2(self):
        '''
        Signal-to-noise of the quadrupole of the cross-spectrum
        '''
        if not self.smooth:
            print('Warning: Set smooth=True to take into account\
                   field size and beam smoothing')
        SNR_k = (self.Px_2**2/self.covmat_x_22).decompose()
        return np.sqrt(SNR_k.sum())
    
    @cached_property
    def SNR_x_4(self):
        '''
        Signal-to-noise of the hexadecapole of the cross-spectrum
        '''
        if not self.smooth:
            print('Warning: Set smooth=True to take into account\
                   field size and beam smoothing')
        SNR_k = (self.Px_4**2/self.covmat_x_44).decompose()
        return np.sqrt(SNR_k.sum())        
    
    def get_covmat_x(self,Nmul):
        '''
        Build covariance matrix of first Nmul cross-spectrum multipoles
        '''
        covmat = np.zeros((self.nk*Nmul,self.nk*Nmul))#*self.Px.unit**2
        nk = self.nk
        for ii in range(0,Nmul):
            for jj in range(0,Nmul):
                l1 = 2*ii
                l2 = 2*jj
                cmi = self.get_covmat_x_l1l2(l1,l2)
                covmat[ii*nk:(ii+1)*nk,jj*nk:(jj+1)*nk] = np.diag(cmi)
        return covmat
    
    def get_Px_combined(self,Nmul):
        '''
        Get the power spectra for a given number of multipoles 
        of the cross-spectrum (starting always from the monopole
        and without skipping any)
        '''
        P = np.zeros(self.nk*Nmul)*self.Px.unit
        nk = self.nk
        for ii in range(0,Nmul):
            l = 2*ii
            P[ii*nk:(ii+1)*nk] = self.get_Px_l(l)
        return P
    
    @cached_property
    def covmat_x(self):
        '''
        Full covariance matrix of cross-spectrum multipoles
        '''
        return self.get_covmat_x(self.Nmul)
    
    @cached_property
    def Px_combined(self):
        '''
        Combined cross-spectrum multipoles
        '''
        return self.get_Px_combined(self.Nmul)
        
    @cached_property
    def SNR_x_multipoles(self):
        '''
        Signal-to-noise ratio for the combination of the monopole,
        quadrupole, and hexadecapole of the cross-spectrum.
        '''
        # Avoid inverting covariance matrix for stability
        # TODO: Apply this fix elsewhere?
        cm_dot_p = np.linalg.solve(self.covmat_x,self.Px_combined)
        return np.sqrt(np.dot(self.Px_combined,cm_dot_p))
    
    #####################################
    # Combining cross- and auto-spectra #
    #####################################
    
    def get_covmat_12_l1l2(self,l1,l2):
        '''
        Covariance between auto-spectra of lines 1 and 2 for multipoles
        l1 and l2
        '''
        itgrnd = self.Px**2/self.m1x.Nmodes
        Ll1 = legendre(l1)(self.mui_grid)
        Ll2 = legendre(l2)(self.mui_grid)
        nrm = 0.5*(2.*l1+1.)*(2.*l2+1.)
        return nrm*np.trapz(itgrnd*Ll1*Ll2,self.mu,axis=0)
    
    def get_covmat_1x_l1l2(self,l1,l2):
        '''
        Covariance between cross-spectrum and line 1 auto-spectrum for multipoles
        l1 and l2
        '''
        itgrnd = self.Px*(self.m1x.Pk+self.m1x.Pnoise)/self.m1x.Nmodes
        Ll1 = legendre(l1)(self.mui_grid)
        Ll2 = legendre(l2)(self.mui_grid)
        nrm = 0.5*(2.*l1+1.)*(2.*l2+1.)
        return nrm*np.trapz(itgrnd*Ll1*Ll2,self.mu,axis=0)
    
    def get_covmat_2x_l1l2(self,l1,l2):
        '''
        Covariance between cross-spectrum and line 2 auto-spectrum for multipoles
        l1 and l2
        '''
        itgrnd = self.Px*(self.m2x.Pk+self.m2x.Pnoise)/self.m1x.Nmodes
        Ll1 = legendre(l1)(self.mui_grid)
        Ll2 = legendre(l2)(self.mui_grid)
        nrm = 0.5*(2.*l1+1.)*(2.*l2+1.)
        return nrm*np.trapz(itgrnd*Ll1*Ll2,self.mu,axis=0)
    
    def get_covmat_12(self,Nmul):
        '''
        Portion of the covariance matrix correlating the line 1 and 2
        auto-spectra for the first Nmul multipoles
        '''
        covmat = np.zeros((self.nk*Nmul,self.nk*Nmul))#*self.Px.unit**2
        nk = self.nk
        for ii in range(0,Nmul):
            for jj in range(0,Nmul):
                l1 = 2*ii
                l2 = 2*jj
                cmi = self.get_covmat_12_l1l2(l1,l2)
                covmat[ii*nk:(ii+1)*nk,jj*nk:(jj+1)*nk] = np.diag(cmi)
        return covmat
    
    def get_covmat_1x(self,Nmul):
        '''
        Portion of the covariance matrix correlating the line 1 auto
        spectrum and the cross-spectrum for the first Nmul multipoles
        '''
        covmat = np.zeros((self.nk*Nmul,self.nk*Nmul))#*self.Px.unit**2
        nk = self.nk
        for ii in range(0,Nmul):
            for jj in range(0,Nmul):
                l1 = 2*ii
                l2 = 2*jj
                cmi = self.get_covmat_1x_l1l2(l1,l2)
                covmat[ii*nk:(ii+1)*nk,jj*nk:(jj+1)*nk] = np.diag(cmi)
        return covmat
    
    def get_covmat_2x(self,Nmul):
        '''
        Portion of the covariance matrix correlating the line 2 auto
        spectrum and the cross-spectrum for the first Nmul multipoles
        '''
        covmat = np.zeros((self.nk*Nmul,self.nk*Nmul))#*self.Px.unit**2
        nk = self.nk
        for ii in range(0,Nmul):
            for jj in range(0,Nmul):
                l1 = 2*ii
                l2 = 2*jj
                cmi = self.get_covmat_2x_l1l2(l1,l2)
                covmat[ii*nk:(ii+1)*nk,jj*nk:(jj+1)*nk] = np.diag(cmi)
        return covmat
    
    @cached_property
    def covmat_12(self):
        '''
        Portion of the covariance matrix correlating the line 1 and 2
        auto-spectra
        '''
        return self.get_covmat_12(self.Nmul)
    
    @cached_property
    def covmat_1x(self):
        '''
        Portion of the covariance matrix correlating the line 1 auto
        spectrum and the cross-spectrum
        '''
        return self.get_covmat_1x(self.Nmul)
    
    @cached_property
    def covmat_2x(self):
        '''
        Portion of the covariance matrix correlating the line 2 auto
        spectrum and the cross-spectrum
        '''
        return self.get_covmat_2x(self.Nmul)
        
    @cached_property
    def Pk_full(self):
        '''
        The full power spectrum data set, combination of all of the
        multipoles each for the line 1 auto-spectrum, the cross-spectrum,
        and the line 2 auto-spectrum.
        '''
        return np.concatenate((self.m1x.Pk_combined,self.Px_combined,
                               self.m2x.Pk_combined))
    
    @cached_property
    def Pk_signal_full(self):
        '''
        The full power spectrum data set, combination of all of the
        multipoles each for the line 1 auto-spectrum, the cross-spectrum,
        and the line 2 auto-spectrum.
        '''
        return np.concatenate((self.m1x.Pk_signal_combined,self.Px_signal_combined,
                               self.m2x.Pk_signal_combined))
    
    @cached_property
    def Pk_signal_full(self):
        '''
        The full power spectrum data set, combination of all of the
        multipoles each for the line 1 auto-spectrum, the cross-spectrum,
        and the line 2 auto-spectrum.
        '''
        return np.concatenate((self.m1x.Pk_signal_combined,self.Px_combined,
                               self.m2x.Pk_signal_combined))
    
    @cached_property
    def covmat_full(self):
        '''
        Full covariance matrix of all multipoles of the cross- and two
        auto-spectra
        '''
        c1 = np.concatenate((self.m1x.covmat,self.covmat_1x,
                             self.covmat_12),axis=1)
        c2 = np.concatenate((self.covmat_1x,self.covmat_x,self.covmat_2x),
                             axis=1)
        c3 = np.concatenate((self.covmat_12,self.covmat_2x,
                             self.m2x.covmat),axis=1)
        return np.concatenate((c1,c2,c3),axis=0)
    
    @cached_property
    def SNR_full(self):
        '''
        Full signal-to-noise ratio of combination of auto- and cross-spectra
        '''
        try:
            cm_dot_p = np.linalg.solve(self.covmat_full,self.Pk_signal_full)
        except np.linalg.LinAlgError as err:
            if 'Singular matrix' in str(err):
                print('SNR too high, covmat_full is singular. Reporting SNR for line 1')
                return self.m1x.SNR_multipoles
            else:
                raise
        return np.sqrt(np.dot(self.Pk_signal_full,cm_dot_p))
        
        
    
    ############################
    # Update and reset methods #
    ############################
    
    def update(self,**new_params):
        # Check if params dict contains valid parameters
        check_params(new_params,self._default_params)
                    
        # If model_type or model_name is updated, check if model_name is valid
        if ('model_type' in new_params) and ('model_name' in new_params):
            check_model(new_params['model_type'],new_params['model_name'])
        elif 'model_type' in new_params:
            check_model(new_params['model_type'],self.model_name)
        elif 'model_name' in new_params:
            check_model(self.model_type,new_params['model_name'])
        
        for attribute in self._update_list:
            if attribute != ('m1','m2','m1x','m1x'):
                delattr(self,attribute)
        
        self._update_list = []
        
        for key in new_params:
            setattr(self,key,new_params[key])
 
    def reset(self):
        self.update(**self._input_params)
        
    
    
    
    
    
    