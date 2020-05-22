'''
Base module for generating models of line intensity maps
'''

import numpy as np
import inspect
import astropy.units as u
import astropy.constants as cu

from scipy.interpolate import interp1d
from scipy.special import sici
from scipy.special import legendre


import mass_function_library as MFL
import camb

from _utils import cached_property,get_default_params,check_params,ulogspace,ulinspace
from _utils import check_model
from _utils import check_bias_model
from _utils import log_interp1d
import luminosity_functions as lf
import mass_luminosity as ml
import bias_fitting_functions as bm

class LineModel(object):
    '''
    An object containing all of the relevant astrophysical quantities of a
    LIM model.
    
    The purpose of this class is to calculate many quantities associated with
    a line intensity map, for now mostly with the goal of predicting a power
    spectrum from a different model.
    
    Models are defined by a number of input parameters defining a cosmology,
    and a prescription for assigning line luminosities.  These luminosities
    can either be drawn directly from a luminosity function, or assigned
    following a mass-luminosity relation.  In the latter case, abuundances are
    assigned following a mass function computed with pylians.
    
    Most methods in this class are given as @cached_properties, which means
    they are computed once when the method is called, then the outputs are
    saved for future calls.  Input parameters can be changed with the included
    update() method, which when called will reset the cached properties so
    they can be recomputed with the new values.
    
    WARNING: Parameter values should ONLY be changed with the update() method.
             Changing values any other way will NOT refresh the cached values
    
    Note that the lim package uses astropy units througout.  Input parameters
    must be assigned with the proper dimensions, or an error will be raised.
    
    New models can be easily created. In the case of 'LF' models, add a new
    function with the desired form to luminosity_functions.py.  For 'ML'
    models, do the same for mass_luminosity.py
    
    Defaults to the model from Breysse et al. (2017)
    
    INPUT PARAMETERS:
    cosmo_input:    Dictionary to read and feed to camb

    model_type:     Either 'LF' for a luminosity function model or 'ML' for a
                    mass-luminosity model.  Any other value will raise an
                    error.  Note that some outputs are only available for one
                    model_type. (Default = 'LF')
    
    model_name:     Name of line emission model.  Must be the name of a
                    function defined in luminosity_functions.py (for
                    model_name='LF') or mass_luminosity.py (for model_name=
                    'ML'). (Default = 'SchCut')
                    
    model_par:      Dictionary containing the parameters of the chosen model
                    (Default = Parameters of Breysse et al. 2017 CO model)
                    
    hmf_model:      Fitting function for the halo model using Pylians. 
                    To choose among 'ST, 'Tinker', 'Tinker10'
                    'Crocce', 'Jenkins', 'Warren', 'Watson', 'Angulo'
                    (Default: 'ST').
                    
    bias_model:     Fitting function for the bias model.
                    
    nu:             Rest frame emission frequency of target line
                    (Default = 115 GHz, i.e. CO(1-0))
                    
    nuObs:          Observing frequency, defines target redshift
                    (Default = 30 GHz, i.e. z=2.8 for CO)
                    
    Mmin:           Minimum mass of line-emitting halo. (Default = 10^9 Msun)
    
    Mmax:           Maximum mass of line emitting halo.  Rarely a physical
                    parameter, but necessary to define high-mass cutoffs for
                    mass function integrals (Default = 10^15 Msun)
                    
    nM:             Number of halo mass points (Default = 5000)
    
    Lmin:           Minimum luminosity for luminosity function calculations
                    (Default = 100 Lsun)
                    
    Lmax:           Maximum luminosity for luminosity function calculations
                    (Default = 10^8 Lsun)
                    
    nL:             Number of luminosity points (Default = 5000)
    
    kmin:           Minimum wavenumber for power spectrum computations
                    (Default = 10^-2 Mpc^-1)
                    
    kmax:           Maximum wavenumber for power sepctrum computations
                    (Default = 10 Mpc^-1)
    
    nk:             Number of wavenumber points (Default = 100)
    
    k_kind:         Whether you want k vector to be binned in linear or
                    log space (options: 'linear','log'; Default:'log')
    
    sigma_scatter:  Width of log-scatter in mass-luminosity relation, defined
                    as the width of a Gaussian distribution in log10(L) which
                    preserves the overall mean luminosity.  See Li et al.
                    (2015) for more information. (Default = 0.0)
                    
    fduty:          Duty cycle for line emission, as defined in Pullen et al.
                    2012 (Default = 1.0)
                    
    do_onehalo:     Bool, if True power spectra are computed with one-halo
                    term included (Default = False)
                    
    do_Jysr:        Bool, if True quantities are output in Jy/sr units rather
                    than brightness temperature (Default = False)
                    
    do_RSD:         Bool, if True power spectrum includes RSD (Default:False)
    
    sigma_NL:       Scale of Nonlinearities (Default: 7 Mpc)
    
    nmu:            number of mu bins
    
    nonlinear_pm:   Bool, if True, non linear power spectrum computed
                    (Default: False)
                    If True, it will take longer
    
    FoG_damp:       damping term for Fingers of God (Default:'Lorentzian'
    
    smooth:         smoothed power spectrum, convoluted with beam/channel
                    (Default: False)
    
    DOCTESTS:
    >>> m = LineModel()
    >>> m.hubble
    0.6774
    >>> m.z
    <Quantity 2.833...>
    >>> m.dndL[0:2]
    <Quantity [  7.08...e-05,  7.15...e-05] 1 / (Mpc3 solLum)>
    >>> m.bavg
    <Quantity 1.983...>
    >>> m.nbar
    <Quantity 0.281... 1 / Mpc3>
    >>> m.Tmean
    <Quantity 1.769... uK>
    >>> m.Pk[0:2]
    <Quantity [ 108958..., 109250...] Mpc3 uK2>
    '''
    
    def __init__(self,
                 cosmo_input=dict(f_NL=0,H0=67.0,cosmomc_theta=None,ombh2=0.022, omch2=0.12, 
                               omk=0.0, neutrino_hierarchy='degenerate', 
                               num_massive_neutrinos=1, mnu=0.06, nnu=3.046, 
                               YHe=None, meffsterile=0.0, standard_neutrino_neff=3.046, 
                               TCMB=2.7255, tau=None, deltazrei=None, bbn_predictor=None, 
                               theta_H0_range=[10, 100],w=-1.0, wa=0., cs2=1.0, 
                               dark_energy_model='ppf',As=2e-09, ns=0.96, nrun=0, 
                               nrunrun=0.0, r=0.0, nt=None, ntrun=0.0, 
                               pivot_scalar=0.05, pivot_tensor=0.05,
                               parameterization=2,halofit_version='mead'),
                 model_type='LF',
                 model_name='SchCut', 
                 model_par={'phistar':9.6e-11*u.Lsun**-1*u.Mpc**-3,
                 'Lstar':2.1e6*u.Lsun,'alpha':-1.87,'Lmin':5000*u.Lsun},
                 hmf_model='ST',
                 bias_model='ST99',
                 bias_par={}, #Otherwise, write a dict with the corresponding values
                 nu=115*u.GHz,
                 nuObs=30*u.GHz,
                 Mmin=1e9*u.Msun,
                 Mmax=1e15*u.Msun,
                 nM=5000,
                 Lmin=100*u.Lsun,
                 Lmax=1e8*u.Lsun,
                 nL=5000,
                 kmin = 1e-2*u.Mpc**-1,
                 kmax = 10.*u.Mpc**-1,
                 nk = 100,
                 k_kind = 'log',
                 sigma_scatter=0.,
                 fduty=1.,
                 do_onehalo=False,
                 do_Jysr=False,
                 do_RSD=True,
                 sigma_NL=7*u.Mpc,
                 nmu=1000,
                 nonlinear_pm=False,
                 FoG_damp='Lorentzian',
                 smooth=False):
        

        # Get list of input values to check type and units
        self._lim_params = locals()
        self._lim_params.pop('self')
        
        # Get list of input names and default values
        self._default_lim_params = get_default_params(LineModel.__init__)
        # Check that input values have the correct type and units
        check_params(self._lim_params,self._default_lim_params)
        
        # Set all given parameters
        for key in self._lim_params:
            setattr(self,key,self._lim_params[key])

            
        # Create overall lists of parameters (Only used if using one of 
        # lim's subclasses
        self._input_params = {} # Don't want .update to change _lim_params
        self._default_params = {}
        self._input_params.update(self._lim_params)
        self._default_params.update(self._default_lim_params)
        
        # Create list of cached properties
        self._update_list = []
        
        # Check if model_name is valid
        check_model(self.model_type,self.model_name)
        check_bias_model(self.bias_model)

        #Set cosmology and call camb
        self.cosmo_input = self._default_params['cosmo_input']
        for key in cosmo_input:
            self.cosmo_input[key] = cosmo_input[key]

        self.camb_pars = camb.set_params(H0=self.cosmo_input['H0'], cosmomc_theta=self.cosmo_input['cosmomc_theta'],
             ombh2=self.cosmo_input['ombh2'], omch2=self.cosmo_input['omch2'], omk=self.cosmo_input['omk'],
             neutrino_hierarchy=self.cosmo_input['neutrino_hierarchy'], 
             num_massive_neutrinos=self.cosmo_input['num_massive_neutrinos'],
             mnu=self.cosmo_input['mnu'], nnu=self.cosmo_input['nnu'], YHe=self.cosmo_input['YHe'], 
             meffsterile=self.cosmo_input['meffsterile'], 
             standard_neutrino_neff=self.cosmo_input['standard_neutrino_neff'], 
             TCMB=self.cosmo_input['TCMB'], tau=self.cosmo_input['tau'], 
             deltazrei=self.cosmo_input['deltazrei'], 
             bbn_predictor=self.cosmo_input['bbn_predictor'], 
             theta_H0_range=self.cosmo_input['theta_H0_range'],
             w=self.cosmo_input['w'], cs2=self.cosmo_input['cs2'], 
             dark_energy_model=self.cosmo_input['dark_energy_model'],
             As=self.cosmo_input['As'], ns=self.cosmo_input['ns'], 
             nrun=self.cosmo_input['nrun'], nrunrun=self.cosmo_input['nrunrun'], 
             r=self.cosmo_input['r'], nt=self.cosmo_input['nt'], ntrun=self.cosmo_input['ntrun'], 
             pivot_scalar=self.cosmo_input['pivot_scalar'], 
             pivot_tensor=self.cosmo_input['pivot_tensor'],
             parameterization=self.cosmo_input['parameterization'],
             halofit_version=self.cosmo_input['halofit_version'])
             
        self.camb_pars.WantTransfer=True
        
        
    #################
    # Get cosmology #
    #################
    
    @cached_property
    def cosmo(self):
        self.camb_pars.set_matter_power(redshifts=[self.z,0], 
                                        kmax=self.kmax.value)
        self.camb_pars.NonLinear = camb.model.NonLinear_both
        return camb.get_results(self.camb_pars)
   
    @cached_property
    def transfer(self):
       return self.cosmo.get_matter_transfer_data()
       
    @cached_property
    def f_NL(self):
        return self.cosmo_input['f_NL']
        
    @cached_property
    def Alcock_Packynski_params(self):
        '''
        Returns the quantities needed for the rescaling for Alcock-Paczyinski
           Da/rs, H*rs, DV/rs
        '''
        BAO_pars = self.cosmo.get_BAO([self.z],self.camb_pars)
        #This is rs/DV, H, DA, F_AP
        rs = self.cosmo.get_derived_params()['rdrag']
        return BAO_pars[0,2]/rs,BAO_pars[0,1]*rs,BAO_pars[0,0]**-1.
    
    ####################
    # Define 1/h units #
    ####################
    @cached_property
    def hubble(self):
        '''
        Normalized hubble parameter (H0.value/100). Used for converting to
        1/h units.
        '''
        return self.camb_pars.H0/100.
    
    @cached_property
    def Mpch(self):
        '''
        Mpc/h unit, required for interacting with hmf outputs
        '''
        return u.Mpc / self.hubble
        
    @cached_property
    def Msunh(self):
        '''
        Msun/h unit, required for interacting with hmf outputs
        '''
        return u.Msun / self.hubble
    
    #################################
    # Properties of target redshift #
    #################################  
    @cached_property
    def z(self):
        '''
        Emission redshift of target line
        '''
        return (self.nu/self.nuObs-1.).value
    
    @cached_property
    def H(self):
        '''
        Hubble parameter at target redshift
        '''
        return self.cosmo.hubble_parameter(self.z)*(u.km/u.Mpc/u.s)
        
    @cached_property
    def CLT(self):
        '''
        Coefficient relating luminosity density to brightness temperature
        '''
        if self.do_Jysr:
            x = cu.c/(4.*np.pi*self.nu*self.H*(1.*u.sr))
            return x.to(u.Jy*u.Mpc**3/(u.Lsun*u.sr))
        else:
            x = cu.c**3*(1+self.z)**2/(8*np.pi*cu.k_B*self.nu**3*self.H)
            return x.to(u.uK*u.Mpc**3/u.Lsun)
    
    #########################################
    # Masses, luminosities, and wavenumbers #
    #########################################
    @cached_property
    def M(self):
        '''
        List of masses for computing mass functions and related quantities
        '''
        # ~ # Make sure masses fall within bounds defined for hmf
        # ~ logMmin_h = np.log10((self.Mmin.to(self.Msunh)).value)
        # ~ logMmax_h = np.log10((self.Mmax.to(self.Msunh)).value)
        
        # ~ if logMmin_h<hmf_logMmin:
            # ~ self.h.update(Mmin=logMmin_h/2.)
        # ~ elif logMmax_h>hmf_logMmax:
            # ~ self.h.update(Mmax=logMmax_h*2.)
        
        return ulogspace(self.Mmin,self.Mmax,self.nM)
    
    @cached_property
    def L(self):
        '''
        List of luminosities for computing luminosity functions and related
        quantities.
        '''
        return ulogspace(self.Lmin,self.Lmax,self.nL)
        
    @cached_property
    def k_edge(self):
        '''
        Wavenumber bin edges
        '''
        if self.k_kind == 'log':
            return ulogspace(self.kmin,self.kmax,self.nk+1)
        elif self.k_kind == 'linear':
            return ulinspace(self.kmin,self.kmax,self.nk+1)
        else:
            raise Exception('Invalid value of k_kind. Choose between\
             linear or log')
    
    @cached_property
    def k(self):
        '''
        List of wave numbers for power spectrum and related quantities
        '''
        Nedge = self.k_edge.size
        return (self.k_edge[0:Nedge-1]+self.k_edge[1:Nedge])/2.
    
    @cached_property
    def dk(self):
        '''
        Width of wavenumber bins
        '''
        return np.diff(self.k_edge)
        
    @cached_property
    def mu_edge(self):
        '''
        cos theta bin edges
        '''
        return np.linspace(-1,1,self.nmu+1)
        
    @cached_property
    def mu(self):
        '''
        List of mu (cos theta) values for anisotropic, or integrals
        '''
        Nedge = self.mu_edge.size
        return (self.mu_edge[0:Nedge-1]+self.mu_edge[1:Nedge])/2.
        
    @cached_property
    def dmu(self):
        '''
        Width of cos theta bins
        '''
        return np.diff(self.mu_edge)
        
    @cached_property
    def ki_grid(self):
        '''
        Grid of k for anisotropic
        '''
        return np.meshgrid(self.k,self.mu)[0]
        
    @cached_property
    def mui_grid(self):
        '''
        Grid of mu for anisotropic
        '''
        return np.meshgrid(self.k,self.mu)[1]
        
    @cached_property
    def k_par(self):
        '''
        Grid of k_parallel
        '''
        return self.ki_grid*self.mui_grid
        
    @cached_property
    def k_perp(self):
        '''
        Grid of k_perpendicular
        '''
        return self.ki_grid*np.sqrt(1.-self.mui_grid**2.)
    
    #####################
    # Line luminosities #
    #####################
    @cached_property
    def dndL(self):
        '''
        Line luminosity function.  Only available if model_type='LF'
        
        TODO:
        Add ability to compute dndL for non-monotonic L(M) model
        '''
        #if self.model_type!='LF':
        #    raise Exception('For now, dndL is only available for LF models')
        
        if self.model_type=='LF':
            return getattr(lf,self.model_name)(self.L,self.model_par)
        else:
            # Check if L(M) is monotonic
            if not np.all(np.diff(self.LofM)>=0):
                raise Exception('For now, dndL is only available for ML '+
                        'models where L(M) is monotnoically increasing')
            # Compute masses corresponding to input luminosities
            MofL = (log_interp1d(self.LofM,self.M,bounds_error=False,fill_value=0.)
                    (self.L.value)*self.M.unit)
            # Mass function at these masses
            dndM_MofL = log_interp1d(self.M,self.dndM,bounds_error=False,
                            fill_value=0.)(MofL.value)*self.dndM.unit
            # Derivative of L(M) w.r.t. M
            dM = MofL*1.e-5
            L_p = getattr(ml,self.model_name)(MofL+dM,self.model_par,self.z)
            L_m = getattr(ml,self.model_name)(MofL-dM,self.model_par,self.z)
            dLdM = (L_p-L_m)/(2*dM)
            
            dndL = dndM_MofL/dLdM
            
            # Cutoff M>Mmax and M<Mmin
            dndL[MofL<self.Mmin] = 0.*dndL.unit
            dndL[MofL>self.Mmax] = 0.*dndL.unit
            
            # Include scatter
            if self.sigma_scatter>0.:
                s = self.sigma_scatter
                # Mean-preserving scatter PDF:
                P_scatter = (lambda x:
                    (np.exp(-(np.log(x)+s**2/2.)**2/(2*s**2))/
                     (x*s*np.sqrt(2*np.pi))))
                
                dndL_s = np.zeros(dndL.size)*dndL.unit
                for ii in range(0,self.nL):
                    Li = self.L[ii]
                    itgrnd = dndL*P_scatter(Li/self.L)/self.L
                    dndL_s[ii] = np.trapz(itgrnd,self.L)
                    
                return dndL_s
            else:
                return dndL 
        
    @cached_property
    def LofM(self):
        '''
        Line luminosity as a function of halo mass.
        
        'LF' models need this to compute average bias, and always assume that
        luminosity is linear in M.  This is what is output when this function
        is called on an LF model.  NOTE that in this case, this should NOT be
        taken to be an accurate physical model as it will be off by an overall
        constant.
        '''
        
        if self.model_type=='LF':
            LF_par = {'A':1.,'b':1.,'Mcut_min':self.Mmin,'Mcut_max':self.Mmax}
            L = getattr(ml,'MassPow')(self.M,LF_par,self.z)
        else:
            L = getattr(ml,self.model_name)(self.M,self.model_par,self.z)
        return L
        
    @cached_property
    def Pm_for_HMF(self):
        '''
        Get the matter power spectrum for pylians hmf and sigma
        '''
        kh_camb, z, Pk_camb =self.cosmo.get_nonlinear_matter_power_spectrum()
        kh = np.logspace(np.log10(kh_camb[0]),np.log10(kh_camb[-1]),512)
        P = log_interp1d(kh_camb,Pk_camb[1,:])(kh)
        return kh, P #in Mpch**-1 and Mpch**3 respectively
        
    @cached_property
    def dndM(self):
        '''
        Halo mass function, note the need to convert from 1/h units in the
        output of pylians
        
        Interpolation done in log space
        '''
        kh, P = self.Pm_for_HMF
        
        #mass vector for pylians
        Mvec = (ulogspace(self.Mmin.to(self.Msunh),self.Mmax.to(self.Msunh),256)).to(self.Msunh)
        
        #Compute Halo mass function
        mf = MFL.MF_theory(kh,P,self.camb_pars.omegam,Mvec.value,self.hmf_model,z=self.z)
        
        d = (log_interp1d(Mvec.value,mf)(self.M.to(self.Msunh).value)*
              self.Mpch**-3*self.Msunh**-1).to(self.Mpch**-3*self.Msunh**-1)
        
        return d.to(u.Mpc**-3*u.Msun**-1)
        
    @cached_property
    def sigmaM(self):
        '''
        Mass variance at targed redshift, computed using pylians
        '''
        kh, P = self.Pm_for_HMF
        
        #mass vector for pylians
        Mvec = (ulogspace(self.Mmin.to(self.Msunh),self.Mmax.to(self.Msunh),256)).to(self.Msunh)
                        
        #pylians get sigma(R). Get R
        rho_crit = 2.77536627e11*(self.Msunh*self.Mpch**-3).to(self.Msunh*self.Mpch**-3) #h^2 Msun/Mpc^3
        rhoM = rho_crit*self.camb_pars.omegam
        R = (3.0*Mvec/(4.0*np.pi*rhoM))**(1.0/3.0)

        #Get sigma(M) from pylians
        sigma_vec = np.zeros(len(R))
        
        for ir in range(0,len(R)):
            sigma_vec[ir] = MFL.sigma(kh,P,R[ir].value)        

        Mh = (self.M.to(self.Msunh)).value
        return log_interp1d(Mvec.value,sigma_vec)(Mh)
        
    @cached_property
    def mass_non_linear(self):
        '''
        Mass at which perturbations get non linear 
        ((sigma(M,z)-delta_c)**2 minimizes)
        '''
        delta_c = 1.686
        return np.argmin((self.sigmaM-delta_c)**2.)
        
    
    @cached_property
    def bofM(self):
        '''
        Halo bias as a function of mass.  Currently always uses the Tinker
        et al. 2010 fitting function
        
        TODO:
        Add fitting functions for other hmf_models
        '''
        
        # nonlinear overdensity
        dc = 1.686
        nu = dc/self.sigmaM
        
        bias = getattr(bm,self.bias_model)(self,dc,nu)
            
        return bias
        
    @cached_property
    def ft_NFW(self):
        '''
        Fourier transform of NFW profile, for computing one-halo term
        '''
        [ki,Mi] = np.meshgrid(self.k,self.M)
        # Wechsler et al. 2002 cocentration fit
        a_c =0.1*np.log10((Mi.to(u.Msun)).value)-0.9
        a_c[a_c<0.1] = 0.1
        con = (4.1/(a_c*(1.+self.z)))
        f = np.log(1.+con)-con/(1.+con)
        Delta = 200
        
        rho_crit = (2.77536627e11*self.Msunh*self.Mpch**-3).to(self.Msunh*self.Mpch**-3) #h^2 Msun/Mpc^3
        rhoM = rho_crit*self.camb_pars.omegam
        rhobar = rhoM.to(u.Msun/u.Mpc**3)
        
        Rvir = (3*Mi/(4*np.pi*Delta*rhobar))**(1./3.)
        x = ((ki*Rvir/con).decompose()).value        
        si_x, ci_x = sici(x)
        si_cx, ci_cx = sici((1.+con)*x)
        rho_km = (np.cos(x)*(ci_cx - ci_x) +
                  np.sin(x)*(si_cx - si_x) - np.sin(con*x)/((1.+con)*x))
        return rho_km/f
        
        
    @cached_property
    def bavg(self):
        '''
        Average luminosity-weighted bias for the given cosmology and line
        model.  ASSUMED TO BE WEIGHTED LINERALY BY MASS FOR 'LF' MODELS
        
        Includes the effect of f_NL
        '''
        dc = 1.686
        Delta_b = 0.
        
        if self.model_type == 'TOY':
            b_line = self.model_par['bmean']
        else:
            # Integrands for mass-averaging
            itgrnd1 = self.LofM*self.bofM*self.dndM
            itgrnd2 = self.LofM*self.dndM
            
            b_line = np.trapz(itgrnd1,self.M) / np.trapz(itgrnd2,self.M)
            
        if self.f_NL != 0:
            Delta_b = (b_line-1.)*self.f_NL*dc*                                      \
                      3.*self.camb_pars.omegam*(100.*self.hubble*(u.km/u.s/u.Mpc))**2./   \
                     (cu.c.to(u.km/u.s)**2.*self.k2Tk)
        
        return b_line + Delta_b
    
    @cached_property
    def nbar(self):
        '''
        Mean number density of galaxies, computed from the luminosity function
        in 'LF' models and from the mass function in 'ML' models
        '''
        if self.model_type=='LF':
            nbar = np.trapz(self.dndL,self.L)
        else:
            nbar = np.trapz(self.dndM,self.M)
        return nbar
        
    #############################
    # Power spectrum quantities #
    #############################
    @cached_property
    def RSD(self):
        '''
        Kaiser factor and FoG for RSD
        '''
        if self.do_RSD == True:
            f = self.cosmo.get_fsigma8()[0]/self.transfer.sigma_8[0]
            #D = self.transfer.transfer_data[6][1,0]/self.transfer.transfer_data[6][1,1]
                        
            kaiser = (1.+f/self.bavg*self.mui_grid**2.)**2. #already squared
            
            if self.FoG_damp == 'Lorentzian':
                FoG = (1.+0.5*(self.k_par*self.sigma_NL).decompose()**2.)**-2.
            elif self.FoG_damp == 'Gaussian':
                FoG = np.exp(-((self.k_par*self.sigma_NL)**2.)
                        .decompose()) 
            else:
                raise Exception('Only Lorentzian or Gaussian damping terms for FoG')
                
            return FoG*kaiser
        else:
            return np.ones(self.Pm.shape)

    @cached_property
    def k2Tk(self):
        '''
        Get the k^2T(k), where T(k) is the transfer function.
        To use for the non gaussian bias.
        '''
        kvec = (self.transfer.transfer_data[0][:,0]*self.Mpch**-1).to(u.Mpc**-1)
        Tk = self.transfer.transfer_data[6][:,0]
        #Already k^2*T(k)
        k2tk = kvec**2*Tk/np.max(self.transfer.transfer_data[6][:,1])
        
        return log_interp1d(kvec,k2tk)(self.ki_grid.value)*(kvec**2).unit
        
    @cached_property
    def Pm(self):
        '''
        Matter power spectrum computed from camb. 
        '''
        if not self.nonlinear_pm:
            kh_camb, z, Pk_camb =self.cosmo.get_linear_matter_power_spectrum(params=self.camb_pars)
        else:
            kh_camb, z, Pk_camb =self.cosmo.get_nonlinear_matter_power_spectrum(params=self.camb_pars)
            
        kh = (self.ki_grid.to(self.Mpch**-1)).value
        P = log_interp1d(kh_camb,Pk_camb[1,:])(kh)
        
        return (P*self.Mpch**3).to(self.Mpch**3).to(u.Mpc**3)
                
    
    @cached_property
    def Lmean(self):
        '''
        Sky-averaged luminosity density at nuObs from target line.  Has
        two cases for 'LF' and 'ML' models
        '''
        if self.model_type=='LF':
            itgrnd = self.L*self.dndL
            Lbar = np.trapz(itgrnd,self.L)
        elif self.model_type == 'ML':
            itgrnd = self.LofM*self.dndM
            Lbar = np.trapz(itgrnd,self.M)*self.fduty
            # Special case for Tony Li model- scatter does not preserve LCO
            if self.model_name=='TonyLi':
                alpha = self.model_par['alpha']
                sig_SFR = self.model_par['sig_SFR']
                Lbar = Lbar*np.exp((alpha**-2-alpha**-1)
                                    *sig_SFR**2*np.log(10)**2/2.)
        return Lbar
        
    @cached_property
    def L2mean(self):
        '''
        Sky-averaged squared luminosity density at nuObs from target line.  Has
        two cases for 'LF' and 'ML' models
        '''
        if self.model_type=='LF':
            itgrnd = self.L**2*self.dndL
            L2bar = np.trapz(itgrnd,self.L)
        elif self.model_type=='ML':
            itgrnd = self.LofM**2*self.dndM
            L2bar = np.trapz(itgrnd,self.M)*self.fduty
            # Add L vs. M scatter
            L2bar = L2bar*np.exp(self.sigma_scatter**2*np.log(10)**2)
            # Special case for Tony Li model- scatter does not preserve LCO
            if self.model_name=='TonyLi':
                alpha = self.model_par['alpha']
                sig_SFR = self.model_par['sig_SFR']
                L2bar = L2bar*np.exp((2./alpha**2-1./alpha)
                                    *sig_SFR**2*np.log(10)**2)

        return L2bar
        
    @cached_property
    def Tmean(self):
        '''
        Sky-averaged brightness temperature at nuObs from target line.  Has
        two cases for 'LF' and 'ML' models.
        You can direcyly input Tmean using TOY model
        '''
        if self.model_type == 'TOY':
            return self.model_par['Tmean']
        else:
            return self.CLT*self.Lmean
        
    @cached_property
    def Pshot(self):
        '''
        Shot noise amplitude for target line at frequency nuObs.  Has two
        cases for 'LF' and 'ML' models. 
        You can directly input T2mean using TOY model
        '''
        
        if self.model_type == 'TOY':
            return self.model_par['T2mean']
        else:
            return self.CLT**2*self.L2mean
        
    @cached_property
    def Pk_twohalo(self):
        '''
        Two-halo term in power spectrum, equal to Tmean^2*bavg^2*Pm if
        do_onehalo=False
        '''
        if self.do_onehalo:
            if self.model_type=='LF':
                print("One halo term only available for ML models")
                wt = self.Tmean*self.bavg
            else:
                wt = np.zeros(self.ki_grid.shape)
                Mass_Dep = self.LofM*self.bofM*self.dndM
                itgrnd = (np.tile(Mass_Dep,[self.k.size,1]).transpose()
                            *self.ft_NFW)
                wt[:,:] = self.CLT*np.trapz(itgrnd,self.M,axis=0)
                # Special case for SFR(M) scatter in Tony Li model
                if self.model_name=='TonyLi':
                    alpha = self.model_par['alpha']
                    sig_SFR = self.model_par['sig_SFR']
                    wt = wt*np.exp((alpha**-2-alpha**-1)
                                    *sig_SFR**2*np.log(10)**2/2.)
        else:
            wt = self.Tmean*self.bavg
        
        return wt**2*self.Pm
        
    @cached_property
    def Pk_onehalo(self):
        '''
        One-halo term in power spectrum
        '''
        if self.do_onehalo:
            Mass_Dep = self.LofM**2.*self.dndM
            itgrnd = (np.tile(Mass_Dep,[self.k.size,1]).transpose()
                        *self.ft_NFW**2.)
                        
            # Special case for Tony Li model- scatter does not preserve LCO
            if self.model_name=='TonyLi':
                alpha = self.model_par['alpha']
                sig_SFR = self.model_par['sig_SFR']
                itgrnd = itgrnd*np.exp((2.*alpha**-2-alpha**-1)
                                    *sig_SFR**2*np.log(10)**2)
            wt = np.zeros(self.ki_grid.shape)
            wt[:,:] = np.trapz(itgrnd,self.M,axis=0)
            return self.CLT**2.*wt
        else:
            return np.zeros(self.Pm.shape)*self.Pshot.unit
    
    @cached_property    
    def Pk_clust(self):
        '''
        Clustering power spectrum of target line, i.e. power spectrum without
        shot noise.
        '''
        return (self.Pk_twohalo+self.Pk_onehalo)*self.RSD
    
    @cached_property    
    def Pk_shot(self):
        '''
        Shot-noise power spectrum of target line, i.e. power spectrum without
        clustering
        '''
        return self.Pshot*np.ones(self.Pm.shape)
    
    @cached_property    
    def Pk(self):
        '''
        Full line power spectrum including both clustering and shot noise 
        as function of k and mu
        '''
        if self.smooth:
            return self.Wk*(self.Pk_clust+self.Pk_shot)
        else:
            return self.Pk_clust+self.Pk_shot
        
    @cached_property
    def Pk_0(self):
        '''
        Monopole of the power spectrum as function of k
        '''
        return 0.5*np.trapz(self.Pk,self.mu,axis=0)
        
    @cached_property
    def Pk_2(self):
        '''
        Quadrupole of the power spectrum as function of k
        '''
        L2 = legendre(2)
        return 2.5*np.trapz(self.Pk*L2(self.mui_grid),self.mu,axis=0)
        
    @cached_property
    def Pk_4(self):
        '''
        Hexadecapole of the power spectrum as function of k
        '''
        L4 = legendre(4)
        return 4.5*np.trapz(self.Pk*L4(self.mui_grid),self.mu,axis=0)
        
    def Pk_l(self,l):
        '''
        Multipole l of the power spectrum
        '''
        if l == 0:
            return self.Pk_0
        elif l == 2:
            return self.Pk_2
        elif l == 4:
            return self.Pk_4
        else:
            Ll = legendre(l)
            return (2.*l+1.)/2.*np.trapz(self.Pk*Ll(self.mui_grid),
                                        self.mu,axis=0)
        
    def save_in_file(self, name, lis):
        '''
        Save the list (i.e. [k, Pk, sk]) in a file with path 'name'
        Arguments: name = <path>, lis = <what to save>
        '''
        # ~ if 'LIST' not in kwargs:
            # ~ raise TypeError("Please input the list using the 'LIST=[]' as argument")
        # ~ lis = kwargs['LIST']
        # ~ if 'name' in kwargs:
            # ~ name = kwargs['name']
        # ~ else:
            # ~ name = 'test.txt'
        LEN = len(lis[0])
        lenlis = len(lis)
        for i in range(1, lenlis):
            if len(lis[i]) != LEN:
                raise Exception('ALL items in the list to save MUST be 1d arrays with the same length!')
        
        MAT = np.zeros((LEN,lenlis))
        header = 'Units::   '
        for i in range(0,lenlis):
            MAT[:,i] = lis[i].value
            header += str(lis[i].unit)+'\t || '

        np.savetxt(name,MAT,header=header)
        return
        
    ########################################################################
    # Method for updating input parameters and resetting cached properties #
    ########################################################################
    def update(self, **new_params):

        # Check if params dict contains valid parameters
        #check_params(new_params,self._default_params)
        
        # If model_type or model_name is updated, check if model_name is valid
        if ('model_type' in new_params) and ('model_name' in new_params):
            check_model(new_params['model_type'],new_params['model_name'])
        elif 'model_type' in new_params:
            check_model(new_params['model_type'],self.model_name)
        elif 'model_name' in new_params:
            check_model(self.model_type,new_params['model_name'])
    
        # Clear cached properties so they can be updated
        for attribute in self._update_list:
            # Use built-in hmf updater to change h
            if attribute!='cosmo_input':
                delattr(self,attribute)
        self._update_list = []
        # Set new parameter values
        for key in new_params:
            if key!='cosmo_input':
                setattr(self, key, new_params[key])
        if 'cosmo_input' in new_params:
            temp = new_params['cosmo_input']
            for key in temp:
                self.cosmo_input[key] = temp[key]
            
            self.camb_pars = camb.set_params(H0=self.cosmo_input['H0'], cosmomc_theta=self.cosmo_input['cosmomc_theta'],
                 ombh2=self.cosmo_input['ombh2'], omch2=self.cosmo_input['omch2'], omk=self.cosmo_input['omk'],
                 neutrino_hierarchy=self.cosmo_input['neutrino_hierarchy'], 
                 num_massive_neutrinos=self.cosmo_input['num_massive_neutrinos'],
                 mnu=self.cosmo_input['mnu'], nnu=self.cosmo_input['nnu'], YHe=self.cosmo_input['YHe'], 
                 meffsterile=self.cosmo_input['meffsterile'], 
                 standard_neutrino_neff=self.cosmo_input['standard_neutrino_neff'], 
                 TCMB=self.cosmo_input['TCMB'], tau=self.cosmo_input['tau'], 
                 deltazrei=self.cosmo_input['deltazrei'], 
                 bbn_predictor=self.cosmo_input['bbn_predictor'], 
                 theta_H0_range=self.cosmo_input['theta_H0_range'],
                 w=self.cosmo_input['w'],wa=self.cosmo_input['wa'],cs2=self.cosmo_input['cs2'], 
                 dark_energy_model=self.cosmo_input['dark_energy_model'],
                 As=self.cosmo_input['As'], ns=self.cosmo_input['ns'], 
                 nrun=self.cosmo_input['nrun'], nrunrun=self.cosmo_input['nrunrun'], 
                 r=self.cosmo_input['r'], nt=self.cosmo_input['nt'], ntrun=self.cosmo_input['ntrun'], 
                 pivot_scalar=self.cosmo_input['pivot_scalar'], 
                 pivot_tensor=self.cosmo_input['pivot_tensor'], 
                 parameterization=self.cosmo_input['parameterization'],
                 halofit_version=self.cosmo_input['halofit_version'])
                 
            
    #####################################################
    # Method for resetting to original input parameters #
    #####################################################
    def reset(self):
        self.update(**self._input_params)
            
############
# Doctests #
############

if __name__ == "__main__":
    import doctest

    doctest.testmod(optionflags=doctest.ELLIPSIS |
                    doctest.NORMALIZE_WHITESPACE)
        
