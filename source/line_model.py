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
from scipy.special import erf
from scipy.stats import poisson
from scipy.fft import fft,ifft

import camb
from classy import Class

from source.tools._utils import cached_property,cached_cosmo_property,cached_vid_property,get_default_params,check_params
from source.tools._utils import check_model,check_bias_model,check_halo_mass_function_model
from source.tools._utils import log_interp1d,ulogspace,ulinspace,check_invalid_params,merge_dicts,lognormal
import source.tools._vid_tools as vt
import source.luminosity_functions as lf
import source.mass_luminosity as ml
import source.bias_fitting_functions as bm
import source.halo_mass_functions as HMF

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
    cosmo_input_camb:    Dictionary to read and feed to camb
    
    cosmo_input_class:   Dictionary to read and feed to class

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
                    To choose among 'ST, 'Tinker',
                    'Crocce', 'Jenkins', 'Warren', 'Watson', 'Watson_FOF',
                    'Angulo'
                    (Default: 'ST').
                    
    bias_model:     Fitting function for the bias model.
                    To choose among 'Mo96', 'Jing98', 'ST99', 'SMT01', 
                    'Seljak04', 'Seljak04_cosmo', 'Mandelbaum05',
                    'Tinker05', 'Tinker10', 'Manera10'
                    
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
                 Lmin=10*u.Lsun,
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
                 FoG_damp='Lorentzian',
                 smooth=False,
                 nonlinear=False,
                 #VID params
                 Tmin_VID=1.0e-2*u.uK,
                 Tmax_VID=1000.*u.uK,
                 nT=10**5,
                 do_fast_VID=True,
                 Ngal_max=100,
                 Nbin_hist=101,
                 subtract_VID_mean=False,
                 linear_VID_bin=False,
                 do_sigma_G = True,
                 sigma_G_input = 1.6):
        

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
        self._update_cosmo_list = []
        self._update_obs_list = []
        self._update_vid_list = []
        
        # Check if model_name is valid
        check_model(self.model_type,self.model_name)
        check_bias_model(self.bias_model)
        check_halo_mass_function_model(self.hmf_model)

        #Set cosmology and call camb or class
        if self.cosmo_code == 'camb':
            self.cosmo_input_camb = self._default_params['cosmo_input_camb']
            for key in cosmo_input_camb:
                self.cosmo_input_camb[key] = cosmo_input_camb[key]

            self.camb_pars = camb.set_params(H0=self.cosmo_input_camb['H0'], cosmomc_theta=self.cosmo_input_camb['cosmomc_theta'],
                 ombh2=self.cosmo_input_camb['ombh2'], omch2=self.cosmo_input_camb['omch2'], omk=self.cosmo_input_camb['omk'],
                 neutrino_hierarchy=self.cosmo_input_camb['neutrino_hierarchy'], 
                 num_massive_neutrinos=self.cosmo_input_camb['num_massive_neutrinos'],
                 mnu=self.cosmo_input_camb['mnu'], nnu=self.cosmo_input_camb['nnu'], YHe=self.cosmo_input_camb['YHe'], 
                 meffsterile=self.cosmo_input_camb['meffsterile'], 
                 standard_neutrino_neff=self.cosmo_input_camb['standard_neutrino_neff'], 
                 TCMB=self.cosmo_input_camb['TCMB'], tau=self.cosmo_input_camb['tau'], 
                 deltazrei=self.cosmo_input_camb['deltazrei'], 
                 bbn_predictor=self.cosmo_input_camb['bbn_predictor'], 
                 theta_H0_range=self.cosmo_input_camb['theta_H0_range'],
                 w=self.cosmo_input_camb['w'], cs2=self.cosmo_input_camb['cs2'], 
                 dark_energy_model=self.cosmo_input_camb['dark_energy_model'],
                 As=self.cosmo_input_camb['As'], ns=self.cosmo_input_camb['ns'], 
                 nrun=self.cosmo_input_camb['nrun'], nrunrun=self.cosmo_input_camb['nrunrun'], 
                 r=self.cosmo_input_camb['r'], nt=self.cosmo_input_camb['nt'], ntrun=self.cosmo_input_camb['ntrun'], 
                 pivot_scalar=self.cosmo_input_camb['pivot_scalar'], 
                 pivot_tensor=self.cosmo_input_camb['pivot_tensor'],
                 parameterization=self.cosmo_input_camb['parameterization'],
                 halofit_version=self.cosmo_input_camb['halofit_version'])
                 
            self.camb_pars.WantTransfer=True    
            self.camb_pars.Transfer.accurate_massive_neutrinos = True
                
        elif self.cosmo_code == 'class':
            if not 'f_NL' in self.cosmo_input_class:
                self.cosmo_input_class['f_NL'] = 0.
            pk_pars = {}
            if not ('P_k_max_1/Mpc' in self.cosmo_input_class or 'P_k_max_h/Mpc' in self.cosmo_input_class):
                pk_pars['P_k_max_1/Mpc'] = 100
            if not 'z_max_pk' in self.cosmo_input_class:
                pk_pars['z_max_pk'] = 15.
            if not 'format' in self.cosmo_input_class:
                pk_pars['format'] = 'camb'
            #accelerate class computation: check precision with defaul! (default:10)
            if not 'k_per_decade_for_pk' in self.cosmo_input_class:
                pk_pars['k_per_decade_for_pk'] = 6
            if self.nonlinear:
                pk_pars['non linear'] = 'HMCODE'

            self.class_pars = merge_dicts([self.cosmo_input_class,pk_pars])
            del self.class_pars['f_NL']
            if 'output' not in self.class_pars:
                self.class_pars['output'] = 'mPk,mTk'
            else:
                if 'mPk' not in self.class_pars['output']:
                    self.class_pars['output'] = 'mPk,' + self.class_pars['output']
                if 'mTk' not in self.class_pars['output'] or 'dTk' not in self.class_pars['output']:
                    self.class_pars['output'] = 'mTk,' + self.class_pars['output']
            
            # increase z_max_pk if needed
        else:
            raise ValueError("Only 'class' or 'camb' can be used as cosmological Boltzmann code. Please, choose between them")
        
    #################
    # Get cosmology #
    #################
    
    @cached_cosmo_property
    def zcosmo(self):
        '''
        Get the z array to call camb and interpolate cosmological quantities
        '''
        zmax = 15.  #Increase if interested in higher redshifts.
                    #If cosmo_code == 'class', change it also in the initialization of line_model()
                    # or in the input parameters
        Nz = 150
        if self.z > zmax:
            raise ValueError('Required z_obs outside interpolation region. Increase zmax or change nuObs')
        return np.linspace(0.,zmax,Nz)
    
    
    @cached_cosmo_property
    def cosmo(self):
        '''
        Compute the cosmological evolution, using camb or class
        '''
        if self.cosmo_code == 'camb':
            self.camb_pars.set_matter_power(redshifts=list(self.zcosmo))#, 
            return camb.get_results(self.camb_pars)
        else:
            cos = Class()
            cos.set(self.class_pars)
            cos.compute()
            return cos
   
   
    @cached_property
    def transfer_m(self):
        '''
        return matter transfer for the z of interest
        Argument k in 1/Mpc
        '''
        if self.cosmo_code == 'camb':
            #Find two closest (above and below) indices values for z in zcosmo
            zz = self.zcosmo[::-1] #camb sortes earlier first
            iz_down = np.where(zz - self.z < 0)[0][0]
            iz_up = iz_down - 1
            dz = zz[iz_up] - zz[iz_down]
            
            #Get the transfer
            T = self.cosmo.get_matter_transfer_data()
            kvec = (T.transfer_z('k/h',-1)*self.Mpch**-1).to(u.Mpc**-1)
            Tk_up = T.transfer_z('delta_tot',iz_up)
            Tk_down = T.transfer_z('delta_tot',iz_down)
            #interpolate in z (linear)
            Tz = Tk_down*(1.-(self.z-zz[iz_down])/dz) + Tk_up*(self.z-zz[iz_down])/dz
        else:
            T = self.cosmo.get_transfer(self.z,'camb')
            kvec = (T['k (h/Mpc)']*self.Mpch**-1).to(u.Mpc**-1)
            Tz = T['-T_tot/k2']
        #interpolate in k (linear)
        return log_interp1d(kvec,Tz)
            
   
    @cached_property
    def transfer_cb(self):
        '''
        return cdm+b transfer for the z of interest. 
        Argument k in 1/Mpc
        '''
        if self.cosmo_code == 'camb':
            #Find two closest (above and below) indices values for z in zcosmo
            zz = self.zcosmo[::-1] #camb sortes earlier first
            iz_down = np.where(zz - self.z < 0)[0][0]
            iz_up = iz_down - 1
            dz = zz[iz_up] - zz[iz_down]
            
            #Get the transfer
            T = self.cosmo.get_matter_transfer_data()
            kvec = (T.transfer_z('k/h',-1)*self.Mpch**-1).to(u.Mpc**-1)
            Tk_up = T.transfer_z('delta_nonu',iz_up)
            Tk_down = T.transfer_z('delta_nonu',iz_down)
            #interpolate in z (linear)
            Tz = Tk_down*(1.-(self.z-zz[iz_down])/dz) + Tk_up*(self.z-zz[iz_down])/dz
        else:
            T = self.cosmo.get_transfer(self.z,'camb')
            kvec = (T['k (h/Mpc)']*self.Mpch**-1).to(u.Mpc**-1)
            Tz = (self.cosmo.Omega0_cdm()*T['-T_cdm/k2'] + 
                  (self.cosmo.Omega0_m()-self.cosmo.Omega0_cdm())*T['-T_b/k2'])/self.cosmo.Omega0_m()
        #interpolate in k (linear)
        return log_interp1d(kvec,Tz)

       
    @cached_cosmo_property
    def f_NL(self):
        if self.cosmo_code == 'camb':
            return self.cosmo_input_camb['f_NL']
        else:
            return self.cosmo_input_class['f_NL']
        
        
    @cached_cosmo_property
    def Alcock_Packynski_params(self):
        '''
        Returns the quantities needed for the rescaling for Alcock-Paczyinski
           Da/rs, H*rs, DV/rs
        '''
        if self.cosmo_code == 'camb':
            BAO_pars = self.cosmo.get_BAO(self.zcosmo[1:],self.camb_pars)
            #This is rs/DV, H, DA, F_AP
            rs = self.cosmo.get_derived_params()['rdrag']
            DA = BAO_pars[:,2]
            DV = rs/BAO_pars[:,0]
            Hz = BAO_pars[:,1]
            
        elif self.cosmo_code == 'class':
            rs = self.cosmo.rs_drag()
            Nz = len(self.zcosmo[1:])
            DA, Hz = np.zeros(Nz),np.zeros(Nz)
            for i in range(Nz):
                DA[i] = self.cosmo.angular_distance(self.zcosmo[i+1])
                Hz[i] = self.cosmo.Hubble(self.zcosmo[i+1])*cu.c.to(u.km/u.s).value 
            prefact = cu.c.to(u.km/u.s).value*self.zcosmo[1:]*(1.+self.zcosmo[1:])**2
            DV = (prefact*DA**2/Hz)**(1./3.)
            
        DA_over_rs_int = interp1d(self.zcosmo[1:],DA/rs,kind='cubic',
                                  bounds_error=False,fill_value='extrapolate')
        DV_over_rs_int = interp1d(self.zcosmo[1:],DV/rs,kind='cubic',
                                  bounds_error=False,fill_value='extrapolate')
        H_times_rs_int = interp1d(self.zcosmo[1:],Hz*rs,kind='cubic',
                                  bounds_error=False,fill_value='extrapolate')
    
        return DA_over_rs_int, H_times_rs_int,DV_over_rs_int
        
        
    @cached_cosmo_property
    def PKint(self):
        '''
        Get the interpolator for the matter power spectrum as function of z and k 
        if mnu > 0 -> P_cb (without neutrinos)
        k input in 1/Mpc units
        P(k) output in Mpc^3 units
        '''
        if self.cosmo_code == 'camb':
            zmax = self.zcosmo[-1]
            nz_step=64
            if self.camb_pars.num_nu_massive != 0:
                var = 8
            else:
                var = 7
            PK = camb.get_matter_power_interpolator(self.camb_pars, zmin=0, 
                                                    zmax=zmax, nz_step=nz_step, 
                                                    zs=None, kmax=100, nonlinear=False,
                                                    var1=var, var2=var, hubble_units=False, 
                                                    k_hunit=False, return_z_k=False,
                                                    k_per_logint=None, log_interp=True, 
                                                    extrap_kmax=True)
            return PK.P
        else:
            if self.cosmo.Omega_nu != 0:
                return self.cosmo.get_pk_cb_array
            else:
                return self.cosmo.get_pk_array
        
        
    @cached_property
    def f_eff(self):
        '''
        Get the interpolator for the effective f as function of k for the 
        redshift of interest (includes the tiling to multiply by mu)
        
        if mnu = 0: f_eff = f_m; if mnu > 0: f_eff = f_cb
        '''
        dz = 1e-4
        if self.cosmo_code == 'camb':
            fs8lin = self.cosmo.get_fsigma8()
            s8lin = self.cosmo.get_sigma8()
            fz = interp1d(self.zcosmo[::-1],fs8lin/s8lin,kind='cubic')(self.z)
            #Apply correction if massive nu
            if self.camb_pars.num_nu_massive != 0:
                factor = self.transfer_m(self.k.value)/self.transfer_cb(self.k.value)
            else:
                factor = self.transfer_m(self.k.value)/self.transfer_m(self.k.value)
        else:
            fz = self.cosmo.scale_independent_growth_factor_f(self.z)
            #Apply correction if massive nu
            if self.cosmo.Omega_nu != 0:
                factor = self.transfer_m(self.k.value)/self.transfer_cb(self.k.value)
            else:
                factor = self.transfer_m(self.k.value)/self.transfer_m(self.k.value)
        return np.tile(fz*factor,(self.nmu,1))

                   
    @cached_cosmo_property
    def Dgrowth(self):
        '''
        Get the growth factor (for matter) as function of z
        (Dgrowth(z=0) = 1.)
        '''
        if self.cosmo_code == 'camb':
            s8lin = self.cosmo.get_sigma8()
            return interp1d(self.zcosmo[::-1],s8lin/s8lin[-1],kind='cubic',
                            bounds_error=False,fill_value='extrapolate')
        else:
            Nz = len(self.zcosmo)
            D = np.zeros(Nz)
            for iz in range(Nz):
                D[i] = self.cosmo.scale_independent_growth_factor(self.zcosmo[iz])
            return interp1d(self.zcosmo,D,kind='cubic',
                            bounds_error=False,fill_value='extrapolate')
        
    
    ####################
    # Define 1/h units #
    ####################
    @cached_cosmo_property
    def hubble(self):
        '''
        Normalized hubble parameter (H0.value/100). Used for converting to
        1/h units.
        '''
        if self.cosmo_code == 'camb':
            return self.camb_pars.H0/100.
        else:
            return self.cosmo.h()
    
    
    @cached_cosmo_property
    def Mpch(self):
        '''
        Mpc/h unit, required for interacting with hmf outputs
        '''
        return u.Mpc / self.hubble
        
        
    @cached_cosmo_property
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
        if self.cosmo_code == 'camb':
            return self.cosmo.hubble_parameter(self.z)*(u.km/u.Mpc/u.s)
        else:
            return self.cosmo.Hubble(self.z)*(u.Mpc**-1)*cu.c.to(u.km/u.s)
        
        
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
            raise ValueError('Invalid value of k_kind. Choose between\
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
        Line luminosity function. 
        '''
        if self.model_type=='LF':
            return getattr(lf,self.model_name)(self.L,self.model_par)
        else:
            #compute LF from the conditional LF
            if self.Lmin > self.LofM[self.LofM.value>0][0]:
                print('Warning! reduce Lmin to cover all luminosities of the model')
            if self.Lmax < np.max(self.LofM):
                print('Warning! increase Lmax to cover all luminosities of the model')
            #assume a lognormal PDF for the CLF with minimum logscatter of 0.01
            CLF_of_M = np.zeros((self.nM,self.nL))*self.dndM.unit*self.L.unit**-1
            logscatter = max(self.sigma_scatter,0.05)
            for iM in range(self.nM):
                CLF_of_M[iM,:] = lognormal(self.L,self.LofM[iM],logscatter)*self.dndM[iM]
            LF = np.zeros(self.nL)*self.L.unit**-1*self.dndM.unit*self.M.unit
            for iL in range(self.nL):
                LF[iL] = np.trapz(CLF_of_M[:,iL],self.M)
            return LF
        
        
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
            L = getattr(ml,'MassPow')(self,self.M,LF_par,self.z)
        else:
            L = getattr(ml,self.model_name)(self,self.M,self.model_par,self.z)
        return L
        
        
    @cached_property
    def dndM(self):
        '''
        Halo mass function, using functions in halo_mass_functions.py
        '''
        Mvec = self.M.to(self.Msunh)
        rho_crit = 2.77536627e11*(self.Msunh*self.Mpch**-3).to(self.Msunh*self.Mpch**-3) #h^2 Msun/Mpc^3
        #Use Omega_m or Omega_cdm+Omega_b wheter mnu = 0 or > 0
        if self.cosmo_code == 'camb':
            rhoM = rho_crit*(self.camb_pars.omegam-self.camb_pars.omeganu)
        else:
            rhoM = rho_crit*(self.cosmo.Omega0_m()-self.cosmo.Omega_nu)
        
        mf = getattr(HMF,self.hmf_model)(self,Mvec,rhoM)
        
        return mf.to(u.Mpc**-3*u.Msun**-1)
        
        
    @cached_property
    def sigmaM(self):
        '''
        Mass (or cdm+b) variance at target redshift
        '''
        #Get R(M) and P(k)
        rho_crit = 2.77536627e11*(self.Msunh*self.Mpch**-3).to(u.Msun*u.Mpc**-3) #Msun/Mpc^3
        k = np.logspace(-2,2,128)*u.Mpc**-1
        #Use rho_m or rho_cb depending on mnu
        if self.cosmo_code == 'camb':
            Pk = self.PKint(self.z,k.value)*u.Mpc**3
            rhoM = rho_crit*(self.camb_pars.omegam-self.camb_pars.omeganu)
        else:
            Pk = self.PKint(k.value,np.array([self.z]),len(k),1,0)*u.Mpc**3
            rhoM = rho_crit*(self.cosmo.Omega0_m()-self.cosmo.Omega_nu)

        R = (3.0*self.M/(4.0*np.pi*rhoM))**(1.0/3.0)

        #Get the window of a configuration space tophat
        kvec = (np.tile(k,[R.size,1]).T)
        Pk = np.tile(Pk,[R.size,1]).T
        R = np.tile(R,[k.size,1])
        x = ((kvec*R).decompose()).value
        W = 3.0*(np.sin(x) - x*np.cos(x))/(x)**3 
        
        #Compute sigma(M)
        integrnd = Pk*W**2*kvec**2/(2.*np.pi**2)
        sigma = np.sqrt(np.trapz(integrnd,kvec[:,0],axis=0))
        
        return sigma
        
        
    @cached_cosmo_property
    def sigmaMz0(self):
        '''
        Mass (or cdm+b) variance at redshift 0
        '''
        #Get R(M) and P(k)
        rho_crit = 2.77536627e11*(self.Msunh*self.Mpch**-3).to(u.Msun*u.Mpc**-3) #Msun/Mpc^3
        k = np.logspace(-2,2,128)*u.Mpc**-1
        #Use rho_m or rho_cb depending on mnu
        if self.cosmo_code == 'camb':
            Pk = self.PKint(0.,k.value)*u.Mpc**3
            rhoM = rho_crit*(self.camb_pars.omegam-self.camb_pars.omeganu)
        else:
            Pk = self.PKint(k.value,np.array([0.]),len(k),1,0)*u.Mpc**3
            rhoM = rho_crit*(self.cosmo.Omega0_m()-self.cosmo.Omega_nu)

        R = (3.0*self.M/(4.0*np.pi*rhoM))**(1.0/3.0)

        #Get the window of a configuration space tophat
        kvec = (np.tile(k,[R.size,1]).T)
        Pk = np.tile(Pk,[R.size,1]).T
        R = np.tile(R,[k.size,1])
        x = ((kvec*R).decompose()).value
        W = 3.0*(np.sin(x) - x*np.cos(x))/(x)**3 
        
        #Compute sigma(M)
        integrnd = Pk*W**2*kvec**2/(2.*np.pi**2)
        sigma = np.sqrt(np.trapz(integrnd,kvec[:,0],axis=0))
        
        return sigma
        
        
    @cached_property
    def dsigmaM_dM(self):
        '''
        Computes the derivative of sigma(M) with respect to M at target redshift
        '''
        sigmaint = log_interp1d(self.M,self.sigmaM,fill_value='extrapolate')
        Mminus = self.M/1.0001
        Mplus =  self.M*1.0001
        sigma_minus = sigmaint(Mminus.value)
        sigma_plus = sigmaint(Mplus.value)
        return (sigma_plus-sigma_minus)/(Mplus-Mminus)
        
        
    @cached_cosmo_property
    def dsigmaM_dM_z0(self):
        '''
        Computes the derivative of sigma(M) with respect to M at z=0
        '''
        sigmaint = log_interp1d(self.M,self.sigmaMz0,fill_value='extrapolate')
        Mminus = self.M/1.0001
        Mplus =  self.M*1.0001
        sigma_minus = sigmaint(Mminus.value)
        sigma_plus = sigmaint(Mplus.value)
        return (sigma_plus-sigma_minus)/(Mplus-Mminus)
    
    
    @cached_property
    def bofM(self):
        '''
        Halo bias as a function of mass (and scale, if fNL != 0).  
        '''
        # nonlinear overdensity
        dc = 1.686
        nu = dc/self.sigmaM
        
        bias = np.tile(getattr(bm,self.bias_model)(self,dc,nu),(self.k.size,1)).T
        Delta_b = 0.
        if self.f_NL != 0:
            #get the transfer function, depending on whether mnu = 0 or mnu > 0
            if self.cosmo_code == 'camb':
                if self.camb_pars.num_nu_massive != 0:
                    Tk = self.transfer_cb(self.k.value)
                else:
                    Tk = self.transfer_m(self.k.value)
                Om0 = self.camb_pars.omegam
            else:
                if self.cosmo.Omega_nu != 0:
                    Tk = self.transfer_cb(self.k.value)
                else:
                    Tk = self.transfer_m(self.k.value)
                Om0 = self.cosmo.Omega0_m()
            #Compute non-Gaussian correction Delta_b
            factor = self.f_NL*dc*                                      \
                      3.*Om0*(100.*self.hubble*(u.km/u.s/u.Mpc))**2./   \
                     (cu.c.to(u.km/u.s)**2.*self.k**2*(Tk/np.max(Tk))*self.Dgrowth(self.z))
            Delta_b = (bias-1.)*np.tile(factor,(self.nM,1))
            
        return bias + Delta_b
        
        
    @cached_property
    def c_NFW(self):
        '''
        concentration-mass relation for the NFW profile.
        Following Diemer & Joyce (2019)
        c = R_delta / r_s (the scale radius, not the sound horizon)
        '''
        #smaller sampling of M
        Mvec = ulogspace(self.Mmin,self.Mmax,256).value
        #fit parameters
        kappa = 0.42
        a0 = 2.37
        a1 = 1.74
        b0 = 3.39
        b1 = 1.82
        ca = 0.2
        #Compute the effective slope of the growth factor
        dz = self.z*0.001
        alpha_eff = -(np.log(self.Dgrowth(self.z+dz))-np.log(self.Dgrowth(self.z-dz)))/ \
                    (np.log(1.+self.z+dz)-np.log(1.+self.z-dz))
        #Compute the effective slope to the power spectrum (as function of M)
        fun_int = -2.*3.*self.M/self.sigmaM*self.dsigmaM_dM-3.
        neff = interp1d(np.log10(self.M.value),fun_int,fill_value='extrapolate',kind='linear')(np.log10(kappa*Mvec))
        #Quantities for c
        A = a0*(1.+a1*(neff+3))
        B = b0*(1.+b1*(neff+3))
        C = 1.-ca*(1.-alpha_eff)
        nu = 1.686/log_interp1d(self.M.value,self.sigmaM)(Mvec)
        arg = A/nu*(1.+nu**2/B)
        #Compute G(x), with x = r/r_s, and evaluate c
        x = np.logspace(-3,3,256)
        g = np.log(1+x)-x/(1.+x)

        c = np.zeros(len(Mvec))
        for iM in range(len(Mvec)):
            G = x/g**((5.+neff[iM])/6.)
            invG = log_interp1d(G,x,fill_value='extrapolate',kind='linear')
            c[iM] = C*invG(arg[iM])
            
        return log_interp1d(Mvec,c,fill_value='extrapolate',kind='cubic')(self.M.value)
        
        
    @cached_property
    def ft_NFW(self):
        '''
        Fourier transform of NFW profile, for computing one-halo term
        '''
        #Radii of the SO collapsed (assuming 200*rho_crit)
        Delta = 200.
        rho_crit = 2.77536627e11*(self.Msunh*self.Mpch**-3).to(u.Msun*u.Mpc**-3) #Msun/Mpc^3
        R_NFW = (3.*self.M/(4.*np.pi*Delta*rho_crit))**(1./3.)
        #get characteristic radius
        r_s = np.tile(R_NFW/self.c_NFW,(self.nk,1)).T
        #concentration to multiply with ki
        c = np.tile(self.c_NFW,(self.nk,1)).T
        gc = np.log(1+c)-c/(1.+c)
        #argument: k*rs
        ki = np.tile(self.k,(self.nM,1))
        x = ((ki*r_s).decompose()).value        
        si_x, ci_x = sici(x)
        si_cx, ci_cx = sici((1.+c)*x)
        u_km = (np.cos(x)*(ci_cx - ci_x) +
                  np.sin(x)*(si_cx - si_x) - np.sin(c*x)/((1.+c)*x))
        return u_km/gc
        
        
    @cached_property
    def bavg(self):
        '''
        Average luminosity-weighted bias for the given cosmology and line
        model.  ASSUMED TO BE WEIGHTED LINERALY BY MASS FOR 'LF' MODELS
        
        Includes the effect of f_NL (inherited from bofM)
        '''
        #Apply dNL correction if model_type = TOY
        if self.model_type == 'TOY':
            dc = 1.686
            Delta_b = 0.
            b_line = self.model_par['bmean']*np.ones(self.nk)
            if self.f_NL != 0:
                #get the transfer function, depending on whether mnu = 0 or mnu > 0
                if self.cosmo_code == 'camb':
                    if self.camb_pars.num_nu_massive != 0:
                        Tk = self.transfer_cb(self.k.value)
                    else:
                        Tk = self.transfer_m(self.k.value)
                    Om0 = self.camb_pars.omegam
                else:
                    if self.cosmo.Omega_nu != 0:
                        Tk = self.transfer_cb(self.k.value)
                    else:
                        Tk = self.transfer_m(self.k.value)
                    Om0 = self.cosmo.Omega0_m()
                #Compute non-Gaussian correction Delta_b
                factor = self.f_NL*dc*                                      \
                          3.*Om0*(100.*self.hubble*(u.km/u.s/u.Mpc))**2./   \
                         (cu.c.to(u.km/u.s)**2.*self.k**2*(Tk/np.max(Tk))*self.Dgrowth(self.z))
                Delta_b = (bias-1.)*np.tile(factor,(self.nM,1))
                b_line += Delta_b
        else:
            # Integrands for mass-averaging
            factor = np.tile(self.LofM*self.dndM,(self.nk,1)).T
            itgrnd1 = self.bofM*factor
            itgrnd2 = factor
            
            b_line = np.trapz(itgrnd1,self.M,axis=0) / np.trapz(itgrnd2,self.M,axis=0)
        
        return b_line 
    
    
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
            kaiser = (1.+self.f_eff/self.bavg*self.mui_grid**2.)**2. #already squared
            
            if self.FoG_damp == 'Lorentzian':
                FoG = (1.+0.5*(self.k_par*self.sigma_NL).decompose()**2.)**-2.
            elif self.FoG_damp == 'Gaussian':
                FoG = np.exp(-((self.k_par*self.sigma_NL)**2.)
                        .decompose()) 
            else:
                raise ValueError('Only Lorentzian or Gaussian damping terms for FoG')
                
            return FoG*kaiser
        else:
            return np.ones(self.Pm.shape)

        
    @cached_property
    def Pm(self):
        '''
        Matter power spectrum from the interpolator computed by camb. 
        '''
        if self.cosmo_code == 'camb':
            return self.PKint(self.z,self.ki_grid.value)*u.Mpc**3
        else:
            Pkvec = self.PKint(self.k.value,np.array([self.z]),self.nk,1,0)*u.Mpc**3
            return np.tile(Pkvec,(self.nmu,1))
                
    
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
                L2bar = L2bar*np.exp(2.*(alpha**-2-alpha**-1)
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
            return self.model_par['Pshot']
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
                Mass_Dep = self.LofM*self.dndM
                itgrnd = np.tile(Mass_Dep,(self.k.size,1)).T*self.ft_NFW*self.bofM
                wt = self.CLT*np.trapz(itgrnd,self.M,axis=0)*self.fduty
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
            if self.model_type=='LF':
                print("One halo term only available for ML models")
                return np.zeros(self.Pm.shape)*self.Pshot.unit
            else:
                Mass_Dep = self.LofM**2.*self.dndM
                itgrnd = np.tile(Mass_Dep,(self.nk,1)).T*self.ft_NFW**2.
                            
                # Special case for Tony Li model- scatter does not preserve LCO
                if self.model_name=='TonyLi':
                    alpha = self.model_par['alpha']
                    sig_SFR = self.model_par['sig_SFR']
                    itgrnd = itgrnd*np.exp((2.*alpha**-2-alpha**-1)
                                        *sig_SFR**2*np.log(10)**2)
                wt = np.trapz(itgrnd,self.M,axis=0)*self.fduty
                return np.tile(self.CLT**2.*wt,(self.nmu,1))
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
                 
                 
    #############################################
    #############################################
    ### Voxel Intensity Distribution Functions ##
    #############################################
    #############################################
    
    ##################
    # Intensity bins #
    ##################
    @cached_vid_property
    def Tedge(self):
        '''
        Edges of intensity bins. Uses linearly spaced bins if do_fast_VID=True,
        logarithmically spaced if do_fast=False
        '''
        if self.do_fast_VID:
            Te = ulinspace(self.Tmin_VID,self.Tmax_VID,self.nT+1)
        else:
            Te = ulogspace(self.Tmin_VID,self.Tmax_VID,self.nT+1)
            
        if self.subtract_VID_mean:
            return Te-self.Tmean
        else:
            return Te
            
        
    @cached_vid_property
    def T(self):
        '''
        Centers of intensity bins
        '''
        return vt.binedge_to_binctr(self.Tedge)
        
        
    @cached_vid_property
    def dT(self):
        '''
        Widths of intensity bins
        '''
        return np.diff(self.Tedge)
        
        
    ######################################### 
    # Number count probability distribution #
    #########################################
    @cached_vid_property
    def Nbar(self):
        '''
        Mean number of galaxies per voxel
        '''
        return self.nbar*self.Vvox
        
        
    @cached_vid_property
    def Ngal(self):
        '''
        Vector of galaxy number counts, from 0 to self.Ngal_max
        '''
        return np.array(range(0,self.Ngal_max+1))
        

    @cached_vid_property
    def sigma_G(self):
        '''
        rms of fluctuations in a voxel (Gaussian window)
        '''
        if self.do_sigma_G:
            #kvalues, and kpar, kperp. Power spectrum in observed redshift
            k = np.logspace(-2,2,128)*u.Mpc**-1
            ki,mui = np.meshgrid(k,self.mu)
            if self.cosmo_code == 'camb':
                Pk = self.PKint(self.z,ki.value)*u.Mpc**3
            else:
                Pkvec = self.PKint(k.value,np.array([self.z]),len(k),1,0)*u.Mpc**3
                Pk = np.tile(Pkvec,(self.nmu,1))
            
            kpar = ki*mui
            kperp = ki*np.sqrt(1.-mui**2.)
            
            #Gaussian window for voxel -> FT
            Wkpar2 = np.exp(-((kpar*self.sigma_par)**2).decompose())
            Wkperp2 = np.exp(-((kperp*self.sigma_perp)**2).decompose())
            Wk2 = Wkpar2*Wkperp2
            
            #Compute sigma_G
            if self.f_NL == 0:
                bias = self.bavg[-1]
            else:
                bias = interp1d(self.k,self.bavg,kind='linear',bounds_error=False,fill_value=[self.bavg[0],self.bavg[-1]])
            integrnd = bias**2*Pk*Wk2*ki**2/(4.*np.pi**2)
            integrnd_mu = np.trapz(integrnd,self.mu,axis=0)
            sigma = np.sqrt(np.trapz(integrnd_mu,ki[0,:]))
            
            return sigma
        else:
            return self.sigma_G_input
    
    
    @cached_vid_property
    def PofN(self):
        '''
        Probability of a voxel containing N galaxies.  Uses the lognormal +
        Poisson model from Breysse et al. 2017
        '''
        # PDF of galaxy density field mu
        logMuMin = np.log10(self.Nbar)-20*self.sigma_G
        logMuMax = np.log10(self.Nbar)+5*self.sigma_G
        mu = np.logspace(logMuMin.value,logMuMax.value,10**4)
        mu2,Ngal2 = np.meshgrid(mu,self.Ngal) # Keep arrays for fast integrals
        Pln = vt.lognormal_Pmu(mu2,self.Nbar,self.sigma_G)

        P_poiss = poisson.pmf(Ngal2,mu2)
                
        return np.trapz(P_poiss*Pln,mu)
        
        
    ###################
    # Intensity PDF's #
    ###################
    @cached_vid_property
    def XLT(self):
        '''
        Constant relating total luminosity in a voxel to its observed
        intensity.  Equal to CLT/Vvox
        '''
        return self.CLT/self.Vvox
        
    
    @cached_vid_property
    def P1(self):
        '''
        Probability of observing a given intensity in a voxel which contains
        exactly one emitter
        '''
        # Compute dndL at L's equivalent to T bins        
        if self.model_type == 'ML':
            dndL_T = interp1d(self.L,self.dndL,bounds_error=False,fill_value='extrapolate')
            if self.subtract_VID_mean:
                LL = ((self.T+self.Tmean)/self.XLT).to(u.Lsun)
            else:
                LL = (self.T/self.XLT).to(u.Lsun)
            dndL = dndL_T(LL.value)*self.dndL.unit
            PT1 = dndL/(self.nbar*self.XLT)
        else:
            dndL_T = lambda L: getattr(lf,self.model_name)(L, self.model_par)
            if self.subtract_VID_mean:
                return dndL_T((self.T+self.Tmean)/self.XLT)/(self.nbar*self.XLT)
            else:
                return dndL_T(self.T/self.XLT)/(self.nbar*self.XLT)
        return PT1
        
        
    @cached_vid_property
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
            fP1 = fft(self.P1)*self.dT
            # FT of PDF should be dimensionless, but the fft function removes
            # the unit from P1
            fP1 = ((fP1*self.P1.unit).decompose()).value 
            
            fPT_N = np.zeros((self.Ngal_max+1,self.T.size),dtype=complex)

            for ii in range(1,self.Ngal_max+1):
                fPT_N[ii,:] = fP1**(ii)*self.PofN[ii]
            
            fPT = fPT_N.sum(axis=0)
            
            # Errors in fft's leave a small imaginary part, remove for output
            return (ifft(fPT)/self.dT).real
            
        else:
            P_N = np.zeros([self.Ngal_max,self.T.size])*self.P1.unit
            P_N[0,:] = self.P1
            
            for ii in range(1,self.Ngal_max):
                P_N[ii,:] = vt.conv_parallel(self.T,P_N[ii-1,:],
                                            self.T,self.P1,self.T)
            
            PT = np.zeros(self.T.size)

            for ii in range(0,self.Ngal_max):
                PT = PT+P_N[ii,:]*self.PofN[ii+1]
                
            return PT
            
            
    @cached_vid_property
    def PT_zero(self):
        '''
        P(T) contains a delta function at T=0 from voxels which contain zero 
        sources.  Delta functions are difficult to include naturally in
        arrays, so we model it separately here.  This quantity will need to be
        taken into account for any integrals over P(T) which cover T=0. (See
        the self.normalization function below)
        '''
        return self.PofN[0]
                
                
    @cached_vid_property
    def normalization(self):
        '''
        Outputs the value of integral(P(T)dT) including the spike at T=0.
        Used as a numerical check, should come out quite close to 1.0
        '''
        return np.trapz(self.PT,self.T)+self.PT_zero
        
        
    @cached_vid_property
    def PT_total(self):
        '''
        VID of the total observed temperature (signal + noise)
        '''
        return vt.PT_add_signal(self.PT,self.PDFnoise,self.T,self.dT,self.do_fast_VID)
        
        
    ########################
    # Predicted histograms #
    ########################
                                            
    @cached_vid_property
    def Tedge_i(self):
        '''
        Edges of histogram bins
        '''
        if self.linear_VID_bin:
            Te = ulinspace(-self.Tmax_VID,self.Tmax_VID,self.Nbin_hist+1)
        else:
            Te = ulogspace(self.Tmin_VID,self.Tmax_VID,self.Nbin_hist+1)
        
        if self.subtract_VID_mean:
            return Te-self.Tmean
        else:
            return Te
        
    @cached_vid_property
    def Ti(self):
        '''
        Centers of histogram bins
        '''
        return vt.binedge_to_binctr(self.Tedge_i)
        
    @cached_vid_property
    def Bi(self):
        '''
        Predicted number of voxels with a given binned temperature
        '''
        if self.subtract_VID_mean:
            return vt.pdf_to_histogram(self.T,self.PT,self.Tedge_i,self.Nvox,
                                        self.Tmean,self.PT_zero)
        else:
            return vt.pdf_to_histogram(self.T,self.PT,self.Tedge_i,self.Nvox,
                                        0.*self.Tmean.unit,self.PT_zero)
                                        
    @cached_vid_property
    def Bi_total(self):
        '''
        Predicted number of voxels with a given binned temperature
        (including also the contribution of noise. Therefore, the probability
        at T=0 is PofN(0)*Pnoise(0))
        '''
        if self.subtract_VID_mean:
            return vt.pdf_to_histogram(self.T,self.PT_total,self.Tedge_i,self.Nvox,
                            self.Tmean,self.PT_zero*2./((2.*np.pi)**0.5*self.sigma_N))
        else:
            return vt.pdf_to_histogram(self.T,self.PT_total,self.Tedge_i,self.Nvox,
                            0.*self.Tmean.unit,self.PT_zero*2./((2.*np.pi)**0.5*self.sigma_N))
                                        
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
            PL = log_interp1d(self.L.value,self.dndL.value)(Lgal.value)*dL
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

        
    ########################################################################
    # Method for updating input parameters and resetting cached properties #
    ########################################################################
    def update(self, **new_params):
        # Check if params dict contains valid parameters
        check_invalid_params(new_params,self._default_params)
        
        # If model_type or model_name is updated, check if model_name is valid
        if ('model_type' in new_params) and ('model_name' in new_params):
            check_model(new_params['model_type'],new_params['model_name'])
        elif 'model_type' in new_params:
            check_model(new_params['model_type'],self.model_name)
        elif 'model_name' in new_params:
            check_model(self.model_type,new_params['model_name'])
            
        #if bias_model is updated, check if bias_model is valid
        if 'bias_model' in new_params:
            check_bias_model(new_params['bias_model'])
        
        #if hmf_model is updated, check if hmf_model is valid
        if 'hmf_model' in new_params:
            check_halo_mass_function_model(new_params['hmf_model'])
    
    
        #List of observable parameters:
        obs_params = ['Tsys_NEFD','Nfeeds','beam_FWHM','Delta_nu','dnu',
                      'tobs','Omega_field','Nfield']
                      
        vid_params = ['Tmin_VID','Tmax_VID','nT','do_fast_VID','Ngal_max',
                       'Nbin_hist','subtract_VID_mean','linear_VID_bin',
                       'do_sigma_G','sigma_G_input']
        
        # Clear cached properties so they can be updated. If only obs changes,
        #   only update cached obs and vid properties. If only vid changes,
        #   update cached vid properties. Otherwise, cached normal properties
        #   always will be updated, and cosmo, only if needed
        if all(item in obs_params for item in new_params.keys()):
            for attribute in self._update_obs_list:
                delattr(self,attribute)
            self._update_obs_list = []
            for attribute in self._update_vid_list:
                delattr(self,attribute)
            self._update_vid_list = []
            #if smooth Pk, needs to be recomputed, since Pk changes
            if self.smooth:
                Pklist = ['Pk','Pk_0','Pk_2','Pk_4']
                for attribute in Pklist:
                    if attribute in self._update_list:
                        delattr(self,attribute)
                        self._update_list.remove(attribute)
        elif all(item in vid_params for item in new_params.keys()):
            for attribute in self._update_vid_list:
                delattr(self,attribute)
            self._update_vid_list = []
        else:
            for attribute in self._update_obs_list:
                delattr(self,attribute)
            self._update_obs_list = []
            for attribute in self._update_vid_list:
                delattr(self,attribute)
            self._update_vid_list = []
            for attribute in self._update_list:
                delattr(self,attribute)
            self._update_list = []

        # Clear cached cosmo properties only if needed, and update camb_pars
        if 'cosmo_input_camb' in new_params and self.cosmo_code == 'camb':
            if len(list(new_params['cosmo_input_camb'].keys())) == 1 and \
               list(new_params['cosmo_input_camb'].keys())[0] == 'f_NL':
                if 'f_NL' in self._update_cosmo_list:
                    delattr(self,'f_NL')
                    self._update_cosmo_list.remove('f_NL')
                    self.cosmo_input_camb['f_NL'] = new_params['cosmo_input_camb']['f_NL']
            else:
                for attribute in self._update_cosmo_list:
                    delattr(self,attribute)
                self._update_cosmo_list = []
                
                for key in new_params['cosmo_input_camb']:
                    self.cosmo_input_camb[key] = new_params['cosmo_input_camb'][key]
                
                self.camb_pars = camb.set_params(H0=self.cosmo_input_camb['H0'], cosmomc_theta=self.cosmo_input_camb['cosmomc_theta'],
                     ombh2=self.cosmo_input_camb['ombh2'], omch2=self.cosmo_input_camb['omch2'], omk=self.cosmo_input_camb['omk'],
                     neutrino_hierarchy=self.cosmo_input_camb['neutrino_hierarchy'], 
                     num_massive_neutrinos=self.cosmo_input_camb['num_massive_neutrinos'],
                     mnu=self.cosmo_input_camb['mnu'], nnu=self.cosmo_input_camb['nnu'], YHe=self.cosmo_input_camb['YHe'], 
                     meffsterile=self.cosmo_input_camb['meffsterile'], 
                     standard_neutrino_neff=self.cosmo_input_camb['standard_neutrino_neff'], 
                     TCMB=self.cosmo_input_camb['TCMB'], tau=self.cosmo_input_camb['tau'], 
                     deltazrei=self.cosmo_input_camb['deltazrei'], 
                     bbn_predictor=self.cosmo_input_camb['bbn_predictor'], 
                     theta_H0_range=self.cosmo_input_camb['theta_H0_range'],
                     w=self.cosmo_input_camb['w'],wa=self.cosmo_input_camb['wa'],cs2=self.cosmo_input_camb['cs2'], 
                     dark_energy_model=self.cosmo_input_camb['dark_energy_model'],
                     As=self.cosmo_input_camb['As'], ns=self.cosmo_input_camb['ns'], 
                     nrun=self.cosmo_input_camb['nrun'], nrunrun=self.cosmo_input_camb['nrunrun'], 
                     r=self.cosmo_input_camb['r'], nt=self.cosmo_input_camb['nt'], ntrun=self.cosmo_input_camb['ntrun'], 
                     pivot_scalar=self.cosmo_input_camb['pivot_scalar'], 
                     pivot_tensor=self.cosmo_input_camb['pivot_tensor'], 
                     parameterization=self.cosmo_input_camb['parameterization'],
                     halofit_version=self.cosmo_input_camb['halofit_version'])
            del new_params['cosmo_input_camb']
                     
        elif 'cosmo_input_class' in new_params and self.cosmo_code == 'class':
            if new_params['cosmo_input_class'].keys() == ['f_NL']:
                if len(list(new_params['cosmo_input_class'].keys())) == 1 and \
                   list(new_params['cosmo_input_class'].keys())[0] == 'f_NL':
                    delattr(self,'f_NL')
                    self._update_cosmo_list.remove('f_NL')
                    self.cosmo_input_class['f_NL'] = new_params['cosmo_input_class']['f_NL']
            else:
                for attribute in self._update_cosmo_list:
                    delattr(self,attribute)
                self._update_cosmo_list = []
                
                self.class_pars = merge_dicts([self.class_pars,new_params['cosmo_input_class']])
                
            del new_params['cosmo_input_class']
                 
        #If the mass range changes, but cosmo_input_camb/class doesn't, sigmaM related functions
        #   at z=0 (otherwise, need to change always when z changes, no cosmo_propery)
        #   need to be updated 
        if ('Mmin' in new_params) or ('Mmax' in new_params):
            if 'sigmaMz0' in self._update_cosmo_list:
                delattr(self,'sigmaMz0')
                self._update_cosmo_list.remove('sigmaMz0')
            if 'dsigmaM_dM_z0' in self._update_cosmo_list:
                delattr(self,'dsigmaM_dM_z0')
                self._update_cosmo_list.remove('dsigmaM_dM_z0')
                 
        #Avoid cosmo_code as an update
        if 'cosmo_code' in new_params:
            print("Please, use a new lim() run if you want to use other Boltzmann code")
            del new_params['cosmo_code']
            
        #update parameters
        for key in new_params:
            setattr(self, key, new_params[key])
                 
            
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
        
