'''
Base module for generating models of line intensity maps
'''

import numpy as np
import inspect
import astropy.units as u
import astropy.constants as cu
from scipy.interpolate import interp1d
from scipy.special import sici


from hmf import MassFunction

from _utils import cached_property,get_default_params,check_params,ulogspace
from _utils import check_model
import luminosity_functions as lf
import mass_luminosity as ml

# hmf parameters. Need wide mass range with enough points for interpolation
hmf_logMmin = 7.
hmf_logMmax = 16.
hmf_dlog10m = 0.05

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
    assigned following a mass function computed with Steven Murray's hmf
    package.
    
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
    cosmo_model:    Either an astropy FlatLambdaCDM object or the string for
                    one of the default astropy cosmologies.  Defines
                    cosmological parameters. (Default = 'Planck15')

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
                 cosmo_model='Planck15',
                 model_type='LF',
                 model_name='SchCut', 
                 model_par={'phistar':8.7e-11*u.Lsun**-1*u.Mpc**-3,
                 'Lstar':2.1e6*u.Lsun,'alpha':-1.87,'Lmin':500*u.Lsun},
                 nu=115*u.GHz,
                 nuObs=30*u.GHz,
                 Mmin=1e9*u.Msun,
                 Mmax=1e15*u.Msun,
                 nM=5000,
                 hmf_model='Tinker08',
                 Lmin=100*u.Lsun,
                 Lmax=1e8*u.Lsun,
                 nL=5000,
                 kmin = 1e-2*u.Mpc**-1,
                 kmax = 10.*u.Mpc**-1,
                 nk = 100,
                 sigma_scatter=0.,
                 fduty=1.,
                 do_onehalo=False,
                 do_Jysr=False):
        

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
    
    ####################
    # Define 1/h units #
    ####################
    @cached_property
    def hubble(self):
        '''
        Normalized hubble parameter (H0.value/100). Used for converting to
        1/h units.
        '''
        return (self.h.cosmo.H0.to(u.km/(u.Mpc*u.s))).value/100
    
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
        return self.nu/self.nuObs-1.
    
    @cached_property
    def H(self):
        '''
        Hubble parameter at target redshift
        '''
        return self.h.cosmo.H(self.z)
        
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
        # Make sure masses fall within bounds defined for hmf
        logMmin_h = np.log10((self.Mmin.to(self.Msunh)).value)
        logMmax_h = np.log10((self.Mmax.to(self.Msunh)).value)
        
        if logMmin_h<hmf_logMmin:
            self.h.update(Mmin=logMmin_h/2.)
        elif logMmax_h>hmf_logMmax:
            self.h.update(Mmax=logMmax_h*2.)
        
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
        return ulogspace(self.kmin,self.kmax,self.nk+1)
    
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
            MofL = (interp1d(self.LofM,self.M,bounds_error=False,fill_value=0.)
                    (self.L)*self.M.unit)
            # Mass function at these masses
            dndM_MofL = interp1d(self.M,self.dndM,bounds_error=False,
                            fill_value=0.)(MofL)*self.dndM.unit
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
            bias_par = {'A':1.,'b':1.,'Mcut_min':self.Mmin,'Mcut_max':self.Mmax}
            L = getattr(ml,'MassPow')(self.M,bias_par,self.z)
        else:
            L = getattr(ml,self.model_name)(self.M,self.model_par,self.z)
            
        return L
        
    ########################################
    # Mass function, bias, and NFW profile #
    ########################################
    @cached_property
    def h(self):
        '''
        Initialzes hmf MassFunction() model.  Note that, unlike other cached
        properties, h is NOT cleared when update() is called.  Properties of h
        are updated using the hmf built-in update methods.  This speeds up 
        calculations where the astrophysical model is updated without changing
        the underlying cosmology
        '''
        return MassFunction(cosmo_model=self.cosmo_model,Mmin=hmf_logMmin,
                                Mmax=hmf_logMmax,dlog10m=hmf_dlog10m,z=self.z,
                                hmf_model=self.hmf_model)
        
    @cached_property
    def dndM(self):
        '''
        Halo mass function, note the need to convert from 1/h units in the
        output of hmf
        
        Interpolation done in log space
        '''
        logMh = np.log10((self.M.to(self.Msunh)).value)
        d = (10**interp1d(np.log10(self.h.m),np.log10(self.h.dndm))(logMh)
                *self.Mpch**-3*self.Msunh**-1)
        # Convert from 1/h units
        #d = (10.**logd)*self.Mpch**-3*self.Msunh**-1
        
        return d.to(u.Mpc**-3*u.Msun**-1)
        
    @cached_property
    def sigmaM(self):
        '''
        Mass variance at targed redshift, computed using hmf
        '''
        Mh = (self.M.to(self.Msunh)).value
        return interp1d(self.h.m,self.h.sigma)(Mh)
        
    
    @cached_property
    def bofM(self):
        '''
        Halo bias as a function of mass.  Currently always uses the Tinker
        et al. 2010 fitting function
        
        TODO:
        Add fitting functions for other hmf_models
        '''
        
        # nonlinear overdensity
        dc = self.h.delta_c
        nu = dc/self.sigmaM
        
        # Parameters of bias fit
        y = np.log10(200.)
        A = 1. + 0.24*y*np.exp(-(4./y)**4.)
        a = 0.44*y - 0.88
        B = 0.183
        b = 1.5
        C = 0.019 + 0.107*y + 0.19*np.exp(-(4./y)**4.)
        c = 2.4
        
        #return 1.- A*nu**a/(nu**a+dc**a) + B*nu**b + C*nu**c
        return 1+(nu**2-1)/dc
        
    @cached_property
    def ft_NFW(self):
        '''
        Fourier transform of NFW profile, for computing one-halo term
        '''
        [ki,Mi] = np.meshgrid(self.k,self.M)
        # Wechsler et al. 2002 cocentration fit
        a_c =0.1*np.log10((Mi.to(u.Msun)).value)-0.9
        a_c[a_c<0.1] = 0.1
        con = (4.1/(a_c*(1.+self.z))).value
        f = np.log(1.+con)-con/(1.+con)
        Delta = 200
        rhobar = (self.h.mean_density0*self.Msunh/self.Mpch**3).to(u.Msun/u.Mpc**3)
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
        '''
        
        # Integrands for mass-averaging
        itgrnd1 = self.LofM*self.bofM*self.dndM
        itgrnd2 = self.LofM*self.dndM
        
        return np.trapz(itgrnd1,self.M) / np.trapz(itgrnd2,self.M)
    
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
    def Pm(self):
        '''
        Matter power spectrum computed from hmf. Uses camb if camb is
        installed, uses the EH transfer function if not.
        '''
        kh = (self.k.to(self.Mpch**-1)).value
        P = interp1d(self.h.k,self.h.power)(kh)*self.Mpch**3
        
        return P.to(u.Mpc**3)
    
    @cached_property
    def Tmean(self):
        '''
        Sky-averaged brightness temperature at nuObs from target line.  Has
        two cases for 'LF' and 'ML' models
        '''
        if self.model_type=='LF':
            itgrnd = self.L*self.dndL
            Lbar = np.trapz(itgrnd,self.L)
        else:
            itgrnd = self.LofM*self.dndM
            Lbar = np.trapz(itgrnd,self.M)*self.fduty
            # Special case for Tony Li model- scatter does not preserve LCO
            if self.model_name=='TonyLi':
                alpha = self.model_par['alpha']
                sig_SFR = self.model_par['sig_SFR']
                Lbar = Lbar*np.exp((alpha**-2-alpha**-1)
                                    *sig_SFR**2*np.log(10)**2/2.)
            
        return self.CLT*Lbar
        
    @cached_property
    def Pshot(self):
        '''
        Shot noise amplitude for target line at frequency nuObs.  Has two
        cases for 'LF' and 'ML' models
        '''
        if self.model_type=='LF':
            itgrnd = self.L**2*self.dndL
            L2bar = np.trapz(itgrnd,self.L)
        else:
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
            
        return self.CLT**2*L2bar
        
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
                Mass_Dep = self.LofM*self.bofM*self.dndM
                itgrnd = (np.tile(Mass_Dep,[self.k.size,1]).transpose()
                            *self.ft_NFW)
                wt = self.CLT*np.trapz(itgrnd,self.M,axis=0)
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
            return self.CLT**2.*np.trapz(itgrnd,self.M,axis=0)
        else:
            return np.zeros(self.k.size)*self.Pshot.unit
            
            
    
    @cached_property    
    def Pk_clust(self):
        '''
        Clustering power spectrum of target line, i.e. power spectrum without
        shot noise.
        '''
        return self.Pk_twohalo+self.Pk_onehalo
    
    @cached_property    
    def Pk_shot(self):
        '''
        Shot-noise power spectrum of target line, i.e. power spectrum without
        clustering, or Pshot*ones(k.size)
        '''
        return self.Pshot*np.ones(self.k.size)
    
    @cached_property    
    def Pk(self):
        '''
        Full line power spectrum including both clustering and shot noise
        '''
        return self.Pk_clust+self.Pk_shot
        
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
            if attribute!='h':
                delattr(self,attribute)
        self._update_list = []
        
        # Set new parameter values
        for key in new_params:
            setattr(self, key, new_params[key])
        
        # Update hmf if cosmological model has changed
        if 'cosmo_model' in new_params:
            self.h.update(cosmo_model=new_params['cosmo_model'])
        elif 'hmf_model' in new_params:
            self.h.update(hmf_model=new_params['hmf_model'])
        elif 'nuObs' in new_params:
            self.h.update(z=self.z)
        elif 'nu' in new_params:
            self.h.update(z=self.z)
            
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
        
