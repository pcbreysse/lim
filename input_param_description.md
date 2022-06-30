# INPUT PARAMETERS

Here we briefly describe all the input parameters that can be set for lim, specifying the default parameters.



- **cosmo_code**: Whether to use class or camb (default: 'camb')
    
- **cosmo_input_camb**: Dictionary to read and feed to camb
    
- **cosmo_input_class**: Dictionary to read and feed to class

- **model_type**: Either 'LF' for a luminosity function model or 'ML' for a mass-luminosity model.  Any other value will raise an error.  Note that some outputs are only available for one model_type. (Default = 'LF')
    
- **model_name**: Name of line emission model.  Must be the name of a function defined in luminosity_functions.py (for model_name='LF') or mass_luminosity.py (for model_name='ML'). (Default = 'SchCut')
                    
- **model_par**: Dictionary containing the parameters of the chosen model (Default = Parameters of Breysse et al. 2017 CO model). Check mass_luminosity.py for 'ML' and luminosity_functions.py for 'LF' for information of the parameters for each model
                    
- **hmf_model**: Fitting function for the halo model using Pylians. To choose among 'ST, 'Tinker', 'Crocce', 'Jenkins', 'Warren', 'Watson', 'Watson_FOF', 'Angulo', (Default: 'ST').
                    
- **bias_model**: Fitting function for the bias model. To choose among 'Mo96', 'Jing98', 'ST99', 'SMT01', 'Seljak04', 'Seljak04_cosmo', 'Mandelbaum05', 'Tinker05', 'Tinker10', 'Manera10'
                    
- **bias_par**: A dictionary to pass non-standard values for the parameters of each bias model (look at bias_fitting_functions.py to see the parameters needed)
                    
- **nu**: Rest frame emission frequency of target line (Default = 115 GHz, i.e. CO(1-0))
                    
- **nuObs**: Observing frequency, defines target redshift (Default = 30 GHz, i.e. z=2.8 for CO)
                    
- **Mmin**: Minimum mass of line-emitting halo. (Default = 10^9 Msun)
    
- **Mmax**: Maximum mass of line emitting halo.  Rarely a physical parameter, but necessary to define high-mass cutoffs for mass function integrals (Default = 10^15 Msun)
                    
- **nM**: Number of halo mass points (Default = 5000)
    
- **Lmin**: Minimum luminosity for luminosity function calculations (Default = 100 Lsun)
                    
- **Lmax**: Maximum luminosity for luminosity function calculations (Default = 10^8 Lsun)
                    
- **nL**: Number of luminosity points (Default = 5000)
    
- **kmin**: Minimum wavenumber for power spectrum computations (Default = 10^-2 Mpc^-1)
                    
- **kmax**: Maximum wavenumber for power sepctrum computations (Default = 10 Mpc^-1)
    
- **nk**: Number of wavenumber points (Default = 100)
    
- **k_kind**: Whether you want k vector to be binned in linear or log space (options: 'linear','log'; Default:'log')
    
- **sigma_scatter**: Width of log-scatter in mass-luminosity relation, defined as the width of a Gaussian distribution in log10(L) which preserves the overall mean luminosity.  See Li et al.(2015) for more information. (Default = 0.0)
                    
- **fduty**: Duty cycle for line emission, as defined in Pullen et al. 2012 (Default = 1.0)
                    
- **do_onehalo**: Bool, if True power spectra are computed with one-halo term included (Default = False)
                    
- **do_Jysr**: Bool, if True quantities are output in Jy/sr units rather than brightness temperature (Default = False)
                    
- **do_RSD**: Bool, if True power spectrum includes RSD (Default:False)
    
- **sigma_NL**: Scale of Nonlinearities (Default: 7 Mpc)
    
- **nmu**: number of mu bins
    
- **FoG_damp**: damping term for Fingers of God (Default:'Lorentzian', other option 'Gaussian')
    
- **smooth**: smoothed power spectrum, convoluted with beam/channel (Default: False)
                    
- **do_conv_Wkmin**: Convolve the power spectrum with Wkmin instead of using a exponential suppression. Only relevant if smooth==True. (Default = False) Assumes a cylindrical volume
                    
- **nonlinear**: Using the non linear matter power spectrum in PKint (from halofit) (Boolean, default = False)
                    
- **Tmin_VID**: Minimum temperature to compute the temperature PDF (default: 1e-2 uK)
    
- **Tmax_VID**: Maximum temperature to compute the temperature PDF (default: 1e3 uK)
    
- **nT**: Number of points in temperature to compute the PDF (default: 1e5) (If using do_fast_VID, may require a very high number)
    
- **do_fast_VID**: Using FFTs to convolve the temperature PDF and compute the VID faster (Boolean, default: True)
                    
- **Ngal_max**: Maximum value for the galaxies in a voxel (default: 100)
    
- **Nbin_hist**: Number of bins to compute the VID histogram from the PDF (default=101)
    
- **subtract_VID_mean**:  Remove the mean from the VID measurements (default=False)
    
- **linear_VID_bin**: Using a linear sampling for the VID bins in the histogram (Boolean, default=False, which results in log binning)
                    
- **do_sigma_G**: Compute the variance of galaxies in a voxel to get PofN according to the power spectrum. (Boolean, default=True)
                    
- **sigma_G_input**: Value of the standard deviation of galaxy number in a voxel (Only relevant if do_sigma_G = False, default= 1.6)
                    
- **dndL_Lcut**: Cut in the luminosity function (at low L) if computed from L(M), useful for numerical performance if scatter is too big. (Only relevant if model_type='ML', default = 0 Lsun)

- **Tsys**: Instrument system temperature (Default = 40 K)
    
- **Nfeeds**: Number of feeds (Default = 19)
    
- **beam_FWHM**: Beam full width at half maximum (Default = 4.1")
    
- **Delta_nu**: Total frequency range covered by instrument (Default = 8 GHz)
    
- **dnu**: Width of a single frequency channel (Default = 15.6 MHz)
    
- **tobs**: Observing time on a single field (Default = 6000 hr)
    
- **Omega_field**: Solid angle covered by a single field (Default = 2.25 deg^2)    
    
- **Nfield**: Number of fields observed (Default = 1)
    
- **N_FG_par**: Multiplicative factor in the volume window for kmin_par to account for foregrounds. Default = 1, No foregrounds (only volume effects)
                    
- **N_FG_perp**: Multiplicative factor in the volume window for kmin_perp to account for foregrounds. Default = 1, No foregrounds (only volume effects)
                    
- **do_FG_wedge**: Apply foreground wedge removal. Default = False
    
- **a_FG**: Constant superhorizon buffer for foregrounds. Default = 0
    
- **b_FG**: Foreground parameter accounting for antenna chromaticity. Default = 0 
