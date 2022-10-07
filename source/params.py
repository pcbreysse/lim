import astropy.units as u

default_par = dict(
              cosmo_code = 'camb', #choose between using camb or class
              cosmo_input_camb = dict(f_NL=0,H0=67.36,cosmomc_theta=None,ombh2=0.02237, omch2=0.12, 
                               omk=0.0, neutrino_hierarchy='degenerate', 
                               num_massive_neutrinos=3, mnu=0.06, nnu=3.046, 
                               YHe=None, meffsterile=0.0, standard_neutrino_neff=3.046, 
                               TCMB=2.7255, tau=None, deltazrei=None, bbn_predictor=None, 
                               theta_H0_range=[10, 100],w=-1.0, wa=0., cs2=1.0, 
                               dark_energy_model='ppf',As=2.1e-09, ns=0.9649, nrun=0, 
                               nrunrun=0.0, r=0.0, nt=None, ntrun=0.0, 
                               pivot_scalar=0.05, pivot_tensor=0.05,
                               parameterization=2,halofit_version='mead'),
              cosmo_input_class = dict(f_NL=0,H0=67.36,omega_b=0.02237, omega_cdm=0.12, 
                               A_s=2.1e-9,n_s=0.9649,
                               N_ncdm=3, m_ncdm='0.2,0.2,0.2', 
                               output='mPk,mTk'),
              model_type = 'LF',    # Type of line model, either 'LF' or 'ML'
              model_name='SchCut',
              model_par = {'phistar':9.6e-11*u.Lsun**-1*u.Mpc**-3,
                           'Lstar':2.1e6*u.Lsun,
                           'alpha':-1.87,
                           'Lmin':5000*u.Lsun},    # Model parameters
              nu = 115*u.GHz,    # Rest frequency of target line
              nuObs = 30*u.GHz,    # Central frequency of observation
              Mmin = 1e9*u.Msun,    # Minimum mass for line emission
              Mmax = 1e15*u.Msun,    # Maximum mass for line emission
              nM = 5000,    # Number of points to compute mass functions
              hmf_model = 'ST',    # Mass function model. Choose from hmf
              bias_model = 'ST99',   # halo bias model
              bias_par={}, #Otherwise, write a dict with the corresponding values
              Lmin = 10*u.Lsun,    # Minimum luminosity for dn/dL
              Lmax = 1e8*u.Lsun,    # Maximum luminosity for dn/dL
              nL = 5000,    # Number of points to compute luminosity functions
              kmin = 1e-2/u.Mpc,    # Minimum wavenumber for power spectra
              kmax = 10/u.Mpc,    # Maximum wavenumber for power spectra
              nk = 100,    # Number of wavenumber points for power spectra
              k_kind = 'log',   #Kind of sampling in k
              sigma_scatter = 0.,    # Scatter in mass-luminosity relation
              fduty = 1.,    # Fraction of halos emitting at any given time
              do_onehalo = False,    # Bool to include one-halo contributions
              do_Jysr = False,    # Compute quantities in brightness temp if
                                 # False, Jy/sr if True
              do_RSD = True,    #Bool to include RSD contributions
              sigma_NL = 7.*u.Mpc,   #Scale for non linear suppresion due to FoG
              nmu = 1000,           #list of mu (cos theta) for integration or anisotropic
              FoG_damp='Lorentzian', #Kind of damping for FoG
              smooth=False, #Smoothed power spectrum (convoluted with beam/channel)
              nonlinear=False, #Compute the non-linear power spectrum using HMcode
              Tmin_VID = 1e-2*u.uK,    # Minimum intensity for VID
              Tmax_VID = 1000*u.uK,    # Maximum intensity for VID
              nT = 10**5,    # Number of intensity points for VID
              do_fast_VID = True, # Bool, do VID with FFT's if True, brute-
                                 # force integration if False
              Ngal_max = 100,    # Max sources/voxel for VID integrals
              Nbin_hist = 101,   # Number of bins to predict voxel counts
              subtract_VID_mean = False,    # Bool to use absolute or relative
                                           # intensity for VID
              linear_VID_bin = False,    # Output predicted histogram in linear
                                        # or logarithmically spaced bins
              do_sigma_G = True,        # Compute theoretically the rms of fluctuations in a voxel 
                                        # default: True
              sigma_G_input = 1.6,      # input rms of fluctuations in voxel
              dndL_Lcut=0.*u.Lsun,      # Cut off at Lmin in dndL computed from L(M) to ease computations
              # PARAMETERS USED FOR Line_Obs MODELS
              Tsys_NEFD = 40*u.K,    # System temperature
              Nfeeds = 19,    # Number of detectors
              beam_FWHM = 4.1*u.arcmin,     # Beam full width at half max
              Delta_nu = 8*u.GHz,    # Total frequency bandwidth
              dnu = 15.6*u.MHz,    # Width of single frequency channel
              tobs = 6000*u.hr,    # Observing time per field
              Omega_field = 2.25*u.deg**2,     # Solid angle of single field
              Nfield = 1,    # Number of fields to be observed
              N_FG_par = 1, #Foreground multiplicative number to kmin_par in window
              N_FG_perp = 1, #Foreground multiplicative number to kmin_perp in window
              do_FG_wedge = False, #Apply foreground wedge removal
              a_FG = 0.*u.Mpc**-1, #Constat superhorizon buffer for foregrounds
              b_FG = 0., #Foreground parameter accounting for antenna chromaticity
              # PARAMETERS USED FOR LIMLAM SIMULATIONS
              catalogue_file = 
                  'limlam_mocker/catalogues/default_catalogue.npz',
                  # Location of peak-patch catalog file
              map_output_file = 'limlam_output.npz' # Output file location
              )
