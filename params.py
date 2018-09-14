import astropy.units as u

default_par = dict(cosmo_model = 'Planck15',   # Input cosmological model
              model_type = 'LF',    # Type of line model, either 'LF' or 'ML'
              model_par = {'phistar':8.7e-11*u.Lsun**-1*u.Mpc**-3,
                           'Lstar':2.1e6*u.Lsun,
                           'alpha':-1.87,
                           'Lmin':500*u.Lsun},    # Model parameters
              nu = 115*u.GHz,    # Rest frequency of target line
              nuObs = 30*u.GHz,    # Central frequency of observation
              Mmin = 1e9*u.Msun,    # Minimum mass for line emission
              Mmax = 1e15*u.Msun,    # Maximum mass for line emission
              nM = 5000,    # Number of points to compute mass functions
              hmf_model = 'Tinker08',    # Mass function model. Choose from hmf
              nL = 5000,    # Number of points to compute luminosity functions
              kmin = 1e-2/u.Mpc,    # Minimum wavenumber for power spectra
              kmax = 7/u.Mpc,    # Maximum wavenumber for power spectra
              nk = 100,    # Number of wavenumber points for power spectra
              sigma_scatter = 0.,    # Scatter in mass-luminosity relation
              fduty = 1.,    # Fraction of halos emitting at any given time
              Tmin_VID = 1e-2*u.uK,    # Minimum intensity for VID
              Tmax_VID = 1000*u.uK,    # Maximum intensity for VID
              nT = 10**5,    # Number of intensity points for VID
              do_fast_VID = True, # Bool, do VID with FFT's if True, brute-
                                 # force integration if False
              sigma_G = 1.6,    # Gaussian variance parameter for VID
              Ngal_max = 100,    # Max sources/voxel for VID integrals
              Nbin_hist = 101,   # Number of bins to predict voxel counts
              subtract_VID_mean = False,    # Bool to use absolute or relative
                                           # intensity for VID
              linear_VID_bin = False,    # Output predicted histogram in linear
                                        # or logarithmically spaced bins
              do_onehalo = False,    # Bool to include one-halo contributions
              do_Jysr = False,    # Compute quantities in brightness temp if
                                 # False, Jy/sr if True
                                 
              # PARAMETERS USED FOR Line_Obs MODELS
              Tsys = 40*u.K,    # System temperature
              Nfeeds = 19,    # Number of detectors
              beam_FWHM = 4.1*u.arcmin,     # Beam full width at half max
              Delta_nu = 8*u.GHz,    # Total frequency bandwidth
              dnu = 15.6*u.MHz,    # Width of single frequency channel
              tobs = 6000*u.hr,    # Observing time per field
              Omega_field = 2.25*u.deg**2,     # Solid angle of single field
              Nfield = 1,    # Number of fields to be observed
              
              # PARAMETERS USED FOR LIMLAM SIMULATIONS
              catalogue_file = 
                  'limlam_mocker/catalogues/default_catalogue.npz',
                  # Location of peak-patch catalog file
              map_output_file = 'limlam_output.npz' # Output file location
              )

# Tony Li model and COMAP1 CURRENTLY WITHOUT SCATTER
TonyLi_PhI = dict(cosmo_model = 'Planck15',
                  model_type = 'ML',
                  model_name = 'TonyLi',
                  model_par = {'alpha':1.17,'beta':0.21,'dMF':1.0,
                               'BehrooziFile':'sfr_release.dat','sig_SFR':0.3},
                  nu = 115*u.GHz,
                  nuObs = 30*u.GHz,
                  Mmin = 5e10*u.Msun, # Set high to compare to incomplete sims
                  Mmax = 1e15*u.Msun,                    
                  nM = 5000,
                  nL = 5000,
                  kmin = 1e-2/u.Mpc,
                  kmax = 7./u.Mpc,
                  nk = 100,
                  sigma_scatter = 0.3,
                  Tsys = 40*u.K,
                  Nfeeds = 19,
                  beam_FWHM = 4.1*u.arcmin, # Set small to see shot noise in sim
                  Delta_nu = 1*u.GHz,
                  dnu = 15.6*u.MHz,
                  tobs = 1500*u.hr,
                  Omega_field = 0.5*u.deg**2, # Set large to see low-k in sims
                  Nfield = 4,
                  catalogue_file = '../../PeakPatch_Runs/COMAP_fullvolume/sim1.npz'
                  )
      
              
                                 
              
