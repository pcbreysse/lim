'''
Some example models from the literature.  Can be called by, e.g.,

m = LineModel(*TonyLi_Model)
'''

import astropy.units as u
from astropy.cosmology import FlatLambdaCDM

# Model from Li et al. (2015)
TonyLi_par = dict(alpha=1.17,beta=0.21,dMF=1.0,BehrooziFile='sfr_release.dat',
					Mcut_min=1e9*u.Msun,Mcut_max=1e15*u.Msun,sig_SFR=0.3)
TonyLi_cosmo = FlatLambdaCDM(Om0=.286,Ob0=.047,H0=70)
TonyLi_nuObs = 32*u.GHz
TonyLi_Model = dict(cosmo_model=TonyLi_cosmo,model_type='ML',model_name='TonyLi',
					model_par=TonyLi_par,nuObs=TonyLi_nuObs,sigma_scatter=0.3)