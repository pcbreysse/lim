'''
Primary module for interacting with the lim package
'''

import numpy as np
import params as params_file
from line_model import LineModel
from line_obs import LineObs
#from limlam import LimLam, set_sim_cosmo
from _utils import get_default_params


def lim(model_params='default_par',doObs=True,doSim=False,match_sim_cosmo=True):
    '''
    Base function for creating lim objects. Which object is generated depends
    on inputs doObs and doSim.
    
    INPUTS:
    model_params    String containing the name of a dict from params.py, or
                    dict object containing parameter names and values.
                    
    doObs           Bool, sets whether or not to inculde instrument parameters
    
    doSim           Bool, sets whether or not to run simulations
    
    match_sim_cosmo Bool, sets whether or not to change LineModel() cosmology
                    to match simulations
                    
    DOCTESTS:
    >>> m = lim()
    >>> m.hubble
    0.6774
    >>> m.z
    <Quantity 2.833...>
    >>> m.bavg
    <Quantity 1.983...>
    >>> m.Tmean
    <Quantity 1.373... uK>
    >>> m.Pshot
    <Quantity 828.8... Mpc3 uK2>
    >>> m.Nvox
    <Quantity 1387152.0>
    >>> m.sigma_N
    <Quantity 25.99... uK>
    >>> m.SNR
    <Quantity 13.7...>
    >>> m1 = lim(model_params='TonyLi_PhI',doSim=True)
    Input cosmological model does not match simulations
    Setting analytic cosmology to match simulation
    >>> m1.update(scatter_seed=1)
    >>> m1.Tmean_sim
    <Quantity 1.517... uK>
    >>> m1.Pshot_halos
    <Quantity 4214... Mpc3 uK2>
    
    '''
    
    # Input parameter values from file
    if type(model_params) is str:
        params = getattr(params_file,model_params)
    elif type(model_params) is dict:
        params = model_params
    else:
        raise ValueError('model_params must be a str or dict')
        
    par1 = remove_invalid_params(params,doObs,doSim)
    
    if doObs:
        if doSim:
            m = LimLam(**par1)
            if match_sim_cosmo and m._cosmo_flag:
                set_sim_cosmo(m)
                print('Setting analytic cosmology to match simulation')
            return m
        else:
            return LineObs(**par1)
    else:
        if doSim:
            print('Simulations require doObs=True')
        return LineModel(**par1)
        
def remove_invalid_params(params,doObs,doSim):
    '''
    Function to remove excess inputs from dictionary.  For example, LineModel
    does not use the inputs of LineObs, such as beam_FWHM, so we want to
    delete them from params before calling LineModel.
    '''

    y = params.copy() # Can't change size of dictionary in for loop, also
                      # removing parameters from params directly messes with
                      # autoreload
    
    x = get_default_params(LineModel.__init__)
    if doObs:
        x1 = get_default_params(LineObs.__init__)
        x.update(x1)
    if doSim:
        x2 = get_default_params(LimLam.__init__)
        x.update(x2)

    for key in params:
        if key not in x:
            y.pop(key)

    return y

############
# Doctests #
############

if __name__ == "__main__":
    import doctest

    doctest.testmod(optionflags=doctest.ELLIPSIS |
                    doctest.NORMALIZE_WHITESPACE)
