'''
Miscellaneous utilities for LIM code
'''
import numpy as np
from astropy.units.quantity import Quantity
from astropy.cosmology.core import FlatLambdaCDM
import inspect
import astropy.units as u

import luminosity_functions as lf
import mass_luminosity as ml
import bias_fitting_functions as bm

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as colors

from scipy.interpolate import interp1d

class cached_property(object):
    """
    From github.com/Django, who wrote a much better version of this than
    the one I had previously.
    
    Decorator that converts a self.func with a single self argument into a
    property cached on the instance.
    """
    def __init__(self, func):
        self.func = func

    def __get__(self, instance, type=None):
        if instance is None:
            return self
        
        # ADDED THIS CODE TO LIST PROPERTY FOR UPDATING
        instance._update_list.append(self.func.__name__)
        
        res = instance.__dict__[self.func.__name__] = self.func(instance)
        return res
        
def get_default_params(func):
    '''
    Gets the default parameters of a function as input to check_params. Output
    is a dictionary of parameter names and values. If the function has a
    "self" argument it is removed from the dictionary.
    '''
    
    args = inspect.getargspec(func)
    
    param_names = args.args
    if 'self' in param_names:
        param_names.remove('self')
    
    default_values = args.defaults
    
    default_params = dict(zip(param_names,default_values))

    return default_params
    
        
def check_params(input_params, default_params):
    '''
    Check input parameter values to ensure that they have the same type and
    unit as the required inputs
    '''
    
    for key in input_params.keys():
        # Check if input is a valid parameter
        if key not in default_params.keys():
            raise AttributeError(key+" is not a valid parameter")
        
        input_value = input_params[key]
        default_value = default_params[key]
        
        # Check if input has the correct type
        if type(input_value)!=type(default_value):
            # Some inputs can have multiple types
            if key=='cosmo_model':
                if type(input_value)==FlatLambdaCDM:
                    pass
                else:
                    raise(TypeError(
                      "Parameter cosmo_model must be a str or FlatLambdaCDM"))
            elif key=='scatter_seed':
                if type(input_value)==int or type(input_value)==float:
                    pass
                
            elif type(default_value)==Quantity:
                raise TypeError("Parameter "+key+
                        " must be an astropy quantity")
            else:
                raise TypeError("Parameter "+key+" must be a "+
                                    str(type(default_value)))
            
        # If input is a quantity, check if it has the correct dimension
        elif (type(default_value)==Quantity and not
                 input_value.unit.is_equivalent(default_value.unit)):
            
            # Tmin/Tmax may be in either uK or Jy/sr depending on do_Jysr     
            if key=='Tmin' or key=='Tmax':
                if (input_params['do_Jysr'] and 
                   input_value.unit.is_equivalent(u.Jy/u.sr)):
                    pass
                else:
                    raise TypeError("Parameter "+key+
                                " must have units equivalent to "
                                +str(default_value.unit))
                                
        # Special requirements for certain parameters
        elif (key=='model_type' and not 
                (input_value=='ML' or input_value=='LF' or input_value=='TOY')):
            # model_type can only be ML or LF
            raise ValueError("model_type must be either 'ML' or 'LF' ot 'TOY' ")
            
            
def check_model(model_type,model_name):
    '''
    Check if model given by model_name exists in the given model_type
    '''
    if model_type=='ML' and not hasattr(ml,model_name):
        if hasattr(lf,model_name):
            raise ValueError(model_name+" not found in mass_luminosity.py."+
                    " Set model_type='LF' to use "+model_name)
        else:
            raise ValueError(model_name+
                    " not found in mass_luminosity.py")
    elif model_type=='LF' and not hasattr(lf,model_name):
        if hasattr(ml,model_name):
            raise ValueError(model_name+
                    " not found in luminosity_functions.py."+
                    " Set model_type='ML' to use "+model_name)
        else:
            raise ValueError(model_name+
                    " not found in luminosity_functions.py")
            
def check_bias_model(bias_name):
    '''
    Check if model given by model_name exists in the given model_type
    '''
    if not hasattr(bm,bias_name):
        raise ValueError(bias_name+
                    " not found in bias_fitting_functions.py")

                                
def ulogspace(xmin,xmax,nx):
    '''
    Computes logarithmically-spaced numpy array between xmin and xmax with nx
    points.  This function calls the usual np.loglog but takes the linear
    values of xmin and xmax (where np.loglog requires the log of those limits)
    and allows the limits to have astropy units.  The output array will be a
    quantity with the same units as xmin and xmax
    '''

    return np.logspace(np.log10(xmin.value),np.log10(xmax.value),nx)*xmin.unit

def ulinspace(xmin,xmax,nx):
    '''
    Computes linearly-spaced numpy array between xmin and xmax with nx
    points.  This function allows the limits to have astropy units. 
    The output array will be a quantity with the same units as xmin and xmax
    '''

    return np.linspace(xmin.value,xmax.value,nx)*xmin.unit


def log_interp1d(xx, yy, kind='linear',bounds_error=False,fill_value=0.):
    try:
        logx = np.log10(xx.value)
    except:
        logx = np.log10(xx)
    try:
        logy = np.log10(yy.value)
    except:
        logy = np.log10(yy)
    lin_interp = interp1d(logx, logy, kind=kind,bounds_error=bounds_error,fill_value=fill_value)
    
    log_interp = lambda zz: np.power(10.0, lin_interp(np.log10(zz)))

    return log_interp
