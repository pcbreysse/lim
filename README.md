# lim

lim is a python application designed to analytically compute various statistics of line intensity maps using a wide variety of models.  It also contains functions to generate simulated intensity maps from peak-patch simulations provided by George Stein.  This code is a work in progress, so it may change significantly and there may be undetected bugs.

Due to being a live code, there may be updates without annoucement as a new release, unless something really big changes. Please make sure to update the code regularly in case eventual bug have been fixed / calculations have been improved

### Changes from previous versions

- Option to use the python wrapper of class or camb. Note that the newest version of camb does not support python 2 any more. Then make sure that both class and camb versions correspond to the same python type (class wrapper can be compiled for python 3 since version 2.8).

- General speed up of the code:
    - Better managing of updates of the input parameters
    - Variance of fluctuations and halo mass function computed internally, without relying in pylians, which is no longer needed.
    - Precomputation of the linear matter power spectrum and other cosmological quantities for several redshifts to interpolate without the need of recomputing cosmology unless cosmological parameters are updated.
    
- VID located in line_model. Available for ML models with monotonically increasing L(M) relations. Added a function to add different signals to VID.

- Added option to work with the non-linear power spectrum

- In the case of massive neutrinos, work with quantities related with the cdm+b distribution (all matter but neutrinos)

- Implementation of concentration-mass relation from Diemer & Joyce (2019)

- New astrophysical models

- Possibility of applying foreground wedge removal or a more limitting volume window due to foregrounds
    
- Reorganization of the directories to organize the modules.

- Bug fixed on halo mass function computation (especifically regarding Tinker 2010)

- Default cosmological values from Planck 2018 TTTEEE + lensing

### Prerequisites

lim requires several packages which should be familiar to astronomers, including numpy, scipy, and astropy.  

Astropy units are used throughout this code to avoid unit conversion errors. To use the output of lim in any code which does not accept astropy units, simply replace output x with x.value.

Using the simulation functionality requires peak-patch catalogs.  One example catalog is included here, more can be obtained from George Stein (github.com/georgestein)

Finally, lim uses the python camb wrapper to compute all needed cosmological quantities. 

### Quickstart

After adding the lim folder to your python path, you can quickly get the default CO power spectrum by running in an interpreter

```
from lim import lim
m = lim()
m.Pk
```

Models are defined by dictionary objects in the params.py file.  The 'default_par' set of parameters is used by default, but any other set can be used by changing the 'model_params' input of lim().  For example, params.py contains another dict which defines parameters for the Li et al. (2015) CO emission model and the COMAP Phase I observation, which can be called with

```
m = lim(model_params='TonyLi_PhI')
```

You can also set parameters directly with a dictionary, for example

```
from params import TonyLi_PhI
m = lim(model_params = TonyLi_PhI)
```

All modules in lim use an update(), which allows parameter values to be changed after creating the model.  Most outputs are created as @cached_properties, which will update themselves using the new value after update() is called.  For example, to change the observing frequency of a survey you could run

```
m = lim()
m.update(nuObs=15*u.GHz)

```

The update() method is somewhat optimized, in that it will only rerun functions if required.  This speeds up update()'s which only change the line emission physics without altering the cosmology.

There is also a reset() method which will reset all input parameters back to the values they had when lim() was originally called.

### Examples

An ipython notebook fully commented is provided as an example. Following this notebook (LIM_PkFisher.ipynb) will get you familiar with the code (especially with the computation of the power spectrum multipoles and the corresponding covariance), and will allow you to reproduce the results appearing on the papers: arXiv:1907.10065 and arXiv:1907.10067. These papers (and the example), focus on the use of the multipoles of the LIM power spectrum to extract robust and optimal cosmological information, marginalizing over astrophysical uncertainties. 

### Modules

The lim.lim() function reads a dict of parameters and creates an object which computes desired quantities from those parameters.  The object created can come from one of several modules, depending on the other inputs to lim().  The base class is the line_model.LineModel() class, which models a signal on the sky independent of survey design.  This object can output power spectra and VID's for a desired model.  If doObs=True in lim(), the line_obs.LineObs() class is used, which is a subclass of LineModel that adds in functionality related to instrumental characteristics, such as noise curves.  If doSim=True, the limlam.LimLam() class is used, which further adds the ability to generate simulated maps and compute statistics from them.

### Line Emission Models

Models for line emission physics are defined in one of two ways: either with a formula for the luminosity function dn/dL or by a mass/luminosity relation L(M).  Which is used is set by the 'model_type' input, which is 'LF' for the former and 'ML' for the latter.  Specific models are defined in the luminosity_functions.py and mass_luminosity.py files respectively.  The model_name input should be a string containing the name of a function in one of these two files, and the model_par should be a dict of that model's parameters.  Custom models can easily be added by adding additional functions to the relevant file.

## DocTests

To quickly check several of the parameters, lim includes doctests.  In a terminal, simply run

```
python lim.py
```

Note that the expected power spectra for the doctests were computed assuming the camb module is installed.  If you do not have camb installed, i.e. if

```
import camb
```
gives an error, hmf will use the EH transfer function and the doctests may fail.


## Usage

When used, please refer to the github page and cite arXiv:1907.10067


## Authors

* **Patrick C. Breysse**
* **José Luis Bernal**

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* Code based on matlab routines originally developed with Ely Kovetz
* LimLam simulation code adapted from limlam_mocker code written by George Stein and Dongwoo Chung



