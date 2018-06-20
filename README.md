# lim

lim is a python application designed to analytically compute various statistics of line intensity maps using a wide variety of models.  This code is a work in progress, so it may change significantly and there may be undetected bugs.

### Prerequisites

lim requires several packages which should be familiar to astronomers working with python, including numpy, scipy, and astropy.  It also makes substantial use of Steven Murray's [hmf](https://www.github.com/steven-murray/hmf) package, which can be installed along with its dependencies with the command

```
pip install hmf
```

Finally, lim uses the hmf matter power spectrum, which uses the python camb wrapper if available, and the Eisenstein-Hu transfer function if not.

### Quickstart

In the folder containing the lim functions, you can quickly get the default CO power spectrum by running in an interpreter

```
from lim import LineModel
m = LineModel()
m.Pk
```

All parameters have default values, which can be changed either when creating the model or using the built-in update() method.  For example, to change the observing frequency from the default you could either run

```
m = LineModel(nuObs=15*u.GHz)
m.z
```

or

```
m = LineModel()
m.update(nuObs=15*u.GHz)
m.z
```

Instrumental effects are included within the line_obs.LineObs() class, which is a subclass of LineModel().  LineObs() allows the specification of instrument parameters such as system temperature, survey area, and resolution, and it includes methods for predicting quantities such as errors on the power spectrum and signal-to-noise ratios.  LineObs() shares the same update() method as LineModel().

The class vid.VID(), which is a subclass of LineObs, contains modules for computing the one-point statistics of a model.  This module currently uses the lognormal-based formalism of Breysse et al. (2017).

The LineModel() class and its subclasses also include a reset() method, which updates the class back to the parameters it had when it was first called.  This is useful when making temporary changes to one or more input parameters.  For example,

```
m = LineModel(nuObs=15*u.GHz)
m.update(nuObs=30*u.GHz)
m.reset()
```

will produce the same results as

```
m = LineModel(nuObs=15*u.GHz)
m.update(nuObs=30*u.GHz)
m.update(nuObs=15*u.GHz)
```

## DocTests

To quickly check several of the parameters, lim includes doctests.  In a terminal, simply run

```
python lim.py
```

Note that the expected power spectra for the doctests were computed assuming the camb module is installed.  If you do not have camb installed, i.e. if

```
import camb
```
gives an error, hmf will use the EH transfer function and the doctest on Pk will give incorrect numbers



## Authors

* **Patrick C. Breysse**

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* Code based on matlab routines originally developed with Ely Kovetz
* Thanks to Dongwoo Chung and George Stein for debugging help



