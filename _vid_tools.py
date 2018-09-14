'''
Miscellaneous functions for VID and CVID calculations
'''

import numpy as np
import astropy.units as u
from scipy.integrate import quad
from scipy.interpolate import interp1d

# Parallelization functions for convolutions
from multiprocessing import Pool as ThreadPool
from functools import partial

def binedge_to_binctr(binedge):
    '''
    Outputs centers of histogram bins given their edges
    
    >>> Tedge = [1.,2.]*u.uK
    >>> print binedge_to_binctr(Tedge)
    [ 1.5] uK
    '''
    
    Nedge = binedge.size
    
    binctr = (binedge[0:Nedge-1]+binedge[1:Nedge])/2.
    
    return binctr
    
def binctr_to_binedge_linear(binctr):
    '''
    Outputs edges of histogram bins given their edges.  Will invert
    binedge_to_binctr if bins are linearly spaced.
    '''
    Nbin = binctr.size
    dx = binctr[1]-binctr[0]
    binedge =  np.append(binctr-dx/2.,binctr[Nbin-1]+dx/2.)
    
    if hasattr(binctr,'unit'):
        binedge = binedge.value*binctr.unit
    
    return binedge
    
def binctr_to_binedge_log(binctr):
    '''
    Outputs edges of histogram bins given their edges.  Will invert
    binedge_to_binctr if bins are logarithmically spaced.
    '''
    Nbin = binctr.size
    r = binctr[1]/binctr[0]
    binedge = np.append(2*binctr/(1+r),2*binctr[Nbin-1]/(1+1/r))
    
    if hasattr(binctr,'unit'):
        binedge = binedge.value*binctr.unit
    
    return binedge
    
    
def pdf_to_histogram(T,PT,Tedge,Nvox,Tmean_sub,PT_zero):
    '''
    Converts continuous probability distribution PT(T) to predicted histogram
    values for bin edges Tedge, assuming total number of voxels Nvox
    
    P(T) is assumed to be zero outside of the given range of T
    
    Bins are assumed to be small compared to the width of P(T), i.e.
    predicted number of voxels = P(T) * (bin width) * (Nvox)
    '''
    
    dT = np.diff(Tedge)
    Tbin = binedge_to_binctr(Tedge)
    PTi = interp1d(T,PT,bounds_error=False,fill_value=0)#(Tbin)
    
    h = np.zeros(Tbin.size)
    
    for ii in range(0,Tbin.size):
        h[ii] = quad(PTi,Tedge[ii].value,Tedge[ii+1].value)[0]*Nvox
        if Tedge[ii]<=-Tmean_sub<=Tedge[ii+1]:
            h[ii] = h[ii]+PT_zero
    
    return h
    
def conv_bruteforce(x1,y1,x2,y2,x):
    '''
    Brute-force numerical convolution between functions y1(x1) and y2(x2)
    computed at points x.  Used for slow VID computations.
    
    y1 and y2 are assumed to be zero outside the given range of x1 and x2
    '''
    
    # Interpolate input functions for integration
    y1f = interp1d(x1,y1,bounds_error=False,fill_value=0)
    y2f = interp1d(x2,y2,bounds_error=False,fill_value=0)
    
    xmin = min(np.append(x1,x2))
    
    Imax = x-xmin
    
    if Imax<=xmin:
        y = 0.
        
    else:
        #print x
        itgrnd = lambda xp: y1f(xp) * y2f(x-xp)
        y = quad(itgrnd, xmin, Imax)[0]
    return y
    
    
def conv_parallel(x1,y1,x2,y2,x):
    '''
    Parallelized version of conv_bruteforce
    '''
    
    # Remove units from quantities
    yunit = y1.unit**2.*x.unit # unit for final convolution
    
    x1 = x1.to(x.unit).value
    x2 = x2.to(x.unit).value
    y2 = y2.to(y1.unit).value
    
    y1 = y1.value
    x = x.value
    
    # Setup parallel pool
    pool = ThreadPool(4)
    
    # Compute convolution in parallel
    fun = partial(conv_bruteforce,x1,y1,x2,y2)
    y = np.array(pool.map(fun,x))
    
    # Close parallel pool
    pool.close()
    pool.join()
    
    # Add units back on 
    return y*yunit
    
def conv_series(x1,y1,x2,y2,x):
    '''
    Un-parallelized version of conv_bruteforce
    '''
    
    # Remove units from quantities
    yunit = y1.unit**2.*x.unit # unit for final convolution
    
    x1 = x1.to(x.unit).value
    x2 = x2.to(x.unit).value
    y2 = y2.to(y1.unit).value
    
    y1 = y1.value
    x = x.value
    y = np.zeros(x.size)
    
    for ii in range(0,x.size):
        y[ii] = conv_bruteforce(x1,y1,x2,y2,x[ii])
        
    return y*yunit
    
def lognormal_Pmu(mu,Nbar,sig_G):
    '''
    Lognormal probability distribution of mean galaxy counts mu, with width
    set by sig_G.  This function gives the PDF of the underlying lognormal
    density field, can be combined with a Poisson distribution to get a model
    for P(Ngal)
    '''
    Pln = (np.exp(-(np.log(mu/Nbar)+sig_G**2/2.)**2/(2*sig_G**2)) /
            (mu*np.sqrt(2.*np.pi*sig_G**2.)))
    return Pln
    
if __name__ == "__main__":
    import doctest

    doctest.testmod(optionflags=doctest.ELLIPSIS |
                    doctest.NORMALIZE_WHITESPACE)
