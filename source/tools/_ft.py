import numpy as np
import pyfftlog as fftlog

def ft_log_mu(y,x,mu,q=0,kxopt=1,kx=1,tdir=1):
    ar = y*x**(1./2-q)
    
    dlnx = np.log(x[1])-np.log(x[0])
    dlogx = np.log10(x[1])-np.log10(x[0])
    
    N = len(x)
    Nc = (N+1)/2.
    
    kx,xsave = fftlog.fhti(N,mu,dlnx,q,kx,kxopt)
    ak = fftlog.fht(ar.copy(),xsave,tdir)
    
    logxc = (np.log10(x.min())+np.log10(x.max()))/2.
    logkc = np.log10(kx)-logxc
    klog = 10**(logkc+(np.arange(1,N+1)-Nc)*dlogx)
    
    fy = ak*np.sqrt(np.pi/2.)*klog**(-1./2-q)
    
    return klog,fy

def ft_log_cos(y,x,q=0,kxopt=1,kx=1,tdir=1):
    return ft_log_mu(y,x,mu=-1./2,q=q,kxopt=kxopt,kx=kx,tdir=tdir)

def ft_log_sin(y,x,q=0,kxopt=1,kx=1,tdir=1):
    return ft_log_mu(y,x,mu=1./2,q=q,kxopt=kxopt,kx=kx,tdir=tdir)

def ft_log(y,x,q=0,kxopt=1,kx=1):
    k_rr,fy_rr = ft_log_cos(y.real,x,q=q,kxopt=kxopt,kx=kx)
    k_ir,fy_ir = ft_log_sin(y.real,x,q=q,kxopt=kxopt,kx=kx)
    k = (k_rr+k_ir)/2.
    fy_real = fy_rr
    fy_imag = -fy_ir
    return k,fy_real+1j*fy_imag

def ift_log(fy,k,q=0,kxopt=1,kx=1):
    x_rr,y_rr = ft_log_cos(fy.real,k,q=q,kxopt=kxopt,kx=kx,tdir=1)
    x_ii,y_ii = ft_log_sin(fy.imag,k,q=q,kxopt=kxopt,kx=kx,tdir=1)
    x = (x_rr+x_ii)/2.
    y_real = (y_rr-y_ii)/np.pi
    return k,y_real

def ft(h,T):
    dT = T[1]-T[0]
    return np.fft.fftshift(np.fft.fft(np.fft.ifftshift(h)))*dT

def ift(f,k):
    dk = k[1]-k[0]
    Nbin = len(k)
    dT = 2*np.pi/(dk*Nbin)
    return np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(f)))/dT

def ft2(H,T,T2=None):
    if T2 is None:
        T2 = T
    dT = T[1]-T[0]
    dT2 = T2[1]-T2[0]
    return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(H)))*dT*dT2

def ift2(f,k1,k2):
    dk1 = k1[1]-k1[0]
    dk2 = k2[1]-k2[0]
    
    Nbin = len(k1)
    
    dT1 = 2*np.pi/(dk1*Nbin)
    dT2 = 2*np.pi/(dk2*Nbin)
    
    F = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(f)))/(dT1*dT2)
    if hasattr(f,'unit'):
        F = F*f.unit
    return F

def Ttok(T):
    dT = T[1]-T[0]
    Nbin = T.size
    kf = np.fft.fftfreq(Nbin)
    kf = np.append(kf[int((Nbin+1)/2):],kf[0:int((Nbin+1)/2)])
    k = kf*2*np.pi/dT
    return k