#!/usr/bin/env python
# coding: utf-8

import numpy as np
import tensorflow as tf
from scipy.signal import windows
from scipy.interpolate import interp1d

def fourier(t, y):

    # interpolation:
    interp = interp1d(t, y, kind='cubic')
    t = np.linspace(t[0], t[-1], t.size)
    y = interp(t)

    Nt = t.size
    dt = t[1] - t[0]

    # windowing:
    y *= windows.tukey(Nt, 0.065)

    # Fourier Transform:
    yTil = dt*np.fft.fft(y)
    f = np.fft.fftfreq(Nt, dt)

    yTil = np.fft.fftshift(yTil)
    f = np.fft.fftshift(f)
    
    return f, yTil


def inner(f, Sn, y, g=None):

    if type(g) == type(None): num = y*np.conj(y)
    else: num = y*np.conj(g)

    integ = 4*(num/Sn).real

    return np.trapz(integ, x=f, dx=(f[1]-f[0]))


def Mismatch(t, h1, h2):
    
    f, h1 = fourier(t, h1)
    f, h2 = fourier(t, h2)
    
    Sn = np.ones(f.size)

    N1 = inner(f, Sn, h1)
    N2 = inner(f, Sn, h2)
    
    I = [inner(f, Sn, h1, h2*np.exp(-1j*phi)) 
         for phi in np.linspace(0, 2*np.pi, 1000)]
    
    match = max(I)/np.sqrt(N1*N2)

    return 1 - match

