#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" v1s_math module

Utility math functions.

"""

import scipy as N
from numpy import *
from numpy.fft import fft2
import sys

def fastnorm(x):
    """ Fast Euclidean Norm (L2)

    This version should be faster than numpy.linalg.norm if 
    the dot function uses blas.

    Inputs:
      x -- numpy array

    Output:
      L2 norm from 1d representation of x
    
    """    
   
    
    x = array(x)
    xv = x.ravel() 
    
    return N.dot(xv, xv)**(1/2.)

def fastsvd(M):
    """ Fast Singular Value Decomposition
    
    Inputs:
      M -- 2d numpy array

    Outputs:
      U,S,V -- see scipy.linalg.svd    

    """
    
    h, w = M.shape
    
    # -- thin matrix
    if h >= w:
        # subspace of M'M
        U, S, V = N.linalg.svd(N.dot(M.T, M))
        U = N.dot(M, V.T)
        # normalize
        for i in xrange(w):
            S[i] = fastnorm(U[:,i])
            U[:,i] = U[:,i] / S[i]
            
    # -- fat matrix
    else:
        # subspace of MM'
        U, S, V = N.linalg.svd(N.dot(M, M.T))
        V = N.dot(U.T, M)
        # normalize
        for i in xrange(h):
            
            S[i] = fastnorm(V[i])
            V[i,:] = V[i] / S[i]
            
    return U, S, V

def gabor2d(gsw, gsh, gx0, gy0, wfreq, worient, wphase, shape):
    """ Generate a gabor 2d array
    
    Inputs:
      gsw -- standard deviation of the gaussian envelope (width)
      gsh -- standard deviation of the gaussian envelope (height)
      gx0 -- x indice of center of the gaussian envelope
      gy0 -- y indice of center of the gaussian envelope
      wfreq -- frequency of the 2d wave
      worient -- orientation of the 2d wave
      wphase -- phase of the 2d wave
      shape -- shape tuple (height, width)

    Outputs:
      gabor -- 2d gabor with zero-mean and unit-variance

    """
    
    height, width = shape
    y, x = N.mgrid[0:height, 0:width]
    
    
    X = x * N.cos(worient) * wfreq
    Y = y * N.sin(worient) * wfreq
    
    env = N.exp( -.5 * ( ((x-gx0)**2./gsw**2.) + ((y-gy0)**2./gsh**2.) ) )
    wave = N.exp( 1j*(2*N.pi*(X+Y) + wphase) )
    gabor = N.real(env * wave)
    
    gabor -= gabor.mean()
    gabor /= fastnorm(gabor)
    
    return gabor

# EVERYTHING BELOW THIS LINE ----------------------------------------------
# bparks (Brian Parks) 27 Aug 2009

def padzeros(mat, dim):
    """ Pad the given matrix with zeros to the given dimensions

    Inputs:
      mat -- (scipy) matrix to pad
      dim -- final dimensions of (scipy) matrix to return

    Outputs:
      newmat -- matrix of dimensions dim. the contents of mat are in the
            upper left-hand corner

    """

    h, w = dim
    newmat = N.zeros((h, w))

    oldh, oldw = mat.shape

    if h < oldh or w < oldw:
        print "Size mismatch: (", h, ", ", w, ") < (", oldh, ", ", oldw, ")"
        # sys.exit(1)

    for i in xrange(min(oldh,h)):
        for j in xrange(min(oldw,w)):
            newmat[i, j] = mat[i, j]

    return newmat

def circshift(mat, dir, shift):
    """ Shift the given matrix in the given direction by the given amount.
        Rows/columns get shifted around to the opposite side if they would
        otherwise "fall off" the end of the matrix.

        Horizontal movement is accomplished by postmultiplication of mat by
        an identity matrix shifted the same amount vertically.

        Vertical movement is accomplished by premultiplication of mat by an
        identity matrix shifted the same amount horizontally.

    Inputs:
      mat -- (scipy) matrix to shift
      dir -- direction (must be 'horiz' or 'vert')
      shift -- the amount to shift the matrix. no size limits are
            enforced, as the modulus is taken of the computed
            coordinate. negative values shift up or to the left; positive
            values shift down or to the right

    Outputs:
      newmat -- matrix shifted accordingly

    """

    h, w = mat.shape
    
    

    if dir == 'horiz':
        # identity must be w * w for resultant matrix to be h * w
        zeromat = N.zeros((w, w))

        for i in xrange(w):
            zeromat[i, (i+shift)%w] = 1

        newmat = dot(mat, zeromat)
        return newmat

    if dir == 'vert':
        # identity must be h * h for resultant matrix to be h * w
        zeromat = N.zeros((h, h))

        for i in xrange(h):
            zeromat[(i+shift)%h, i] = 1

        newmat = dot(zeromat, mat)
        return newmat

    return mat # no direction specified

def psf2otf(psf, dim):
    """ Based on the matlab function of the same name

    """

    h, w = psf.shape

    mat = padzeros(psf, dim)

    mat = circshift(mat, 'horiz', -(w/2))
    mat = circshift(mat, 'vert', -(h/2))

    otf = fft2(mat)

    return otf
