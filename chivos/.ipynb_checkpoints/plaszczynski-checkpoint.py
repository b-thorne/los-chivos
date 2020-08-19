""" This module contains functions to calculate the Plaszczinsky
modified asymptotic estimator of the polarization fraction from
Stokes parameters in the presence of noise. See the reference
here for details, and for referenced equations and figures
in the docstrings:


"""
import numpy as np

__all__ = ['PMAS']

def _phi(q, u):
    """ Calculate :math:`\\phi`:
    """
    
    return np.arctan2(u, q)

def _theta(sq, su, rsqsu):
    """ Calculates rotation of the principal axes of the
    isoprobability ellipse. See Fig 1 and Eq 32 of reference.
    
    .. math::
        \theta = \frac{1}{2}\tan^{-1}\left(\frac{\rho \sigma_q\sigma_u}{\sigma_q^2 - \sigma_u^2}\right)
        
    Parameters
    ----------
    sq, su: ndarray
        :math:`\sigma_{q / u}`
    rsqsu: ndarray
        :math:`\rho \sigma_q\sigma_u`
        
    Returns
    -------
    ndarray
        :math:`\theta`.
        
    """
    return np.mod(0.5 * np.arctan2(2. * rsqsu, sq ** 2 - su ** 2), np.pi)

def _bias(sqp, sup, phi, theta):
    """ Calculates bias parameters in MAS estimator. See Eq 35 of
    reference.
    
    .. math::
        b_i = \sqrt{\sigma_u^{\prime 2} \cos^2(\phi_i - \theta) +
        \sigma_q^{\prime 2}\sin^2(\phi_i - \theta)}
        
    Parameters
    ----------
    sqp, sup: ndarray
        :math:`\sigma_{q / u}^\prime
    phi: ndarray
        :math:`\phi`
    theta: ndarray
        :math:`\theta`
        
    Returns
    -------
    ndarray
        :math:`b_i`.
    """
    return np.sqrt(sup ** 2 * np.cos(phi - theta) ** 2 + sqp ** 2 * np.sin(phi - theta) ** 2)

def _sigma_q_prime(sq, su, rsqsu, theta):
    """ Rotated values of the Stokes U noise. See Eq 33 of reference.
    
    .. math::
        \sigma_q^\prime = \sqrt{\sigma_q^2 \cos^2\theta + \sigma_u^2\sin^2\theta
        + \rho \sigma_q \sigma_u \sin 2\theta}
        
    Parameters
    ----------
    sq, su: ndarray
        :math:`\sigma_{\rm Q / U}`
    rsqsu: ndarray
        :math:`\rho \sigma_q \sigma_u`
    theta: ndarray
        :math:`\theta`
        
    Returns
    -------
    ndarray
        :math:`\sigma_q^\prime`.
    """
    return np.sqrt(sq ** 2 * np.cos(theta) ** 2 + su ** 2 * np.sin(theta) ** 2 + rsqsu * np.sin(2. * theta))

def _sigma_u_prime(sq, su, rsqsu, theta):
    """ Rotated values of the Stokes U noise. See Eq 33 of reference.
    
    .. math::
        \sigma_u^\prime = \sqrt{\sigma_q^2 \sin^2\theta + \sigma_u^2\cos^2\theta
        - \rho \sigma_q \sigma_u \sin 2\theta}
        
    Parameters
    ----------
    sq, su: ndarray
        :math:`\sigma_{\rm Q / U}`
    rsqsu: ndarray
        :math:`\rho \sigma_q \sigma_u`
    theta: ndarray
        :math:`\theta`
        
    Returns
    -------
    ndarray
        :math:`\sigma_u^\prime`.
    """
    return np.sqrt(sq ** 2 * np.sin(theta) ** 2 + su ** 2 * np.cos(theta) ** 2 - rsqsu * np.sin(2. * theta))

def _P(q, u):
    return np.sqrt(q ** 2 + u ** 2)

def PMAS(q, u, sq, su, rsqsu):
    """ General form of the modified asymptotic estimator (MAS).
    
    .. math::
        P_{\rm MAS} = p_i - b_i^2 \frac{1-e^{-p_i^2 / b_i^2}}{2 p_i}
    
    Parameters
    ----------
    t, q, u: ndarray
        :math:`T,~Q,~U`.
    sq, su, rsqsu: ndarray
        :math:`\sigma_q,~\sigma_u,~\rho\sigma_q\sigma_u`.
    
    Returns
    -------
    ndarray
        :math:`P_{\rm MAS}`.
    """
    P = _P(q, u)
    phi = _phi(q, u)
    theta = _theta(sq, su, rsqsu)
    sqp = _sigma_q_prime(sq, su, rsqsu, theta)
    sup = _sigma_u_prime(sq, su, rsqsu, theta)
    bias = _bias(sqp, sup, phi, theta)
    return P - bias ** 2 * (1. - np.exp(- P ** 2 / bias ** 2)) / 2. / P