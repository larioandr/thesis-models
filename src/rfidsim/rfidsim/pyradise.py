"""
Module for RFID channel simulation.

Original version: https://github.com/severmore/pyradise/blob/dev/pyradise/channel.py

Author: Roman Ivanov <iromcorp@gmail.com>
"""
import numpy as np
from numpy import linalg as la
import scipy.special as special


def vectorize(fn):
    def wrapper(*args, **kw):
        return fn(*args, **kw)
    return np.vectorize(wrapper)


def dbm2w(value_dbm):
    return 10 ** (value_dbm / 10 - 3)


def w2dbm(value_watt):
    return 10 * np.log10(value_watt) + 30 if value_watt >= 1e-15 else -np.inf


def db2lin(value_db):
    return 10 ** (value_db / 10)


@vectorize
def lin2db(value_linear):
    return 10 * np.log10(value_linear) if value_linear >= 1e-15 else -np.inf


# noinspection PyUnusedLocal
def isotropic_rp(**kwargs):
    """
    Returns constant (1.0)
    :return:
    """
    return 1.0


# noinspection PyUnusedLocal
# @vectorize
def dipole_rp(*, azimuth, **kwargs):
    """
    Returns dipole directional gain
    :param azimuth:
    :return:
    """
    c = np.cos(azimuth)
    s = np.sin(azimuth)
    if c > 1e-9:
        return np.abs(np.cos(np.pi / 2 * s) / c)
    else:
        return 0.0


# TODO: check right formula, guess type of antenna element if considering not dipole
# noinspection PyUnusedLocal
# @vectorize
def array_dipole_rp(*, azimuth, n, **kwargs):
    """
    Returns dipole array directional gain
    :param azimuth:
    :param n:
    :return:
    """
    c = np.cos(azimuth)
    s = np.sin(azimuth)
    if c < 1e-9:
        return 0.
    if np.abs(s) < 1e-9:
        return 1.
    elif c > 1e-9:
        return np.abs(np.sin(np.pi / 2 * n * s) / np.sin(np.pi / 2 * s)) / n


# TODO: check right formula
# noinspection PyUnusedLocal
# @vectorize
def helix_rp(*, azimuth, n, **kwargs):
    """
    Returns helix antenna directional gain
    :param azimuth:
    :param n:
    :return:
    """
    c = np.cos(azimuth)
    s = np.sin(azimuth)
    if c > 1e-9:
        return np.abs(c * np.sin(np.pi / 2 * n * c) / np.sin(np.pi / 2 * c))
    else:
        return 0.0


# @vectorize
def _patch_rp_factor(azimuth, tilt, wavelen, width, length):
    s_a = np.sin(azimuth)
    c_a = np.cos(azimuth)
    s_t = np.sin(tilt)
    c_t = np.cos(tilt)
    kw = np.pi / wavelen * width
    kl = np.pi / wavelen * length
    # if c_t < 1e-9 or c_a < 1e-9:
    if c_a < 1e-9:
        return 0
    if np.abs(s_a) < 1e-9:
        return 1.
    elif np.abs(s_t) < 1e-9:
        return np.cos(kl * s_a)
    else:
        return np.sin(kw * s_a * s_t) / (kw * s_a * s_t) * np.cos(kl * s_a * c_t)


# @vectorize
def _patch_theta_rp(azimuth, tilt, wavelen, width, length):
    return _patch_rp_factor(azimuth, tilt, wavelen, width, length) * np.cos(tilt)


# @vectorize
def _patch_phi_rp(azimuth, tilt, wavelen, width, length):
    return -1 * _patch_rp_factor(azimuth, tilt, wavelen, width, length) * np.sin(tilt) * np.cos(azimuth)


# noinspection PyUnusedLocal
# @vectorize
def patch_rp(*, azimuth, tilt, wavelen, width, length, **kwargs):
    """
    Returns directional gain (in linear scale, 0..1)
    :param azimuth:
    :param tilt:
    :param wavelen:
    :param width:
    :param length:
    :return:
    """
    return ( np.abs(_patch_rp_factor(azimuth, tilt, wavelen, width, length)) *
            (np.cos(tilt) ** 2 + np.cos(azimuth) ** 2 * np.sin(tilt) ** 2) ** 0.5)


def _reflection_c_parallel(grazing_angle, permittivity, conductivity, wavelen):
    eta = permittivity - 60j * wavelen * conductivity
    c = np.cos(grazing_angle)
    return (eta - c ** 2) ** 0.5


def _reflection_c_perpendicular(grazing_angle, permittivity, conductivity, wavelen):
    eta = permittivity - 60j * wavelen * conductivity
    c = np.cos(grazing_angle)
    return (eta - c ** 2) ** 0.5 / eta


# noinspection PyUnusedLocal
def reflection_constant(*, grazing_angle, polarization, permittivity, conductivity, wavelen, **kwargs):
    return -1.0 + 0.j


# noinspection PyUnusedLocal
# @vectorize
def reflection(*, grazing_angle, polarization, permittivity, conductivity, wavelen, **kwargs):
    """
    Computes reflection coefficient from conducting surface with defined grazing angle and supported relative
    permittivity and conductivity of the surface. In order to set type of wave polarization, polarization
    parameter is specified. For parallel to surface polarized wave polarization should set to 1, for perpendicular -
    to 0; for circular - to 0.5. For different type of elliptic polarization use other value in the range of 0..1
    :param grazing_angle: an angle between normal to surface and wave vector
    :param polarization: determine the type of polarization of the grazing wave
    :param permittivity: the relative_permittivity of two media divided by the surface
    :param conductivity: the conductivity of the surface
    :param wavelen: the wave length of the grazing wave
    :return: the reflection coefficient for specified parameters and the given grazing angle
    """
    s = np.sin(grazing_angle)

    if polarization < 0 or polarization > 1:
        return float('nan')

    if polarization != 0:
        c_parallel = _reflection_c_parallel(grazing_angle, permittivity, conductivity, wavelen)
        reflection_parallel = (s - c_parallel) / (s + c_parallel)
    else:
        reflection_parallel = 0.j

    if polarization != 1:
        c_perpendicular = _reflection_c_perpendicular(grazing_angle, permittivity, conductivity, wavelen)
        reflection_perpendicular = (s - c_perpendicular) / (s + c_perpendicular)
    else:
        reflection_perpendicular = 0.j

    return polarization * reflection_parallel + (1 - polarization) * reflection_perpendicular


# @vectorize
def free_space_path_loss_2d(*, distance, tx_rp, rx_rp, tx_angle, rx_angle, tx_height, rx_height, wavelen, **kwargs):
    """
    Computes free space signal attenuation between the transmitter and the receiver in linear scale.
    :param distance: the distance between transmitter and receiver
    :param rx_angle: a mount angle of transmitter antenna
    :param tx_angle: a mount angle of receiver antenna
    :param tx_rp: a radiation pattern of the transmitter
    :param rx_rp: a radiation pattern of the receiver
    :param tx_height: a mount height of the transmitter
    :param rx_height: a mount height of the receiver
    :param wavelen: a wavelen of signal carrier
    :return: free space path loss in linear scale
    """
    # Ray geometry computation
    delta_height = np.abs(tx_height - rx_height)
    d0 = (delta_height ** 2 + distance ** 2) ** 0.5
    alpha0 = np.arctan(distance / delta_height)

    # Attenuation caused by radiation pattern
    g0 = (tx_rp(azimuth=alpha0 - tx_angle, tilt=np.pi/2, wavelen=wavelen, **kwargs) *
          rx_rp(azimuth=alpha0 - rx_angle, tilt=np.pi/2, wavelen=wavelen, **kwargs))

    k = wavelen / (4 * np.pi)
    return (k * g0 / d0) ** 2


def two_ray_path_loss(*, distance, time, speed, ground_reflection, tx_rp, rx_rp, tx_angle, rx_angle, tx_height,
                      rx_height, wavelen, **kwargs):
    """
    Computes free space signal attenuation between the transmitter and the receiver in linear scale.
    :param distance: the distance between transmitter and receiver
    :param time: current time
    :param speed: relative speed of the receiver
    :param ground_reflection: a function to compute a complex-valued reflection coefficient
    :param tx_rp: a radiation pattern of the transmitter
    :param rx_rp: a radiation pattern of the receiver
    :param rx_angle: a mount angle of transmitter antenna
    :param tx_angle: a mount angle of receiver antenna
    :param tx_height: a mount height of the transmitter
    :param rx_height: a mount height of the receiver
    :param wavelen: a wavelen of signal carrier
    :return: free space path loss in linear scale
    """
    # Ray geometry computation
    delta_height = np.abs(tx_height - rx_height)
    sigma_height = np.abs(tx_height + rx_height)
    d0 = (delta_height ** 2 + distance ** 2) ** 0.5
    d1 = (sigma_height ** 2 + distance ** 2) ** 0.5
    alpha0 = np.arctan(distance / delta_height)
    alpha1 = np.arctan(distance / sigma_height)

    # Attenuation caused by radiation pattern
    g0 = (tx_rp(azimuth=alpha0 - tx_angle, tilt=np.pi/2, wavelen=wavelen, **kwargs) *
          rx_rp(azimuth=alpha0 - rx_angle, tilt=np.pi/2, wavelen=wavelen, **kwargs))

    g1 = (tx_rp(azimuth=alpha1 - tx_angle, tilt=np.pi/2, wavelen=wavelen, **kwargs) *
          rx_rp(azimuth=alpha1 - rx_angle, tilt=np.pi/2, wavelen=wavelen, **kwargs))

    # Attenuation due to reflections (reflection coefficient) computation
    r1 = ground_reflection(grazing_angle=alpha1, wavelen=wavelen, **kwargs)

    k = 2 * np.pi / wavelen
    return (0.5 / k) ** 2 * np.absolute(   g0/d0*np.exp(-1j*k*(d0 + speed*time*np.sin(alpha0))) +
                                        g1*r1/d1*np.exp(-1j*k*(d1 + speed*time*np.sin(alpha1)))) ** 2


def two_ray_path_loss_3d(*, time, ground_reflection, wavelen,
                         tx_pos, tx_dir_theta, tx_dir_phi, tx_velocity, tx_rp,
                         rx_pos, rx_dir_theta, rx_dir_phi, rx_velocity, rx_rp, **kwargs):
    """
    Computes free space signal attenuation between the transmitter and the receiver in linear scale.
    :param wavelen: a wavelen of signal carrier
    :param time: Time passed from the start of reception
    :param ground_reflection: a function to compute a complex-valued reflection coefficient
    :param tx_velocity: the velocity of the transmitter
    :param tx_dir_phi: the vector pointed the direction with tilt angle equals 0 of the transmitter antenna.
    :param tx_dir_theta: the vector pointed the direction with azimuth angle equals 0 of the transmitter antenna.
    :param tx_pos: a current position of the transmitter.
    :param tx_rp: a radiation pattern of the transmitter
    :param rx_velocity: the velocity of the receiver
    :param rx_dir_phi: the vector pointed the direction with tilt angle equals 0 of the transmitter antenna.
    :param rx_dir_theta: the vector pointed the direction with azimuth angle equals 0 of the transmitter antenna.
    :param rx_pos: a current position of the receiver
    :param rx_rp: a radiation pattern of the receiver
    :return: free space path loss in linear scale
    """
    # LoS - Line-of-Sight, NLoS - Non-Line-of-Sight

    # Ray geometry computation
    ground_normal = np.array([0, 0, 1])
    rx_pos_refl = np.array([rx_pos[0], rx_pos[1], -rx_pos[2]])  # Reflect RX relatively the ground

    d0_vector = rx_pos - tx_pos            # LoS ray vector
    d1_vector = rx_pos_refl - tx_pos       # NLoS ray vector
    d0 = la.norm(d0_vector)                # LoS ray length
    d1 = la.norm(d1_vector)                # NLoS ray length
    d0_vector_tx_n = d0_vector / d0        # LoS ray vector normalized
    d0_vector_rx_n = -d0_vector_tx_n
    d1_vector_tx_n = d1_vector / d1        # NLoS ray vector normalized
    d1_vector_rx_n = np.array([-d1_vector_tx_n[0], -d1_vector_tx_n[1], d1_vector_tx_n[2]])

    # Azimuth and tilt angle computation for computation of attenuation
    # caused by deflection from polar direction
    tx_azimuth_0 = np.arccos(np.dot(d0_vector_tx_n, tx_dir_theta))
    rx_azimuth_0 = np.arccos(np.dot(d0_vector_rx_n, rx_dir_theta))
    tx_azimuth_1 = np.arccos(np.dot(d1_vector_tx_n, tx_dir_theta))
    rx_azimuth_1 = np.arccos(np.dot(d1_vector_rx_n, rx_dir_theta))

    tx_tilt_0 = np.arccos(np.dot(d0_vector_tx_n, tx_dir_phi))
    rx_tilt_0 = np.arccos(np.dot(d0_vector_rx_n, rx_dir_phi))
    tx_tilt_1 = np.arccos(np.dot(d1_vector_tx_n, tx_dir_phi))
    rx_tilt_1 = np.arccos(np.dot(d1_vector_rx_n, rx_dir_phi))

    # A grazing angle of NLoS ray for computation of reflection coefficient
    grazing_angle = np.arccos(-1*np.dot(d1_vector_rx_n, ground_normal))

    relative_velocity = rx_velocity - tx_velocity
    velocity_pr_0 = np.dot(d0_vector_tx_n, relative_velocity)
    velocity_pr_1 = np.dot(d1_vector_tx_n, relative_velocity)

    # Attenuation caused by radiation pattern
    g0 = (tx_rp(azimuth=tx_azimuth_0, tilt=tx_tilt_0, wavelen=wavelen, **kwargs) *
          rx_rp(azimuth=rx_azimuth_0, tilt=rx_tilt_0, wavelen=wavelen, **kwargs))

    g1 = (tx_rp(azimuth=tx_azimuth_1, tilt=tx_tilt_1, wavelen=wavelen, **kwargs) *
          rx_rp(azimuth=rx_azimuth_1, tilt=rx_tilt_1, wavelen=wavelen, **kwargs))

    # Attenuation due to reflections (reflection coefficient) computation
    r1 = ground_reflection(grazing_angle=grazing_angle, wavelen=wavelen, **kwargs)

    k = 2 * np.pi / wavelen
    return (0.5/k)**2 * np.absolute(   g0/d0*np.exp(-1j*k*(d0 - time * velocity_pr_0)) +
                                    r1*g1/d1*np.exp(-1j*k*(d1 - time * velocity_pr_1)))**2


def two_ray_path_loss_2d(*, distance, start_position, ground_reflection, tx_rp, rx_rp, tx_angle, rx_angle, tx_height,
                         rx_height, wavelen, **kwargs):
    # Ray geometry computation
    pass_distance = start_position - distance
    delta_height = np.abs(tx_height - rx_height)
    sigma_height = np.abs(tx_height + rx_height)
    d0 = (delta_height ** 2 + distance ** 2) ** 0.5
    d1 = (sigma_height ** 2 + distance ** 2) ** 0.5
    alpha0 = np.arctan(distance / delta_height)
    alpha1 = np.arctan(distance / sigma_height)

    # Attenuation caused by radiation pattern
    g0 = (tx_rp(azimuth=alpha0 - tx_angle, tilt=np.pi/2, wavelen=wavelen, **kwargs) *
          rx_rp(azimuth=alpha0 - rx_angle, tilt=np.pi/2, wavelen=wavelen, **kwargs))

    g1 = (tx_rp(azimuth=alpha1 - tx_angle, tilt=np.pi/2, wavelen=wavelen, **kwargs) *
          rx_rp(azimuth=alpha1 - rx_angle, tilt=np.pi/2, wavelen=wavelen, **kwargs))

    # Attenuation due to reflections (reflection coefficient) computation
    r1 = ground_reflection(grazing_angle=alpha1, wavelen=wavelen, **kwargs)

    k = 2 * np.pi / wavelen
    return (0.5 / k) ** 2 * np.absolute(   g0/d0*np.exp(-1j*k*(d0 + pass_distance*distance/d0)) +
                                        g1*r1/d1*np.exp(-1j*k*(d1 + pass_distance*distance/d1))) ** 2


# noinspection PyUnusedLocal
def signal2noise(*, rx_power, noise_power, **kwargs):
    """
    Computes Signal-to-Noise ratio. Input parameters are in logarithmic scale.
    :param rx_power:
    :param noise_power:
    :param kwargs:
    :return:
    """
    return db2lin(rx_power - noise_power)


# noinspection PyUnusedLocal
def sync_angle(*, snr, preamble_duration=9.3e-6, bandwidth=1.2e6, **kwargs):
    """
    Computes the angle of de-synchronisation.
    :param snr: an SNR of the received signal
    :param preamble_duration: the duration of PHY-preamble in seconds
    :param bandwidth: the bandwidth of the signal in herzs
    :param kwargs:
    :return: the angle of de-synchronisation
    """
    return (snr * preamble_duration * bandwidth) ** -0.5


# noinspection PyUnusedLocal
def snr_extended(*, snr, sync_phi=0, miller=1, symbol_duration=1.25e-6, bandwidth=1.2e6, **kwargs):
    """
    Computes the extended SNR for BER computation.
    :param snr: an SNR of the received signal
    :param sync_phi: the de-synchronization
    :param miller: the order of Miller encoding
    :param symbol_duration: the symbol duration in seconds
    :param bandwidth: the bandwidth of the signal in herzs
    :param kwargs:
    :return: the extended SNR for BER computation
    """
    return miller * snr * symbol_duration * bandwidth * np.cos(sync_phi) ** 2


# noinspection PyUnusedLocal
def ber_over_awgn(*, snr, **kwargs):
    """
    Computes BER in an additive white gaussian noise (AWGN) channel for Binary Phase Shift Keying (BPSK)
    :param snr: the extended SNR
    :return:
    """

    def q_function(x):
        return 0.5 - 0.5 * special.erf(x / 2 ** 0.5)

    t = q_function(snr ** 0.5)
    return 2 * t * (1 - t)


# noinspection PyUnusedLocal
def ber_over_rayleigh(*, snr, **kwargs):
    """
    Computes BER in the channel with Rayleigh fading for Binary Phase Shift Keying (BPSK)
    :param snr:
    :param kwargs:
    :return:
    """
    t = (1 + 2 / snr) ** 0.5
    return 0.5 - 1 / t + 2 / np.pi * np.arctan(t) / t

