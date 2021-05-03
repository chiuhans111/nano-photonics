import numpy as np


def angle_average(angles, weight=1):
    """
    計算角度(0~360)平均
    angles: ndarray
    weight: ndarray
    """
    angles = np.array(angles)*np.pi/180
    weight = np.array(weight)
    complex = np.exp(1j*angles) * weight
    average = np.average(complex) / np.average(weight)
    return np.arctan2(np.imag(average), np.real(average))*180/np.pi


def angle_normalize(angles):
    return (angles+180) % 360-180
