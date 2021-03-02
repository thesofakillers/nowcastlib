"""
Functions for computing metrics related to Geographical information science.
"""
import numpy as np


def great_circle_distance(point_a, point_b, circle_radius):
    """
    Calculates the gcd (https://en.wikipedia.org/wiki/Great-circle_distance) between two
    points on a sphere.

    Parameters
    ----------
    point_a : tuple of float
        (lat_a, lon_a) -- tuple specifying latitude and longitude
    point_b : tuple of float
        (lat_b, lon_b) -- tuple specifying latitude and longitude
    circle_radius : float
        The radius of the circle upon which the points are.

    Returns
    -------
    float
        The great circle distance between the two points
    """
    lat_a, lon_a = np.deg2rad(point_a)
    lat_b, lon_b = np.deg2rad(point_b)
    delta_lon = lon_b - lon_a
    central_angle = np.arccos(
        np.sin(lat_a) * np.sin(lat_b)
        + np.cos(lat_a) * np.cos(lat_b) * np.cos(delta_lon)
    )
    return circle_radius * central_angle


def initial_bearing(point_a, point_b):
    """
    Calculates the initial bearing (also known as forward azimuth) from point_a to point_b

    Parameters
    ----------
    point_a : tuple of float
        (lat_a, lon_a) -- tuple specifying latitude and longitude
    point_b : tuple of float
        (lat_b, lon_b) -- tuple specifying latitude and longitude

    Returns
    -------
    float
        The initial bearing in radians
    """
    lat_a, lon_a = np.deg2rad(point_a)
    lat_b, lon_b = np.deg2rad(point_b)
    delta_lon = lon_b - lon_a
    return np.arctan2(
        np.sin(delta_lon) * np.cos(lat_b),
        np.cos(lat_a) * np.sin(lat_b)
        - np.sin(lat_a) * np.cos(lat_b) * np.cos(delta_lon),
    )
