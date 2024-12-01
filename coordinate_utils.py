import numpy as np
from datetime import datetime

def convert_lon_0_360(lon):
    return lon if lon >= 0 else lon + 360
npconvert_lon_0_360 = np.vectorize(convert_lon_0_360)

def convert_lon_180W_180E(lon):
    return lon - 360 if lon > 180 else lon
npconvert_lon_180W_180E = np.vectorize(convert_lon_180W_180E)

def ts_to_dt(timestamp):
    try:
        return datetime.strptime(timestamp, '%Y-%m-%dT%H:%M:%S.%f')
    except ValueError:
        pass
    
    try:
        return datetime.strptime(timestamp, '%Y-%m-%dT%H:%M:%S')
    except ValueError:
        pass
    
    try:
        return datetime.strptime(timestamp, '%Y-%m-%dT%H:%M')
    except ValueError:
        pass
    
    try:
        return datetime.strptime(timestamp, '%Y-%m-%dT%H')
    except ValueError:
        pass
    
    return datetime.strptime(timestamp, '%Y-%m-%d')

def sphere_surface_area(lat, lon, resolution):
    lon1 = lon - resolution/2
    lon2 = lon + resolution/2
    lat1 = lat - resolution/2
    lat2 = lat + resolution/2
    
    a = 6.378*10**6
    
    return (np.pi * (a**2)) * (np.sin(np.deg2rad(lat2)) - np.sin(np.deg2rad(lat1))) * ((lon2-lon1)/180)