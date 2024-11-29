import numpy as np

from typing import Tuple, List, Union, TypeAlias
import numpy.typing as npt

FloatLike: TypeAlias = Union[float, np.floating]

def quadratic_detrend(full_data: npt.NDArray):
    means = np.mean(full_data, axis=(1, 2))
    long_term_mean = np.mean(means)
    
    days = np.arange(full_data.shape[0])
    a, b, c = np.polyfit(days, means - long_term_mean, 2)
    trend = a*days**2 + b*days + c
    
    trendshape = (-1,) + (1,) * (len(full_data.shape)-1)
    full_data_detrended = full_data - trend.reshape(trendshape)
    
    return (full_data_detrended, (a, b, c))

def inv_quadratic_detrend(full_data_detrended: npt.NDArray, coeff: Tuple[FloatLike, FloatLike, FloatLike]):
    a, b, c = coeff
    days = np.arange(full_data_detrended.shape[0])
    trend = a*days**2 + b*days + c
    
    trendshape = (-1,) + (1,) * (len(full_data_detrended.shape)-1)
    full_data = full_data_detrended + trend.reshape(trendshape)
    
    return full_data

def _custom_year_pad(data, n_years):
    pad_width = n_years*365
    
    if pad_width > len(data):
        raise ValueError(f"Pad length ({pad_width}) must be shorter than data length ({len(data)})")
    
    start_year = data[:pad_width]
    end_year = data[-pad_width:]
    
    return np.concatenate((start_year, data, end_year)), pad_width
    

def detrend_seasonal(full_data: npt.NDArray):
    means = np.mean(full_data, axis=(1, 2))
    long_term_mean = np.mean(means)
    mean_anom = means-long_term_mean

    padded_data, pad_width = _custom_year_pad(mean_anom, 2)
    
    fft_result = np.fft.fft(padded_data)
    freqs = np.fft.fftfreq(len(padded_data), d=1) #sample spacing is once per day
    
    #construct our filter
    center_notch = 365.25
    half_bandwidth = 30
    period_ranges_pos = np.array([center_notch-half_bandwidth, center_notch+half_bandwidth])
    period_ranges_neg = -np.flip(period_ranges_pos)

    filter = np.full(shape=freqs.shape, fill_value=True)
    filter[(freqs < 1/period_ranges_pos[0]) & (freqs > 1/period_ranges_pos[1])] = False
    filter[(freqs < 1/period_ranges_neg[0]) & (freqs > 1/period_ranges_neg[1])] = False
    
    #execute filter
    fft_result[~filter] = 0
    detrended_means = np.real(np.fft.ifft(fft_result))[pad_width:-pad_width]
    
    #calculate differences and detrend
    diffs = mean_anom - detrended_means
    trendshape = (-1,) + (1,) * (len(full_data.shape)-1)
    full_data_detrended = full_data - diffs.reshape(trendshape)
    
    return (full_data_detrended, diffs)

def inv_detrend_seasonal(full_data_detrended: npt.NDArray, diffs: npt.NDArray):
    trendshape = (-1,) + (1,) * (len(full_data_detrended.shape)-1)
    inv_detrend = full_data_detrended + diffs.reshape(trendshape)
    
    return inv_detrend