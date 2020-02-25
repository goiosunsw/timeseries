
import logging
import numpy as np
import scipy.signal as sig 
from scipy.interpolate import interp1d

from .time_series import TimeSeries

class SampledTimeSeries(TimeSeries):
    def __init__(self,xvec,tvec=None,dt=None,t_start=0.0,label='',interp='linear'):
        if dt is not None:
            self.dt = dt
        self.t_start = t_start
        if tvec is not None:
            tdiff = np.diff(tvec)
            self.dt = np.median(tdiff)
            self.t_start = np.min(tvec)
            maxerr = np.max(np.abs(self.t-tvec))
            logging.info("Maximum error in time vector is {} ({}%%)".format(maxerr,maxerr/self.dt))
            
        self.v=xvec
        self.label=label
        self.interp_mode=interp
    
    @property
    def t(self):
        return np.arange(len(self.v))*self.dt + self.t_start 

    def _padding(self,n):
        """
        return n extrapolated samples at end if n>0, or 
              -n extrapolated samples at beginning if n<0
        """
        return np.zeros(abs(n))

    def shifted_values(self,n):
        if n>=0:
            return np.concatenate((self.v[n:], self._padding(n)))
        else:
            return np.concatenate((self._padding(n),self.v[:n])) 
        
    def window_filter(self, func, twind=0.0):
        nrad = int(twind/self.dt/2)
        xarr = np.tile(self.v[:,np.newaxis],(1,nrad*2+1))
        for ii in range(-nrad,nrad+1):
            xarr[:,nrad+ii] = self.shifted_values(ii)
        xs = func(xarr, axis=1)
        ts = self.t
        return SampledTimeSeries(xs,dt=self.dt,t_start=self.t_start)

    def time_to_index(self, time, approx_to='left'):
        index_frac = (time-self.t_start)/self.dt
        if approx_to == 'none' or approx_to is None:
            return index_frac
        elif approx_to == 'left':
            return int(index_frac)
        elif approx_to == 'right':
            return int(np.ceil(index_frac))
        
    def range_to_slice(self, from_time=None, to_time=None):
        if from_time is None:
            from_time = self.t_start
        if to_time is None:
            to_time = self.t_start + len(self.v)*self.dt

        from_idx = self.time_to_index(from_time, approx_to='right')
        to_idx = self.time_to_index(to_time, approx_to='left')
        return slice(from_idx,to_idx)
    
    def _values_in_range(self, from_time=None, to_time=None):
        idx = self.range_to_slice(from_time, to_time)
        return self.v[idx]

    def times_values_in_range(self, from_time=None, to_time=None):
        idx = self.range_to_slice(from_time, to_time)
        return self.t[idx], self.v[idx]
