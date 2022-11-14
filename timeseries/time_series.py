
import logging
import numpy as np
import scipy.signal as sig 
from scipy.interpolate import interp1d


class TimeSeries(object):
    def __init__(self,v,t,label='',interp='linear'):
        """
        Arbitrary time series with irregular sampling

        interp : interpolation mode
                 * nearest: return nearest sample
                 * linear: interpolate linearly when a value 
                   is requested in between samples
        """
        self.t=np.asarray(t)
        self.v=np.asarray(v)
        self.label=label
        self.interp_mode=interp

    @property
    def dtype(self):
        return self.v.dtype

    def __getitem__(self,key):
        if hasattr(key, "__iter__"):
            if hasattr(key[0], "__iter__"):
                ret = []
                for ii in key:
                    try:
                        ret.extend(self[ii])
                    except TypeError:
                        ret.append(self[ii])
            else:
                if type(key) == tuple:
                    ret = self._values_in_range(key[0],key[1]) 
                else:
                    ret = self.value_at_time(key)
            return ret
        else:
            return self.value_at_time(key)

    def _join_times(self, t2):
        return np.unique(np.concatenate((self.t,t2)))

    def __add__(self, other):
        if isinstance(other, TimeSeries):
            new_times = self._join_times(other.t)
            new_vals = self[new_times] + other[new_times]
            return self.__class__(new_vals, new_times)
        else:
            return self.__class__(self.v + other, self.t)

    def __sub__(self, other):
        if isinstance(other, TimeSeries):
            new_times = self._join_times(other.t)
            new_vals = self[new_times] - other[new_times]
            return self.__class__(new_vals, new_times)
        else:
            return self.__class__(self.v - other, self.t)

    def __mul__(self, other):
        if isinstance(other, TimeSeries):
            new_times = self._join_times(other.t)
            new_vals = self[new_times] * other[new_times]
            return self.__class__(new_vals, new_times)
        else:
            return self.__class__(self.v * other, self.t)

    def __truediv__(self, other):
        if isinstance(other, TimeSeries):
            new_times = self._join_times(other.t)
            new_vals = self[new_times] / other[new_times]
            return self.__class__(new_vals, new_times)
        else:
            return self.__class__(self.v / other, self.t)

    @property
    def iloc(self):
        return self.v

    def left_right_indices(self,time):
        if time in self.t:
            idx = np.flatnonzero(self.t==time)[0]
            return np.array([idx,idx])
        idx = np.flatnonzero(self.t - time >= 0)
        if len(idx) <= 0:
            return np.array([len(self.t),None])
        elif idx[0] == 0:
            return np.array([None,0])
        else:
            return np.array([idx[0]-1,idx[0]])

    def nearest_sample_time(self, time):
        idx = np.min(np.abs(time-self.t))
        return self.t[idx]
        
    def value_at_time(self,t,interp='linear'):
        try:
            self.interpolator
        except AttributeError:
            self._create_interpolator()
        
        try:
            x = self.interpolator(t)
        except TypeError:
            x = np.array([y(t) for y in self.interpolator])
        try:
            return float(x)
        except TypeError:
            return x
                
    def percentile(self, val, from_time=None, to_time=None):
        xt,xv = self.times_values_in_range(from_time=from_time, to_time=to_time)
        return np.percentile(xv,val)

    def mean(self, from_time=None, to_time=None):
        xt,xv = self.times_values_in_range(from_time=from_time, to_time=to_time)
        return np.mean(xv)
    
    def max(self, from_time=None, to_time=None):
        xt,xv = self.times_values_in_range(from_time=from_time, to_time=to_time)
        return np.max(xv)
    
    def min(self, from_time=None, to_time=None):
        xt,xv = self.times_values_in_range(from_time=from_time, to_time=to_time)
        return np.min(xv)

    def std(self, from_time=None, to_time=None):
        xt,xv = self.times_values_in_range(from_time=from_time, to_time=to_time)
        return np.std(xv)

    def apply(self, fun, from_time=None, to_time=None):
        xt,xv = self.times_values_in_range(from_time=from_time, to_time=to_time)
        return self.__class__(fun(xv),xt)

    def min_time(self, from_time=None, to_time=None):
        xt,xv = self.times_values_in_range(from_time=from_time, to_time=to_time)
        return xt[np.argmin(xv)]

    def max_time(self, from_time=None, to_time=None):
        xt,xv = self.times_values_in_range(from_time=from_time, to_time=to_time)
        return xt[np.argmax(xv)]

    def changepoints(self,n=None,order=0,mindist=0,from_time=None, to_time=None ):
        from .features import changepoints
        xt,xv = self.times_values_in_range(from_time=from_time, to_time=to_time)
        if order == 0:
            xv=np.cumsum(xv)
        elif order>1:
            difford = order-1
            xv = np.diff(xv,difford)
            xt = (xt[:-difford]+xt[difford:])/2
        cp = changepoints(xv,n,mindist=mindist)
        return xt[cp]
    
    def window_filter(self, func, twind=0.0):
        ret = np.zeros_like(self.v)
        for ii, (t,x) in enumerate(zip(self.t, self.v)):
            idx = (self.t>t-twind/2) & (self.t<t+twind/2)
            ret[ii] = func(self.v[idx])
        return self.__class__(t=self.t,v=ret, label=self.label+' (filt.)')

    def _create_interpolator(self):
        if len(self.v.shape)==1:
            self.interpolator = interp1d(self.t,self.v,kind=self.interp_mode, bounds_error=False)
        else:
            self.interpolator = []
            for ii in range(self.v.shape[1]):
                self.interpolator.append(interp1d(self.t,self.v[:,ii],kind=self.interp_mode, bounds_error=False))
        
    def _values_in_range(self, from_time=None, to_time=None):
        if from_time is None:
            from_time=np.min(self.t)
        if to_time is None:
            to_time=np.max(self.t)
        idx = ((self.t>=from_time) &
               (self.t<=to_time))
        return self.v[idx]

    def times_values_in_range(self, from_time=None, to_time=None):
        if from_time is None:
            from_time=np.min(self.t)
        if to_time is None:
            to_time=np.max(self.t)
        idx = ((self.t>=from_time) &
               (self.t<=to_time))
        xvec = self.v
        return self.t[idx], xvec[idx]

    def reverse_interpolate(self,val,indices,interp="linear"):
        """
        Finds the interpolated time corresponding to value val between indices
        """
        if interp == 'linear':
            tv = self.t[indices]
            xv = self.v[indices]
            return (val-xv[0])/(xv[1]-xv[0])*(tv[1]-tv[0])+tv[0]
        
        

    def crossing_times(self,val,index=None,from_time=None,to_time=None,interp=None):
        tvec,xvec = self.times_values_in_range(from_time=from_time,to_time=to_time)
        if index:
            xvec = xvec[:,index]
        crossing_mask = (xvec[:-1]<val) & (xvec[1:]>=val)
        crossing_up = np.flatnonzero(crossing_mask)

        crossing_mask = (xvec[:-1]>=val) & (xvec[1:]<val)
        crossing_down = np.flatnonzero(crossing_mask)
        if interp is None:
            interp = self.interp_mode

        if interp == 'linear':
            tup = np.zeros(len(crossing_up))
            for ii,cu in enumerate(crossing_up):
                tup[ii] = self.reverse_interpolate(val,[cu,cu+1])
            tdn = np.zeros(len(crossing_down))
            for ii,cu in enumerate(crossing_down):
                tdn[ii] = self.reverse_interpolate(val,[cu,cu+1])
            #tv = tvec[crossing_down:crossing_down+2]
            #xv = xvec[crossing_down:crossing_down+2]
            #tdn = (val-xv[0])/(xv[1]-xv[0])*(tv[1]-tv[0])
        else:
            tup = tvec[crossing_up+1]
            tdn = tvec[crossing_down]
        return tup, tdn

    def midpoint_value(self, percentile=5, from_time=None, to_time=None):
        xlow = self.percentile(percentile, from_time, to_time) 
        xhi  = self.percentile(100-percentile, from_time, to_time) 
        return (xhi+xlow)/2

    def start_end_midpoint_crossings(self, percentile=5, from_time=None, to_time=None):
        val = self.midpoint_value(percentile=percentile)
        x_up, x_down = self.crossing_times(val, from_time=from_time, to_time=to_time)
        if (len(x_up)>1) | (len(x_down)>1):
            logging.warn('multiple crossings found')
        return x_up[0],x_down[-1]

    def plot(self, **kwargs):
        """
        Uses matplotlib to generate a plot of the time series
        
        pass "ax" argument to plot in an existing axis 
        """
        import matplotlib.pyplot as pl
        try:
            ax = kwargs.pop("ax")
            fig = ax.figure
        except KeyError:
            fig,ax = pl.subplots(1)
        l = kwargs.pop('label',self.label)
        ax.plot(self.t,self.v,label=l,**kwargs)
        return fig,ax

    def as_sampled_timeseries(self, dt=None):
        if dt is None:
            dt = np.median(np.diff(self.t))
        tst = np.min(self.t)
        tend = np.max(self.t)
        tvec = np.arange(tst,tend,dt)
        xvec = self.value_at_time(tvec)
        from .sampled_time_series import SampledTimeSeries 
        return SampledTimeSeries(xvec,t=self.t,label=self.label+' (samp.)')