
import logging
import numpy as np
import scipy.signal as sig 
from scipy.interpolate import interp1d


class TimeSeries(object):
    def __init__(self,tvec,xvec,label='',interp='linear'):
        self.t=tvec
        self.v=xvec
        self.label=label
        self.interp_mode=interp

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
                ret = self._values_in_range(key[0],key[1]) 
            return ret
        else:
            return self.get_value_at_time(key)
        
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
    
    def apply(self, fun, from_time=None, to_time=None):
        xt,xv = self.times_values_in_range(from_time=from_time, to_time=to_time)
        return fun(xv)

    def min_time(self, from_time=None, to_time=None):
        xt,xv = self.times_values_in_range(from_time=from_time, to_time=to_time)
        return xt[np.argmin(xv)]

    def max_time(self, from_time=None, to_time=None):
        xt,xv = self.times_values_in_range(from_time=from_time, to_time=to_time)
        return xt[np.argmax(xv)]

    def changepoints(self,n=None,order=0,mindist=0,from_time=None, to_time=None ):
        xt,xv = self.times_values_in_region(from_time=from_time, to_time=to_time)
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
            idx = (self.t>t-twind) & (self.t<t+twind)
            ret[ii] = func(self.v[idx])
        return TimeSeries(self.t,ret)

    def _create_interpolator(self):
        if len(self.v.shape)==1:
            self.interpolator = interp1d(self.t,self.v,kind=self.interp_mode)
        else:
            self.interpolator = []
            for ii in range(self.xvec.shape[1]):
                self.interpolator.append(interp1d(self.t,self.v[:,ii],kind=self.interp_mode))
        
    def _values_in_range(self, from_time=None, to_time=None):
        if from_time is None:
            from_time=np.min(self.t)
        if to_time is None:
            to_time=np.max(self.t)
        idx = ((self.t>=from_time) &
               (self.t<=to_time))
        xvec = self.get_values()
        return xvec[idx]

    def times_values_in_range(self, from_time=None, to_time=None):
        if from_time is None:
            from_time=np.min(self.t)
        if to_time is None:
            to_time=np.max(self.t)
        idx = ((self.t>=from_time) &
               (self.t<=to_time))
        xvec = self.v
        return self.tvec[idx], xvec[idx]

    def crossing_times(self,val,index=None,from_time=None,to_time=None,interpolated=False):
        tvec,xvec = self.times_values_in_range(from_time=from_time,to_time=to_time)
        if index:
            xvec = xvec[:,index]
        crossing_mask = (xvec[:-1]<val) & (xvec[1:]>=val)
        crossing_up = np.flatnonzero(crossing_mask)

        crossing_mask = (xvec[:-1]>=val) & (xvec[1:]<val)
        crossing_down = np.flatnonzero(crossing_mask)

        if interpolated:
            tup = np.zeros(len(crossing_up))
            for ii,cu in enumerate(crossing_up):
                tv = tvec[cu:cu+2]
                xv = xvec[cu:cu+2]
                tup[ii] = (val-xv[0])/(xv[1]-xv[0])*(tv[1]-tv[0])+tv[0]
            tdn = np.zeros(len(crossing_down))
            for ii,cu in enumerate(crossing_down):
                tv = tvec[cu:cu+2]
                xv = xvec[cu:cu+2]
                tdn[ii] = (val-xv[0])/(xv[1]-xv[0])*(tv[1]-tv[0])+tv[0]
            #tv = tvec[crossing_down:crossing_down+2]
            #xv = xvec[crossing_down:crossing_down+2]
            #tdn = (val-xv[0])/(xv[1]-xv[0])*(tv[1]-tv[0])
        else:
            tup = tvec[crossing_up+1]
            tdn = tvec[crossing_down]
        return tup, tdn

    def midpoint_value(self, percentile=5, from_time=None, to_time=None):
        xlow = self.percentile(percentile, from_time, to_time) 
        xhi  = self.percentile(1-percentile, from_time, to_time) 
        return (xhi+xlow)/2

    def start_end_midpoint_crossings(self, percentiles=5, from_time=None, to_time=None):
        val = self.get_midpoint_value(percentiles=percentiles)
        x_up, x_down = self.get_crossing_times(val, from_time=from_time, to_time=to_time)
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
        ax.plot(self.t,self.v,**kwargs)
        return fig,ax

    def to_sampled_timeseries(self, dt=None):
        if dt is None:
            dt = np.median(np.diff(self.tvec))
        tst = np.min(self.tvec)
        tend = np.max(self.tvec)
        tvec = np.arange(tst,tend,dt)
        xvec = self.value_at_time(tvec)
        from .sampled_times_series import SampledTimeSeries 
        return SampledTimeSeries(tvec,xvec)