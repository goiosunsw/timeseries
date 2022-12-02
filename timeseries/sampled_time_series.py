
import logging
import numpy as np
import scipy.signal as sig 
from scipy.interpolate import interp1d

from .time_series import TimeSeries

class SampledTimeSeries(TimeSeries):
    def __init__(self,v,t=None,dt=None,t_start=0.0,label='',interp='linear'):
        self.v=np.asarray(v)
        if dt is not None:
            self.dt = dt
        self.t_start = t_start
        if t is not None:
            tdiff = np.diff(t)
            self.dt = np.median(tdiff)
            self.t_start = np.min(t)
            maxerr = np.max(np.abs(self.t-t))
            #logging.info("Maximum error in time vector is {} ({}%%)".format(maxerr,maxerr/self.dt))
            
        self.label=label
        self.interp_mode=interp
    
    @property
    def t(self):
        return np.arange(len(self.v))*self.dt + self.t_start 

    def diff(self,ord=1,dist=1):
        """
        Return the finite difference of order n of the time series
        """
        t = self.t
        dt = self.dt
        if dist==1:
            dv = np.diff(self.v,ord)/(dt**ord)
        
            t = (t[:-ord]+t[ord:])/2
        else:
            dv = self.v
            for ii in range(ord):
                dv = (dv[dist:]-dv[:-dist])/(dt*dist)
                t = (t[:-dist]+t[dist:])/2
        return SampledTimeSeries(t=t,v=dv, label= self.label+' (diff{})'.format(ord))

    def peaks(self,twind=0.0,sign=1,**kwargs):
        distance = int(twind/self.dt)
        dist = kwargs.pop('distance',None)
        peak_idx, properties = sig.find_peaks(self.v*sign,distance=distance,**kwargs)
        return self.t[peak_idx], properties
        

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
        
    def window_filter_matrix(self, func, twind=0.0):
        """
        Apply a function to a window centered on every sample of the time-series
        
        func should be func(x) returning a scalar, for example np.mean
        
        Very memory intensive : use only for short signals or short windows
        """
        nrad = int(twind/self.dt/2)
        xarr = np.tile(self.v[:,np.newaxis],(1,nrad*2+1))
        for ii in range(-nrad,nrad+1):
            xarr[:,nrad+ii] = self.shifted_values(ii)
        xs = func(xarr, axis=1)
        ts = self.t
        return SampledTimeSeries(xs,dt=self.dt,t_start=self.t_start, label=self.label+' (filt.)')

    def window_filter(self, func, twind=0.0, thop=None):
        """
        Apply a function to a window centered on time spaced by thop approximately

        where func: func = np.mean for example
        
        """
        nwind = int(twind/self.dt)
        if thop:
            nhop = int(thop/self.dt)
        else:
            nhop=1
        newv=[]
        newt=[]
        
        for ii in range(0,len(self.v)-nwind,nhop):
            xx = self.v[ii:ii+nwind]
            xw = xx 
            newv.append(func(xw))
            newt.append(self.index_to_time(ii+nwind/2))
        return SampledTimeSeries(newv,t=newt,label=self.label+' (filt)')

    def reduce_with_function(self, func, twind=0.0, thop=None, windfunc=np.ones):
        """
        Apply a function to a window centered on time spaced by thop approximately
        
        func should be func(x, wsum), for example
        
        def func(x, wsum):
            return sum(x**2)/wsum
            
        for a second moment estimation
        """
        nwind = int(twind/self.dt)
        nhop = int(thop/self.dt)
        wind = windfunc(nwind)
        newt = []
        newv = []
        wsum = np.sum(wind)
        
        for ii in range(0,len(self.v)-nwind,nhop):
            xx = self.v[ii:ii+nwind]
            xw = xx*wind 
            newv.append(func(xw,wsum))
            newt.append(self.index_to_time(ii+nwind/2))
        return SampledTimeSeries(newv,t=newt,label=self.label+' (reduced)')
    

    def time_to_index(self, time, approx_to='left'):
        index_frac = (time-self.t_start)/self.dt
        if index_frac<0:
            index_frac=0
        if index_frac>len(self.v):
            index_frac=len(self.v)
        if approx_to == 'none' or approx_to is None:
            return index_frac
        elif approx_to == 'left':
            return int(index_frac)
        elif approx_to == 'right':
            return int(np.ceil(index_frac))

    def index_to_time(self, index):
        """
        Return the time corresponding to a given fractional index
        """
        return index*self.dt + self.t_start
        
    def range_to_slice(self, from_time=None, to_time=None,inclusive=True):
        if from_time is None:
            from_time = self.t_start
        if to_time is None:
            to_time = self.t_start + len(self.v)*self.dt

        if inclusive:
            from_idx = self.time_to_index(from_time, approx_to='left')
            to_idx = self.time_to_index(to_time, approx_to='right')
        else:
            from_idx = self.time_to_index(from_time, approx_to='right')
            to_idx = self.time_to_index(to_time, approx_to='left')
        return slice(from_idx,to_idx)
    
    def _values_in_range(self, from_time=None, to_time=None):
        idx = self.range_to_slice(from_time, to_time)
        return self.v[idx]

    def times_values_in_range(self, from_time=None, to_time=None, nmin=0):
        idx = self.range_to_slice(from_time, to_time)
        if idx.stop <= idx.start:
            return super().times_values_in_range(from_time=from_time, to_time=to_time, nmin=nmin)
        else:
            return self.t[idx], self.v[idx]

    def resample(self, fact=1):
        x = self.v
        n = len(x)
        y = np.floor(np.log2(n + 2*fact))
        nextpow2 = int(np.power(2, y+1))
        npadl = (nextpow2 - n)//2
        npadr = nextpow2 - n - npadl

        x = np.pad(x, ((npadl, npadr), ), mode='linear_ramp')
        print(len(x))
        xlf = sig.resample(x, int(len(x)/fact))
        ilf = np.arange(0,len(xlf))*fact
        idx = (ilf > npadl) & (ilf < n+npadl)
        tidx = np.array([ilf[idx]-npadl])
        return SampledTimeSeries(xlf[idx], t=self.index_to_time(tidx))
    

    def copy(self):
        return SampledTimeSeries(self.v,dt=self.dt,t_start=self.t_start)

    def as_irregular_timeseries(self):
        return TimeSeries(t=self.t,v=self.v,label=self.label+' (irreg.)')
