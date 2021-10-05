import random
import numpy as np
from numpy.core.fromnumeric import std
from numpy.lib.function_base import median
import scipy
from random import choice
from scipy.optimize import least_squares
import librosa
import os
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import (LinearRegression, TheilSenRegressor, RANSACRegressor, HuberRegressor)
import matplotlib.patches as mpatches
from constants_parser import Constants

import matplotlib.pyplot as plt

# from ransac import RansacModel
# from linearleastsquare import LinearLeastSqaureModel


class Partial():
    def __init__(self, frequency, order, peak_idx):
        self.frequency = frequency
        self.order = order
        self.peak_idx = peak_idx

class ToolBox():
    """here all tools developed are stored. designed this way so 
    it can be expanded and incorporate other methods for partial detection 
    or computing beta coefficient etc. For example usage/alterations see bellow"""

    def __init__(self, partial_tracking_func, inharmonicity_compute_func, partial_func_args, inharmonic_func_args):
        self.partial_func = partial_tracking_func
        self.inharmonic_func = inharmonicity_compute_func
        self.partial_func_args = partial_func_args
        self.inharmonic_func_args = inharmonic_func_args
    
class NoteInstance():
    """move to other level of package"""
    def __init__(self, fundamental, onset, audio, ToolBoxObj:ToolBox ,sampling_rate, constants : Constants, longaudio=np.array([]), midi_flag = False):
        self.fundamental = fundamental
        self.onset = onset
        self.audio = audio
        self.longaudio = longaudio # may be used for ("internal") f0 computation
        self.sampling_rate = constants.sampling_rate
        self.polyfit = constants.polyfit
        self.fft=np.fft.fft(self.audio,n = constants.size_of_fft)
        self.frequencies=np.fft.fftfreq(constants.size_of_fft,1/self.sampling_rate)
        self.partials = []
        self.differences = []
        self.abc = []
        self.large_window = None
        # self.train = False
        self.string = None
        if midi_flag:
            self.recompute_fundamental(constants, fundamental/2)

        ToolBoxObj.partial_func(self, ToolBoxObj.partial_func_args) # if a different partial tracking is incorporated keep second function arguement, else return beta from second function and change entirely

    def get_string(string):
        self.string = string

    def plot_DFT(self, peaks=None, peaks_idx=None, lim=None, ax=None, save_path=None):
        [a,b,c] = self.abc
        w = self.large_window
     
        # main function
        ax.plot(self.frequencies, self.fft.real)

        for k in range(50): # draw vertical red dotted lines indicating harmonics
            ax.axvline(x=self.fundamental*k, color='r', ls='--', alpha=0.85, lw=0.5)     

        if w: # draw windows as little boxes
            f0 = self.fundamental
            for k in range(1,lim+1):
                # f = k*f0 * np.sqrt(1+b_est*k**2)
                f = window_centering_func(k,f0, a=a,b=b,c=c)
                rect=mpatches.Rectangle((f-w//2,-80),w,160, fill=False, color="purple", linewidth=2)
                # plt.gca().add_patch(rect)
                ax.add_patch(rect)

        if peaks and peaks_idx: # draw peaks
            ax.plot(peaks, self.fft.real[peaks_idx], "x", alpha=0.7)

        ax.set_xlim(0, window_centering_func(lim+1,f0, a=a,b=b,c=c))
        ax.set_ylim(-100, 100)

        return ax


    def plot_partial_deviations(self, lim=None, res=None, peaks_idx=None, ax=None, note_instance=None, annos_instance=None, tab_instance=None):

        differences = self.differences
        w = self.large_window
        [a,b,c] = res
        kapa = np.linspace(0, lim, num=lim*10)
        y = a*kapa**3 + b*kapa + c

        if peaks_idx:
            PeakAmps = [ self.frequencies[peak_idx] for peak_idx in peaks_idx ]
            PeakAmps = PeakAmps / max(PeakAmps) # Normalize
        else:
            PeakAmps = 1
        ax.scatter(np.arange(2,len(differences)+2), differences, alpha=PeakAmps)
        f0 = self.fundamental
        # plot litte boxes
        for k in range(2, len(differences)+2):
            pos = window_centering_func(k, f0, a, b, c) - k*f0

            rect=mpatches.Rectangle((k-0.25, pos-w//2), 0.5, w, fill=False, color="purple", linewidth=2)
            # plt.gca().add_patch(rect)
            ax.add_patch(rect)

        ax.plot(kapa,y, label = 'new_estimate')
        ax.grid()
        ax.legend()

        if annos_instance:
            if note_instance.string == annos_instance.string:
                c = 'green'
            else:
                c = 'red'
        else:
            c = 'black'
        
        plt.title("pred: "+ str(note_instance.string) + ", annotation: " + str(annos_instance.string) + ', fret: ' + str(tab_instance.fret) + ' || f0: ' + str(round(self.fundamental,2)) + ', beta_estimate: '+ str(round(self.beta,6)) + '\n a = ' + str(round(a,5)), color=c)

        return ax

    def weighted_argmean(self, peak_idx, w=6):
        min_idx = max(peak_idx - w//2, 0)
        max_idx = min(peak_idx + w//2, len(self.frequencies)-1)
        window = range(min_idx, max_idx+1)
        amp_sum = sum([self.fft.real[i] for i in window])
        res = sum( [self.fft.real[i]/amp_sum * self.frequencies[i] for i in window] )
        return res

    def recompute_fundamental(self, constants : Constants, window = 10): 
        
        if not self.longaudio.any(): # if we want to work on given self.audio (and not on self.longaudio)
            filtered = zero_out(self.fft, self.fundamental, window, constants)
        else:
            longaudio_fft = np.fft.fft(self.longaudio,n = constants.size_of_fft)
            filtered = zero_out(longaudio_fft, self.fundamental, window, constants)
        peaks, _  =scipy.signal.find_peaks(np.abs(filtered),distance=100000) # better way to write this?
        
        max_peak_freq = self.weighted_argmean(peak_idx=peaks[0], w=0) # w=0 means that no weighted argmean is employed, Note: equivalent to # max_peak_freq = self.frequencies[peaks[0]]

        self.fundamental = max_peak_freq
        return max_peak_freq


def window_centering_func(k,f0=None,a=None,b=None,c=None, b_est=None):
    if b_est: # standard inharmonicity equation indicating partials position
        center_freq = k*f0 * np.sqrt(1+b_est*k**2)
    else: # polynomial approximation of partials
        center_freq = a*k**3+b*k+c + (k*f0)
    return center_freq        


def compute_partials(note_instance, partial_func_args):
    """compute up to no_of_partials partials for note instance. 
    large_window is the length of window arround k*f0 that the partials are tracked with highest peak."""
    # no_of_partials = partial_func_args[0] NOTE: deal with it somehow
    note_instance.large_window = partial_func_args[1]
    constants = partial_func_args[2]
    diviate = round(note_instance.large_window*note_instance.fft.size/note_instance.sampling_rate)
    f0 = note_instance.fundamental

    a, b, c = 0, 0, 0
    N=6 # n_iterations # TODO: connect iterations with the value constants.no_of_partials
    for i in range(N):
        lim = 5*(i+1)+1 # NOTE: till 30th/50th partial
        for k in range(2,lim): # NOTE: 2 stands for the 2nd partial! TODO: use 3 instead if we wan t to start processing from the 2nd partial and further
            # center_freq = k*f0 * np.sqrt(1+b_est*k**2)
            center_freq = window_centering_func(k,f0, a=a,b=b,c=c) # centering window in which to look for peak/partial
            try:
                filtered = zero_out(note_instance.fft, center_freq=center_freq , window_length=diviate, constants=constants)
               
                peaks, _  = scipy.signal.find_peaks(np.abs(filtered),distance=100000) # better way to write this?
                # max_peak = note_instance.frequencies[peaks[0]]
                max_peak = note_instance.weighted_argmean(peak_idx=peaks[0], w=6)
                note_instance.partials.append(Partial(frequency=max_peak, order=k, peak_idx=peaks[0]))
          
            except Exception as e:
                print(e)
                print('MyExplanation: Certain windows where peaks are to be located surpassed the length of the DFT.')
                break
        # iterative beta estimates
        _, [a,b,c] = compute_inharmonicity(note_instance, [])
        note_instance.abc = [a,b,c]
        # compute differences/deviations
        note_instance.differences, orders = zip(*compute_differences(note_instance))
        if i != N-1:
            note_instance.partials=[]

        peak_freqs = [partial.frequency for partial in note_instance.partials]
        peaks_idx = [partial.peak_idx for partial in note_instance.partials]

    # if constants.plot: 
    #     fig = plt.figure(figsize=(15, 10))
    #     note_instance.plot_partial_deviations(lim=lim, res=note_instance.abc, fig=fig)#, peaks_idx=Peaks_Idx)
    #     note_instance.plot_DFT(Peaks, Peaks_Idx, lim, fig=fig)   
    #     # fig.savefig()
    #     plt.show()
    #     plt.clf()
    
    # del Peaks, Peaks_Idx


def compute_differences(note_instance):
    differences = []
    for i, partial in enumerate(note_instance.partials):
        differences.append((abs(partial.frequency-(i+2)*note_instance.fundamental), i)) # i+2 since we start at first partial of order k=2
    return differences

def compute_inharmonicity(note_instance, inharmonic_func_args):
    differences, orders = zip(*compute_differences(note_instance))
  
    u=np.array(orders)+2
    if note_instance.polyfit == 'lsq':
        res=compute_least(u,differences) # least_squares
    if note_instance.polyfit == 'Thei':
        res=compute_least_TheilSen(u,differences) # least_squares
    
    [a,b,c]=res
    beta=2*a/(note_instance.fundamental+b) # Barbancho et al. (17)
    note_instance.beta = beta
    return beta, res
    
def compute_least(u,y):
    def model(x, u):
        return x[0] * u**3 + x[1]*u + x[2]
    def fun(x, u, y):
        return model(x, u)-y
    def jac(x, u, y):
        J = np.empty((u.size, x.size))
        J[:, 0] = u**3
        J[:, 1] = u
        J[:, 2] = 1
        return J
    x0=[0.00001,0.00001,0.000001]
    res = least_squares(fun, x0, jac=jac,bounds=(0,np.inf), args=(u, y),loss = 'soft_l1', verbose=0)
    return res.x    



# https://scikit-learn.org/stable/auto_examples/linear_model/plot_robust_fit.html#sphx-glr-auto-examples-linear-model-plot-robust-fit-py
# https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.TheilSenRegressor.html#sklearn.linear_model.TheilSenRegressor
def compute_least_TheilSen(u,y): 
    u = u[:, np.newaxis]
    poly = PolynomialFeatures(3)
    # print
    u_poly = poly.fit_transform(u)
    u_poly = np.delete(u_poly, 2, axis=1) # delete second coefficient (i.e. b=0, for  b * x**2)

    # estimator =  LinearRegression(fit_intercept=False)
    # estimator.fit(u_poly, y)

    estimator = TheilSenRegressor(random_state=42)
    # estimator = HuberRegressor()
    estimator.fit(u_poly, y)

    # print("coefficients:", estimator.coef_)
    return estimator.coef_[::-1]




def zero_out(fft, center_freq, window_length, constants : Constants):
    """return amplitude values of fft arround a given frequency; when outside window amplitude is zeroed out"""
    sz = fft.size
    x = np.zeros(sz,dtype=np.complex64)
    temp = fft
    dom_freq_bin = int(round(center_freq*sz/constants.sampling_rate))
    window_length = int(window_length)

    # for i in range(dom_freq_bin-window_length,dom_freq_bin+window_length): #NOTE: possible error
    for i in range(dom_freq_bin-window_length//2, dom_freq_bin+window_length//2): # __gb_
        x[i] = temp[i]**2

    return x


"""here on tests"""

#-----------ToolBox Example--------------------

#when changing only partial computing function
def example_compute_partial(note_instance, partial_func_args):
    arg0 = partial_func_args[0]
    arg1 = partial_func_args[1]
    arg2 = partial_func_args[2]
    for i in range(0, arg0):
        x = random.choice([arg1, arg2])
        note_instance.partials.append(Partial(x, i))
# example changing beta computation and probably bypassing previous partial computation
def example_inharmonicity_computation(note_instance, inharmonic_func_args):
    arg0 = inharmonic_func_args[0]
    arg1 = inharmonic_func_args[1]
    # do more stuff...
    note_instance.beta = arg0

#-----------end of ToolBox Example---------------

def wrapper_func(fundamental, audio, sampling_rate, constants : Constants):
    
    ToolBoxObj = ToolBox(compute_partials, compute_inharmonicity, [14, fundamental/2, constants], [])
    note_instance = NoteInstance(fundamental, 0, audio, ToolBoxObj, sampling_rate, constants)
    print(note_instance.beta, [x.frequency for x in note_instance.partials])















def compute_partials_old(note_instance, partial_func_args):
    """compute up to no_of_partials partials for note instance. 
    Freq_deviate is the length of window arround k*f0 that the partials are tracked with highest peak."""
    no_of_partials = partial_func_args[0]
    freq_diviate = partial_func_args[1]
    constants = partial_func_args[2]
    diviate = round(freq_diviate/(note_instance.sampling_rate/note_instance.fft.size))

    for i in range(2,no_of_partials):
        filtered = zero_out(note_instance.fft, center_freq=i*note_instance.fundamental, window_length=diviate, constants=constants)
        peaks, _  =scipy.signal.find_peaks(np.abs(filtered),distance=100000) # better way to write this?
        # print(peaks)
        max_peak = note_instance.frequencies[peaks[0]]
        note_instance.partials.append(Partial(max_peak, i))


def compute_partials_with_order(note_instance, partial_func_args):
    freq_diviate = partial_func_args[0]
    constants = partial_func_args[1]
    StrBetaObj = partial_func_args[2]
    b = []
    midi_note = round(librosa.hz_to_midi(note_instance.fundamental))
    t =  len(StrBetaObj.beta_lim[midi_note - 40])
    for n in range(0, t):
        note_instance.partials = []
        StrBetaObj.beta_lim[midi_note - 40].sort(reverse = True)
        k_max = StrBetaObj.beta_lim[midi_note - 40][n]
        #if k_max > constants.k_max:
        #    k_max = constants.k_max
        compute_partials(note_instance, [k_max, freq_diviate, constants])
        compute_inharmonicity(note_instance, [])
        #print(k_max, note_instance.beta) #uncoment if you want to check variation of betas in relation with k_max. Indicates why bellow code was added

        # NOTE: check for new threshold
        if note_instance.beta > 10**(-7):
            b.append(note_instance.beta)
    # print()
    # print(b, StrBetaObj.beta_lim[midi_note - 40])        
    if std(b)/np.mean(b) < 0.5: #coefficient of variation. If small then low variance of betas so get median, else mark as inconclusive
        note_instance.beta = median(b)
    else:
        note_instance.beta = 10**(-10)
        #    continue
        #else:
        #    return
    return

def compute_partials_with_order_strict(note_instance, partial_func_args):
    freq_diviate = partial_func_args[0]
    constants = partial_func_args[1]
    StrBetaObj = partial_func_args[2]
    midi_note = round(librosa.hz_to_midi(note_instance.fundamental))
    t =  len(StrBetaObj.beta_lim[midi_note - 40])
    for n in range(0, t):
        note_instance.partials = []
        k = min(StrBetaObj.beta_lim[midi_note - 40])
        
        compute_partials(note_instance, [k, freq_diviate, constants])
        compute_inharmonicity(note_instance, [])
        if note_instance.beta < 10**(-7):
            continue
        else:
            return
    return