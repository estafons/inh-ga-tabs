import jams
from pathlib import Path
import matplotlib.pyplot as plt
import itertools
import numpy as np
import configparser
import os, sys
import argparse
from GuitarTrain import GuitarSetTrainWrapper
import threading

BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
cur_path = Path(BASE_PATH + '/src/')
sys.path.append(str(cur_path))

from track_class import *
from Inharmonic_Detector import *
from inharmonic_Analysis import *
from constants_parser import Constants
import genetic
from helper import listen_to_the_intance

import statistics

import warnings
warnings.filterwarnings("ignore")

from GuitarSetTest import read_tablature_from_GuitarSet
import pickle     






def get_annos_for_separate_strings(data, annotation_name, constants : Constants):
    """function that loads annotation and audio file and returns instances"""
    annotations = read_tablature_from_GuitarSet(annotation_name, constants)
    tups = [(x.onset, x.fundamental, 6) for x in annotations.tablature.tablature]
    tablature = Tablature(tups, data, constants)
    track_instance = TrackInstance(tablature, data, constants)
    return track_instance, annotations


def compute_track_betas(track_instance : TrackInstance, annotations : Annotations, constants : Constants, StrBetaObj, channel):
    global betas
    global median_betas
    def close_event(): # https://stackoverflow.com/questions/30364770/how-to-set-timeout-to-pyplot-show-in-matplotlib
        plt.close() #timer calls this function after 3 seconds and closes the window 

    """Inharmonic prediction of tablature for eachstring/channel separately """
    for tab_instance, annos_instance in zip(track_instance.tablature.tablature, annotations.tablature.tablature):
        if annos_instance.string!=channel:
            continue
        ToolBoxObj = ToolBox(partial_tracking_func=compute_partials, inharmonicity_compute_func=compute_inharmonicity, partial_func_args=[constants.no_of_partials, tab_instance.fundamental/2, constants, StrBetaObj], inharmonic_func_args=[])
        note_instance = NoteInstance(tab_instance.fundamental, tab_instance.onset, tab_instance.note_audio, ToolBoxObj, track_instance.sampling_rate, constants)
        Inharmonic_Detector.DetectString(note_instance, StrBetaObj, constants.betafunc, constants)
        tab_instance.string = note_instance.string # predicted string
        if note_instance.string!=6:
            # print("found!!")
            tab_instance.fret = Inharmonic_Detector.hz_to_midi(note_instance.fundamental) - constants.tuning[note_instance.string]
            betas[annos_instance.string,tab_instance.fret].append(note_instance.beta)   
            if constants.plot:
                x = threading.Thread(target=listen_to_the_intance, args=(tab_instance.note_audio,))
                x.start()
                fig = plt.figure(figsize=(15, 10))
                timer = fig.canvas.new_timer(interval = 3000) #creating a timer object and setting an interval of 3000 milliseconds
                timer.add_callback(close_event)
                ax1 = fig.add_subplot(2, 1, 1)
                ax2 = fig.add_subplot(2, 1, 2)
                peak_freqs = [partial.frequency for partial in note_instance.partials]
                peaks_idx = [partial.peak_idx for partial in note_instance.partials]
                
                note_instance.plot_partial_deviations(lim=30, res=note_instance.abc, ax=ax1, note_instance=note_instance, annos_instance=annos_instance, tab_instance=tab_instance) #, peaks_idx=Peaks_Idx)
                note_instance.plot_DFT(peak_freqs, peaks_idx, lim=30, ax=ax2)   
                timer.start()
                plt.show()

def compute_all_betas(constants : Constants, StrBetaObj):
    """ function that runs tests on the jams files mentioned in the given file 
    and plots the confusion matrixes for both the genetic and inharmonic results."""
    print()
    print("Starting computation...")

    lines = os.listdir(constants.dataset_names_path+'/data/audio')
    for count, name in enumerate(lines):
        # print(name, count)
        if '_hex_cln.' not in name:
            continue    
        # if '_hex.' not in name:
        #     continue        
        if '_solo' not in name:
            continue
        print(name)
        track_name = name
        name = name.split('.')[0]
        # name = name[:-4] + '.jams'
        name = name[:-8] + '.jams'
        print(name, count,'/',len(lines))

        """ load 6-channel track and annotations"""
        track_name = Path(constants.track_path + track_name)
        annotation_name = Path(constants.annos_path + name)
        multi_channel_data, _ = librosa.core.load(track_name, constants.sampling_rate, mono=False) # _ cause dont need to reassign sampling rate
        """ loop over each channel in order to compute betas for separate and debleeded note instances """
        for channel in range(6):
            data = multi_channel_data[channel,:]
            try:
                track_instance, annotations = get_annos_for_separate_strings(data, annotation_name, constants)
                compute_track_betas(track_instance, annotations, constants, StrBetaObj, channel)
            except Exception as e:
                print(e)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('config_path', type=str)
    parser.add_argument('workspace_folder', type=str)
    parser.add_argument('-plot', action='store_true') 
    parser.add_argument('-compute', action='store_true') 
    parser.add_argument('-train', action='store_true') 

    args = parser.parse_args()

    try:
        constants = Constants(args.config_path, args.workspace_folder)
    except Exception as e:
        print(e)
    

    # HARDWIRE CONSTANTS
    constants.plot = args.plot
    # constants.compute = args.compute

    print('Check if you are OK with certain important configuration constants:')
    print('****************************')
    print('train_mode:', constants.train_mode)
    print('train_frets:', constants.train_frets)
    print('polyfit:', constants.polyfit)
    print('dataset_names_path:', constants.dataset_names_path)
    print('****************************')
    print()

    StringNames = ['E', 'A', 'D', 'G', 'B', 'e']

    betas = np.array([[None]*20]*6)
    for s in range(6):
        for n in range(20):
            betas[s,n] = []

    median_betas = np.array([[None]*20]*6)

    if args.train:
        StrBetaObj = GuitarSetTrainWrapper(constants)
        # with open('data/train/StrBetaObj.pickle', 'wb') as file:
        #     pickle.dump(StrBetaObj, file)
    else:
        with open('data/train/StrBetaObj.pickle', 'rb') as file:
            StrBetaObj = pickle.load(file)


    # compute_partial_orders(StrBetaObj, constants)
    if args.compute:
        compute_all_betas(constants, StrBetaObj)
        for s in range(6):
            for n in range(20):
                if betas[s,n]: # not None
                    median_betas[s,n] = statistics.median(betas[s,n])
        np.savez("./results/n_tonos.npz", median_betas=median_betas)

    # for s in range(6):
    #     plt.plot(range(15), median_betas[s,:], label=str(s))

    npzfile = np.load('./results/n_tonos.npz', allow_pickle=True) # allow_pickle is needed because dtype=object, since None elements exist.
    # npzfile = np.load('./results/n_tonos_hex_cln_solos.npz', allow_pickle=True) # allow_pickle is needed because dtype=object, since None elements exist.

    median_betas = npzfile['median_betas']


    nf=17
    # for s in range(6):
    #     plt.plot(range(nf), median_betas[s,:], label=str(s))

    for s in range(6):
        diff = np.log2(median_betas[s,:nf].astype(np.float64))-np.log2(median_betas[s,0].astype(np.float64))
        plt.plot(range(nf), 6*diff, label=StringNames[s], linewidth=1.75, alpha=0.85)
    plt.plot(range(nf+1), range(nf+1), label='y=x', linestyle='dashed', color='black')


    plt.xticks(range(1,20))
    plt.yticks(range(1,14))
    plt.xlim(-0.1,20)
    plt.ylim(-0.1,20)
    plt.grid()
    plt.xlabel('n')
    plt.ylabel(r'$6 \cdot \log_2(\hat{\beta}_{s,med}(n)/\hat{\beta}_{s,med}(0))$')
    plt.legend()
    plt.savefig('./results/n_tonos.png', bbox_inches='tight')
    plt.show()