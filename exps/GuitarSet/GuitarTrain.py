
import numpy as np
from pathlib import Path
import glob
import librosa
import os, sys

BASE_PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
cur_path = Path(BASE_PATH + '/src/')
sys.path.append(str(cur_path))

from helper import printProgressBar
from inharmonic_Analysis import *
from Inharmonic_Detector import StringBetas
from constants_parser import Constants
from playsound import playsound

def train_GuitarSet(strBetaObj, constants, train_frets = [0]):
    '''train on selected instances from guitarset saved in specified folder. Fundamental 
    frequency is recomputed (since it is not available otherwise) by computing the expected
     fundamental based on midi note and finding the highest peak in a +-10 hz frame'''
    print('Training on the specified subset...')
 
    for index, open_midi in enumerate(constants.tuning):
        printProgressBar(index,len(constants.tuning),decimals=0, length=50)
        for train_fret in train_frets:
            midi_train = open_midi + train_fret
            path_to_train_data = str(Path(constants.training_path + str(midi_train) + "/" +str(index) + "/*.wav"))
            list_of_names = glob.glob(path_to_train_data)
            for note_name in list_of_names:
                note_audio, _ = librosa.load(note_name, constants.sampling_rate)
                note_instance = strBetaObj.input_instance(note_audio, midi_train, index, constants)
                strBetaObj.add_to_list(note_instance)
    

def GuitarSetTrainWrapper(constants):
    strBetaObj = StringBetas(np.zeros((len(constants.tuning), constants.no_of_frets)), constants)
    train_GuitarSet(strBetaObj, constants, train_frets = constants.train_frets)
    strBetaObj.list_to_medians()
    strBetaObj.set_limits(constants)
    if hasattr(constants, 'verbose') and constants.verbose:
        print()
        print('Beta estimations:')
        print(strBetaObj.betas_array)
        print()
    print('Acceptable range of beta values:', constants.upper_limit, '-', constants.lower_limit)
    print()

    return strBetaObj