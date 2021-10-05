
from pathlib import Path
import argparse
import os, sys
import numpy as np
import librosa
BASE_PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# cur_path = Path(BASE_PATH + '/src/InharmonicStringDetection')
cur_path = Path(BASE_PATH + '/src/')
sys.path.append(str(cur_path))
cur_path = Path(BASE_PATH + '/exps')
sys.path.append(str(cur_path))

from Inharmonic_Detector import StringBetas, Constants
from helper import printProgressBar


def HjerrildChristensenTrain(strBetaObj, constants : Constants, train_frets = [0]):
    if constants.guitar == 'firebrand':
        dataset_nums = [1,2,3,5,6,7,8,9,10]
    elif constants.guitar == 'martin':
        dataset_nums = [1,2,3,4,5,6,7,8,9,10]
    print('Training on the specified subset...')
    count=0    
    for dataset_no in dataset_nums:
        # print(dataset_no)
        for string in range(0,6):
            # print(dataset_no, string)
            printProgressBar(count,6*len(dataset_nums),decimals=0, length=50)
            for fret in constants.train_frets:
                midi = constants.tuning[string] + fret
                path_to_track = Path(constants.workspace_folder + '/' + constants.path_to_hjerrild_christensen +
                                     constants.guitar + str(dataset_no) + 
                                            '/string' +str(string + 1) +'/' + str(fret) +'.wav')
                audio, _ = librosa.load(path_to_track, constants.sampling_rate)
                y = librosa.onset.onset_detect(audio, constants.sampling_rate)
                audio = audio[y[0]:] # adding this line because not all tracks start at the beggining
                note_instance = strBetaObj.input_instance(audio, midi, string, constants)
                strBetaObj.add_to_list(note_instance)
            count+=1

def TrainWrapper(constants : Constants):
    strBetaObj = StringBetas(np.zeros((len(constants.tuning), constants.no_of_frets)), constants)
    HjerrildChristensenTrain(strBetaObj, constants, train_frets = constants.train_frets)
    strBetaObj.list_to_medians()
    # print(strBetaObj.betas_array)
    strBetaObj.set_limits(constants)
    print('Acceptable range of beta values:', constants.upper_limit, '-', constants.lower_limit)
    print()
    return strBetaObj

#TrainWrapper(constants)