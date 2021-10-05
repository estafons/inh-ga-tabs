from pathlib import Path
import os, sys
import librosa
import argparse
import numpy as np

BASE_PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# cur_path = Path(BASE_PATH + '/src/InharmonicStringDetection')
cur_path = Path(BASE_PATH + '/src/')
sys.path.append(str(cur_path))
cur_path = Path(BASE_PATH + '/exps')
sys.path.append(str(cur_path))

from track_class import *
from helper import ConfusionMatrix, compute_partial_orders, printProgressBar

from HjerrildTrain import TrainWrapper
from inharmonic_Analysis import (compute_partials, compute_inharmonicity, NoteInstance, 
									ToolBox, compute_partials_with_order, 
										compute_partials_with_order_strict)
import Inharmonic_Detector


import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument('config_path', type=str)
parser.add_argument('workspace_folder', type=str)
parser.add_argument('-plot', action='store_true') 
parser.add_argument('--guitar', type=str, default= '')
parser.add_argument('--train_mode', type=str, default= '')
args = parser.parse_args()



try:
	os.mkdir('./results')
except Exception as e:
	print('GootToKnow:', e)

try:
	constants = Constants(args.config_path, args.workspace_folder)
except Exception as e:
	print(e)

constants.plot = args.plot
if args.guitar:
    constants.guitar = args.guitar
if args.train_mode:
    constants.train_mode = args.train_mode
    constants.update_betafunc()



def testHjerrildChristensen(constants : Constants, StrBetaObj):
	InhConfusionMatrixObj = ConfusionMatrix((6,7), inconclusive = True)
	if constants.guitar == 'firebrand':
		dataset_nums = [1,2,3,5,6,7,8,9,10]
	elif constants.guitar == 'martin':
		dataset_nums = [1,2,3,4,5,6,7,8,9,10]
	print("Testing to the training set complement...")
	count=0
	for dataset_no in dataset_nums:
		# print(dataset_no)
		for string in range(0,6):
			printProgressBar(count,6*len(dataset_nums)-1,decimals=0, length=50)
			for fret in range(0,12):
				path_to_track = Path(constants.workspace_folder + '/'+ constants.path_to_hjerrild_christensen +
									 constants.guitar + str(dataset_no) + 
											'/string' +str(string + 1) +'/' + str(fret) +'.wav')
				rec_audio, _ = librosa.load(path_to_track, constants.sampling_rate)

				# Onset detection (instances not always occur at the beginning of the recording)
				# y = librosa.onset.onset_detect(audio, constants.sampling_rate)
				# audio = audio[int(y[0]):] # adding this line because ther5e might be more than one onsets occurring in the recording

				path_to_onsettime = Path(constants.workspace_folder + '/data/onsets/' +
									 constants.guitar + str(dataset_no) + 
											'/string' +str(string + 1) +'/' + str(fret) +'.txt')

				with open(path_to_onsettime, 'r') as f:
					onsetsec = float(f.read())
					onsetidx = int(onsetsec*constants.sampling_rate)
					plus60idx = int(0.06*constants.sampling_rate)
					# print(onsetidx, plus60idx)
				# audio = audio[onsetidx:(onsetidx+plus60idx)] # adding this line because ther5e might be more than one onsets occurring in the recording
				audio60ms = rec_audio[onsetidx:(onsetidx+plus60idx)] # restricted to 60ms
				longaudio = rec_audio[onsetidx:] # not restricted to 60ms


			
				# Better fundamental estimation (TODO: use librosa.pyin instead, delete next line and se midi_flag=False to avoid f0 re-compute)
				fundamental_init = librosa.midi_to_hz(constants.tuning[string] + fret)
				if constants.f0again == 'external': 
					# librosa pyin fundamental estimation
					f0s, _, _ = librosa.pyin(audio60ms, fmin=80, fmax=1500, sr=constants.sampling_rate)
					f0s = f0s[~np.isnan(f0s)]

					# ### simple method #### __gb__
					# f0_id = np.argmin(np.abs(f0s-fundamental_init))
					# f0 = f0s[f0_id]

					### more sophisticated method i.e. mean of 10 closest estimations to expected value #### __gb__
					idx = np.argsort(np.abs(f0s-fundamental_init))
					f0 = np.sum( f0s[idx[:10]] )/10
				
				# NOTE: below commented is Stef's alternative method for variable partial numebers to be employed
				# ToolBoxObj = ToolBox(partial_tracking_func=compute_partials_with_order_strict, inharmonicity_compute_func=compute_inharmonicity, 
				#                 partial_func_args=[fundamental_init/2, constants, StrBetaObj], inharmonic_func_args=[])
				# TODO: make inharmonicity_compute_func have a meaning, also 1st arg of partial_func_args
				ToolBoxObj = ToolBox(partial_tracking_func=compute_partials, inharmonicity_compute_func=compute_inharmonicity, 
								partial_func_args=[constants.no_of_partials, fundamental_init/2, constants, StrBetaObj], inharmonic_func_args=[])
				if constants.f0again == 'internal':
					note_instance = NoteInstance(fundamental_init, 0, audio60ms, ToolBoxObj, constants.sampling_rate, constants, longaudio=longaudio, midi_flag=True)
				elif constants.f0again == 'external': 
					note_instance = NoteInstance(f0, 0, audio60ms, ToolBoxObj, constants.sampling_rate, constants)
				elif constants.f0again == 'no': 
					note_instance = NoteInstance(fundamental_init, 0, audio60ms, ToolBoxObj, constants.sampling_rate, constants)
				else:
					print("Wrong arguments given for f0again. Add a valid value in the corresponding field of constants.ini: 'interanal' or 'external' or 'no'")
					exit(1)
				# Detect plucked string (i.e. assigns value to note_instance.string)
				# try:
				Inharmonic_Detector.DetectString(note_instance, StrBetaObj, constants.betafunc, constants)
				# except ValueError:
				# 	note_instance = NoteInstance(fundamental_init, 0, audio60ms, ToolBoxObj, constants.sampling_rate, constants)
				# 	Inharmonic_Detector.DetectString(note_instance, StrBetaObj, constants.betafunc, constants)
				# 	print('Problem with fundamental estimation')

				# Compute Confusion Matrix
				InhConfusionMatrixObj.matrix[string][note_instance.string] += 1
			count+=1
	# print(InhConfusionMatrixObj.get_accuracy())			
	print('Accuracy:', round(InhConfusionMatrixObj.get_accuracy(),3))
	InhConfusionMatrixObj.plot_confusion_matrix(constants, normalize= True, 
													title = str(constants.guitar) + str(constants.no_of_partials) +
														'Inharmonic Confusion Matrix' +
														str(round(InhConfusionMatrixObj.get_accuracy(),3)))
														# str(round(InhConfusionMatrixObj.get_current_accuracy(),3))

if __name__ == '__main__':
	print('Check if you are OK with certain important configuration constants:')
	print('****************************')
	print('train_mode:', constants.train_mode)
	print('train_frets:', constants.train_frets)
	print('polyfit:', constants.polyfit)
	print('f0again:', constants.f0again)
	print('guitar:',constants.guitar)
	print('****************************')
	print()

	StrBetaObj = TrainWrapper(constants)
	# compute_partial_orders(StrBetaObj, constants)
	# StrBetaObj =None

	testHjerrildChristensen(constants, StrBetaObj)