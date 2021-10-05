
import math
import librosa
from constants_parser import Constants
from inharmonic_Analysis import NoteInstance, ToolBox, compute_partials, compute_inharmonicity
import numpy as np

class StringBetas():
    def __init__(self, barray, constants : Constants):
        self.betas_array = barray #0->string 1->fret
        self.betas_list_array = [[[] for x in range(0,constants.no_of_frets)] for i in range(0,len(constants.tuning))]

    def add_to_list(self, note_instance):
        self.betas_list_array[note_instance.string][note_instance.fret].append(note_instance.beta)
        #self.betas_list_array.remove(0)
    
    def input_instance(self, instance_audio, midi_note, string, constants : Constants):
        fundamental = librosa.midi_to_hz(midi_note)
        ToolBoxObj = ToolBox(compute_partials, compute_inharmonicity, [constants.no_of_partials, fundamental/2, constants], [])
        note_instance = NoteInstance(fundamental, 0, instance_audio, ToolBoxObj, constants.sampling_rate, constants, midi_flag = True)
        fundamental = note_instance.recompute_fundamental(constants)
        note_instance = NoteInstance(fundamental, 0, instance_audio, ToolBoxObj, constants.sampling_rate, constants) # compute again with recomputed fundamental
        ToolBoxObj = ToolBox(compute_partials, compute_inharmonicity, [constants.no_of_partials, fundamental/2, constants], [])
        note_instance.string = string
        note_instance.fret = midi_note - constants.tuning[note_instance.string]
        return note_instance  

    def list_to_medians(self):
        for i, r in enumerate(self.betas_list_array):
            for j, l in enumerate(r):
                self.betas_array[i][j] = np.median(l)

    def set_limits(self, constants):
        max_beta = 0
        for string, s in enumerate(constants.tuning):
            if self.betas_array[string][0] > max_beta:
                max_beta = self.betas_array[string][0]

        constants.upper_limit =  max_beta*2**((constants.no_of_frets+2)/6) # +2 fret margin. didnt ceil up to the nearest, because sometimes 10**-3 appear ceiling up to 10**-2 making a huge difference
        constants.lower_limit = 10**(math.floor(math.log(np.nanmin(self.betas_array), 10)))

class InharmonicDetector():
    def __init__(self, NoteObj : NoteInstance, StringBetasObj : StringBetas):
        self.StringBetasObj = StringBetasObj


def DetectString(NoteObj : NoteInstance, StringBetasObj : StringBetas, betafunc, constants : Constants):
    """ betafunc is the function to simulate beta. As input takes the combination and the beta array."""
    combs = determine_combinations(NoteObj.fundamental, constants)
    if (constants.lower_limit < NoteObj.beta < constants.upper_limit):
        betas = [(abs(betafunc(comb, StringBetasObj, constants) - NoteObj.beta), comb) for comb in combs]
        NoteObj.string = min(betas, key = lambda a: a[0])[1][0] # returns comb where 0 argument is string
    else:
        NoteObj.string = 6

def hz_to_midi(fundamental):
    return round(12*math.log(fundamental/440,2)+69)

def midi_to_hz(midi):
    return 440*2**((midi-69)/12)

def determine_combinations(fundamental, constants):
    res = []
    midi_note = hz_to_midi(fundamental)
    fretboard = [range(x, x + constants.no_of_frets) for x in constants.tuning]
    for index, x in enumerate(fretboard):
        if midi_note in list(x):
            res.append((index, midi_note-constants.tuning[index])) # index is string, second is fret
    try:
        assert(res == []), "No combinations found"
    except AssertionError:
        pass
    return res

