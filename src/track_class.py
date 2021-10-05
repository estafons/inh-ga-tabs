from constants_parser import Constants
import Inharmonic_Detector

class Tablature():
    def __init__(self, tups, audio, constants : Constants, annotation = False):
        self.tablature = []
        note_audio = []
        for x in tups:
            if annotation == False:
                start = int(round((x[0])*(constants.sampling_rate)))
                end = int(round((x[0]+constants.crop_win)*(constants.sampling_rate)))
                note_audio = audio[start:end]
            self.tablature.append(TabInstance(x, note_audio, constants))

    def __getitem__(self, item):
        '''added so crossover functions from deap can be incorporated easily'''
        return self.tablature[item]

    def __len__(self):
        return len(self.tablature)

class TrackInstance():
    def __init__(self, tablature : Tablature, audio, constants : Constants): # tup gets the onset -fundamental tuple
        """main class. crop_win is the length 
        of the excerpts we will consider. 
        if we consider 60ms windows then all note 
        instances will be cropped at [onset, onset+60ms]"""
        self.tablature = tablature
        self.audio = audio
        self.sampling_rate = constants.sampling_rate
        #self.crop_win = crop_win

class TabInstance():
    def __init__(self, tup, note_audio, constants : Constants):
        self.onset, self.fundamental, self.string = tup
        if self.string in list(range(0,5)):
            self.fret = Inharmonic_Detector.hz_to_midi(self.fundamental) - constants.tuning[self.string]
        self.note_audio = note_audio

class Annotations():
    def __init__(self, tups, constants : Constants):
        self.tablature = Tablature(tups, [], constants, annotation = True)
