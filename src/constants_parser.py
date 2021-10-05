import configparser
from pathlib import Path

import betafuncs

def is_string(value):
    try:
        str(value)
        return True
    except(ValueError):
        return False

def is_float(value):
    try:
        float(value)
        return True
    except(ValueError):
        return False

def is_int(value):
    try:
        int(value)
        return True
    except(ValueError):
        return False

def is_list_of_int(value):
    try:
        [int(x) for x in value.split(", ")]
        return True
    except(ValueError):
        return False

class Constants():
    def __init__(self, config_name, workspace_folder):
        config = configparser.ConfigParser()
        config.read(config_name)
        self.workspace_folder = workspace_folder
        for section_name in config.sections():
            for key, value in config.items(section_name):
                if str(key) == 'size_of_fft':
                    setattr(self, key, 2**(int(value)))
                elif is_int(value):
                    value = int(value)
                    setattr(self, key, value)
                elif is_float(value):
                    value = float(value)
                    setattr(self, key, value)
                elif is_list_of_int(value):
                    value = [int(x) for x in value.split(", ")]
                    setattr(self, key, value)
                elif is_string(value):
                    setattr(self, key, value)
                else:
                    raise ValueError("constants.ini arguement with name " + str(key) + "is of innapropriate value."+
                         "chansge value or suplement Constants class in constants_parser.py")    
        self.track_path = workspace_folder + '/data/audio/'
    # Path were training data are stored
        self.training_path = workspace_folder + '/data/train/'
        # Path were results should be stored
        # self.result_path = workspace_folder + '/data/results/'
        self.result_path = workspace_folder + '/results/'
        # Path were annotations are stored
        self.annos_path = workspace_folder + '/data/annos/'
        # txt file that contains the list of names of the tracks to be tested (GuitarSet)
        self.listoftracksfile = '/names.txt'
        # Path were listoftracksfile is stored NOTE: commented out!!
        # self.dataset_names_path = workspace_folder
        self.train_frets_copy = self.train_frets.copy()
        self.update_betafunc()
    def update_betafunc(self):
        if self.train_mode == '1Fret':
            self.betafunc = betafuncs.betafunc
            self.train_frets = [self.train_frets_copy[0]]
            assert len(self.train_frets) > 0
        elif self.train_mode == '2FretA':
            self.betafunc = betafuncs.expfunc
            self.train_frets = [self.train_frets_copy[0], self.train_frets_copy[-1]]
            assert len(self.train_frets) >= 2
        if self.train_mode == '2FretB':
            self.betafunc = betafuncs.linfunc
            self.train_frets = [self.train_frets_copy[0], self.train_frets_copy[-1]]
            assert len(self.train_frets) >= 2
        elif self.train_mode == '3Fret':
            self.betafunc = betafuncs.aphfunc
            self.train_frets = self.train_frets_copy[0:3].copy()
            assert len(self.train_frets) >= 3
        '''#PATHS
        self.track_path = config.get('GUITARSET_PATHS', 'track_path')
        self.training_path = config.get('GUITARSET_PATHS', 'training_path')
        self.result_path = config.get('GUITARSET_PATHS', 'result_path')
        self.annos_path = config.get('GUITARSET_PATHS', 'annos_path')
        self.listoftracksfile = config.get('GUITARSET_PATHS', 'listoftracksfile')
        self.dataset_names_path = config.get('GUITARSET_PATHS', 'dataset_names_path')
        self.dataset = config.get('GUITARSET_PATHS', 'dataset')
        #INHARMONICITY
        self.tuning = [int(x) for x in config.get('INHARMONICITY', 'tuning').split(", ")]
        self.no_of_frets = config.getint('INHARMONICITY', 'no_of_frets')
        self.sampling_rate = config.getint('INHARMONICITY', 'sampling_rate')
        self.size_of_fft = 2**config.getint('INHARMONICITY', 'size_of_fft')
        self.crop_win = config.getfloat('INHARMONICITY', 'crop_win')
        self.no_of_partials = config.getint('INHARMONICITY', 'no_of_partials')
        #GENETIC
        self.initial_pop = config.getint('GENETIC', 'initial_pop')
        self.init_mutation_rate = config.getfloat('GENETIC', 'init_mutation_rate')
        self.tournsize = config.getint('GENETIC', 'tournsize')
        self.no_of_parents = config.getint('GENETIC', 'no_of_parents')
        self.ngen = config.getint('GENETIC', 'ngen')
        self.cxpb = config.getfloat('GENETIC', 'cxpb')
        self.mutpb = config.getfloat('GENETIC', 'mutpb')
        self.mutpn = config.getfloat('GENETIC', 'mutpn')
        self.parents_to_next_gen = config.getint('GENETIC', 'parents_to_next_gen')
        self.offspring_to_next_gen = config.getint('GENETIC', 'offspring_to_next_gen')
        self.end_no = config.getint('GENETIC', 'end_no')

        #CONSTRAINTS
        self.constraints_cof = config.getfloat('CONSTRAINTS', 'constraints_cof')
        self.similarity_cof = config.getfloat('CONSTRAINTS', 'similarity_cof')
        self.open_cof = config.getfloat('CONSTRAINTS', 'open_cof')
        self.avg_cof = config.getfloat('CONSTRAINTS', 'avg_cof')
        self.string_cof = config.getfloat('CONSTRAINTS', 'string_cof')  
        self.fret_cof = config.getfloat('CONSTRAINTS', 'fret_cof')
        self.depress_cof = config.getfloat('CONSTRAINTS', 'depress_cof')
        self.avg_length = config.getint('CONSTRAINTS', 'avg_length')
        self.time_limit = config.getfloat('CONSTRAINTS', 'time_limit')

        #TRAINING
        self.train_frets = [int(x) for x in config.get('TRAINING', 'train_frets').split(", ")]'''




