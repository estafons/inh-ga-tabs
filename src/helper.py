import numpy as np
from pathlib import Path
import itertools
import matplotlib.pyplot as plt
from sympy import symbols, solveset, S
from sympy import Poly

from track_class import Annotations
import math
from Inharmonic_Detector import StringBetas

from playsound import playsound
import soundfile as sf

class ConfusionMatrix():
    def __init__(self, size, inconclusive):
        self.matrix = np.zeros(size)
        self.current_matrix = np.zeros(size)
        self.x_classes = ['E', 'A', 'D', 'G', 'B', 'e']
        if inconclusive:
            self.y_classes = ['E', 'A', 'D', 'G', 'B', 'e', 'inconclusive']
        else:
             self.y_classes = ['E', 'A', 'D', 'G', 'B', 'e']
    
    # def reinit_current_matrix():
    #     self.current_matrix = np.zeros(size)
    
    def get_current_accuracy(self, constants):
        current_acc = np.trace(self.current_matrix)/np.sum(self.current_matrix)

        if not constants.run_genetic_alg:
            inconclusive_rate = np.sum(self.current_matrix, axis = 0)[6]/np.sum(self.current_matrix)
            if constants.verbose:
                print("inconclusive rate is {} and pure accuracy {}".format(inconclusive_rate, 
                                                        np.trace(self.current_matrix)/(np.sum(self.current_matrix)-np.sum(self.current_matrix, axis = 0)[6])))
        return current_acc

    def get_accuracy(self, ga=False):
        total_acc = np.trace(self.matrix)/np.sum(self.matrix)
        # if not hasattr(constants, 'run_genetic_alg') or not constants.run_genetic_alg:
        if not ga:
            inconclusive_rate = np.sum(self.matrix, axis = 0)[6]/np.sum(self.matrix)
            # print("inconclusive rate is {} and pure accuracy {}".format(inconclusive_rate, 
                                                        # np.trace(self.matrix)/(np.sum(self.matrix)-np.sum(self.matrix, axis = 0)[6])))
        else:
            inconclusive_rate = None

        return total_acc, inconclusive_rate

    def add_to_track_predictions_to_matrix(self, tab_as_list, annotations : Annotations):
        for tab_instance, annos_instance in zip(tab_as_list, annotations.tablature.tablature):
            self.matrix[annos_instance.string][tab_instance.string] += 1
            self.current_matrix[annos_instance.string][tab_instance.string] += 1
        
    def plot_confusion_matrix(self, constants,
                            normalize=False,
                            title='Confusion matrix',
                            cmap=plt.cm.Blues):
            """
            This function prints and plots the confusion matrix.
            Normalization can be applied by setting `normalize=True`.
            """
            plt.clf()
            if normalize:
                self.matrix = self.matrix.astype('float') / self.matrix.sum(axis=1)[:, np.newaxis]
                np.nan_to_num(self.matrix,False)
                print("Normalized confusion matrix")
            else:
                print('Confusion matrix, without normalization')
            plt.imshow(self.matrix, interpolation='nearest', cmap=cmap)
            plt.title(title)
            plt.colorbar()
            
            tick_marks_y = np.arange(len(self.y_classes))
            tick_marks_x = np.arange(len(self.x_classes))
            plt.xticks(tick_marks_x,self.x_classes , rotation=45)
            plt.yticks(tick_marks_y, self.y_classes)

            fmt = '.2f' if normalize else '.2f'
            thresh = self.matrix.max() / 2.
            for i, j in itertools.product(range(self.matrix.shape[0]), range(self.matrix.shape[1])):
                plt.text(j, i, format(self.matrix[i, j], fmt),
                            horizontalalignment="center",
                            color="white" if self.matrix[i, j] > thresh else "black")

            plt.ylabel('True label')
            plt.xlabel('Predicted label')
            plt.tight_layout()
            #plt.show()
            # if constants.dataset and constants.train_mode and constants.polyfit: # i.e. if GuitarSet
            #     plt.savefig(Path(constants.result_path + title.replace(" ", "") +'_'+constants.dataset+'_'+constants.train_mode+'_'+constants.polyfit+'.png'))
            # elif constants.train_mode and constants.polyfit:
            if hasattr(constants, 'dataset'):   
                plt.savefig(Path(constants.result_path + title.replace(" ", "") +'_'+constants.train_mode+'_'+constants.polyfit+'_'+constants.dataset+'.png'))
            else:
                plt.savefig(Path(constants.result_path + title.replace(" ", "") +'_'+constants.train_mode+'_'+constants.polyfit+'.png'))

            return plt


def compute_partial_orders(StrBetaObj : StringBetas, constants):
    StrBetaObj.beta_lim = [[] for x in range(0, constants.tuning[-1] + constants.no_of_frets - 40)]
    for string, open_midi in enumerate(constants.tuning):
        for fret in range(0,constants.no_of_frets):
            beta = StrBetaObj.betas_array[string][0]*2**(fret/6)
            k = symbols('k')
            sol = solveset(beta*k**4 -k -1/2<0, k, S.Reals)
            StrBetaObj.beta_lim[open_midi + fret - 40].append(math.floor(sol.end))
    print(StrBetaObj.beta_lim)

# Print iterations progress
# https://stackoverflow.com/questions/3173320/text-progress-bar-in-the-console/30740258
def printProgressBar(iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
    # Print New Line on Complete
    if iteration == total: 
        print()

def listen_to_the_intance(audio):
    sf.write('tmp.wav', audio, 44100, 'PCM_16')
    playsound('tmp.wav')
