from constants_parser import Constants
from random import choice
from copy import deepcopy
import random
from itertools import tee, islice
import math

from Inharmonic_Detector import determine_combinations
from track_class import *
import constants_parser

def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)

def pairwisen(iterable, n):
    x = tee(iterable, n)
    for i in range(1,n):
        for j in range(0,i):
            next(x[i], None)
    return zip(*x)

def get_random_position(string, fret, constants : Constants):
    """returns a random possible position that corresponds to a given fundamental"""
    possible_positions = [(s, f) for s in range(0,6) for f in range(0, constants.no_of_frets) if ((constants.tuning[s] + f) == (constants.tuning[string] + fret))]
    return choice(possible_positions)

def get_random_tablature(tablature : Tablature, constants : Constants):
    """make a copy of the tablature under inspection and generate new random tablatures"""
    new_tab = deepcopy(tablature)
    for tab_instance, new_tab_instance in zip(tablature.tablature, new_tab.tablature):
        if tab_instance.string == 6:
            string, fret = random.choice(determine_combinations(tab_instance.fundamental, constants))
            new_tab_instance.string, new_tab_instance.fret = string, fret
        elif constants.init_mutation_rate > random.random():
            string, fret = get_random_position(tab_instance.string, tab_instance.fret, constants)
            new_tab_instance.string, new_tab_instance.fret = string, fret
    return new_tab

def mutate(tablature : Tablature, constants : Constants):
    for tab_instance in tablature:
        if constants.mutpn < random.random():
            string, fret = get_random_position(tab_instance.string, tab_instance.fret, constants)
            tab_instance.string = string
            tab_instance.fret = fret
    return tablature

def evaluate(genetic_tablature : Tablature, inharmonic_tablature : Tablature, constants : Constants):
    weight = constants.constraints_cof*evaluate_constraints(genetic_tablature, constants) - constants.similarity_cof*evaluate_similarity(genetic_tablature, inharmonic_tablature, constants)
    return weight, 0



def similarity_cnt_basic(arg = []):
    return 1

def evaluate_similarity(genetic_tablature : Tablature, inharmonic_tablature : Tablature, constants : Constants, similarity_cnt_func = similarity_cnt_basic, cnt_args = []):
    """evaluate similarity with inharmonic based predictions. Basic functionality counts +1
     for every hit. Function can be changed if required. For example can take into 
     consideration certainty values from inharmonic analysis (cnt+= inharmonic_instance.certainty)."""
    cnt = 0
    inconclusive = 0
    for genetic_instance, inharmonic_instance in zip(genetic_tablature, inharmonic_tablature):
        if inharmonic_instance.string == 6:
            inconclusive += 1
        elif genetic_instance.string == inharmonic_instance.string:
            cnt += similarity_cnt_func(cnt_args)
    return cnt/(len(genetic_tablature) - inconclusive)


def evaluate_constraints(genetic_tablature, constants : Constants):

    open_f = open_function(genetic_tablature)
    avg = avg_function(genetic_tablature, constants)
    string_f = string_function(genetic_tablature)
    fret_f = fret_function(genetic_tablature)
    depress = depress_function(genetic_tablature)
    weight = (constants.open_cof*open_f + constants.avg_cof*avg + 
                constants.string_cof*string_f + constants.fret_cof*fret_f + 
                    constants.depress_cof*depress)/len(genetic_tablature)
    return(weight)

def depress_function(tablature : Tablature):
    depress = 0
    for previous, current in pairwise(tablature):
        if previous.string == current.string and previous.fret == current.fret:
            depress +=1
    return depress

def fret_function(tablature : Tablature):
    fret = 0
    for previous, current in pairwise(tablature):
        fret += abs(previous.fret - current.fret)
    return fret

def string_function(tablature : Tablature):
    string = 0
    for previous, current in pairwise(tablature):
        string += abs(previous.string - current.string)
    return string

def open_function(tablature : Tablature):
    open = 0
    for current in tablature: #.tablature
        if current.fret == 0:
            open -= 1
    return open

def getIndex_limits(tup, constants : Constants):
    """returns indexes that are within the time_limit set on constants file"""
    note_inspected = round((len(tup)-1)/2)
    up_index = len(tup) - 1
    down_index = 0
    for index, b in enumerate(tup[:note_inspected]):
        if abs(tup[note_inspected].onset - b.onset) > constants.time_limit:
            continue
        else:
            down_index = index
            break
    for index, b in enumerate(tup[note_inspected:]):
        if abs(tup[note_inspected].onset - b.onset) > constants.time_limit:
            up_index = note_inspected + index
            break
        else:
            continue
    return down_index, up_index

def avg_function(tablature : Tablature, constants : Constants):
    """computes difference between average position (fret,string) and current position 
    for a given time_limit and length notewise. 
    For example if time_limit is 100ms and avg_length 3 it computes 
    the average of notes (ex. 4,5,6) if abs(onset[4]-onset[5])<100ms and abs(onset[5]-onset[6])<100ms"""
    cnt = 0
    for t in pairwisen(tablature, constants.avg_length):
        note_inspected = round(len(t)/2)
        down_index, up_index = getIndex_limits(t, constants)
        mean_fret = sum([x.fret for x in t[down_index:up_index]])/(up_index-down_index + 1)
        mean_string = sum([x.string for x in t[down_index:up_index]])/(up_index-down_index + 1)
        cnt += math.sqrt((mean_fret-t[note_inspected].fret)**2+(mean_string-t[note_inspected].string)**2)
    return cnt
