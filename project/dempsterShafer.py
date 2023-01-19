import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os as os


# Dempster Shafer combination, belief, and plausibility calculations.
def combine_masses(d1, d2):
    """
    Purpose: Combine loaded, validated masses in accordance with the rules of dempster-shafer
    
    keyword arguments:
    d1 -- detection from first sensor considered of the format
    [[['a'], 0.5],[['b'], 0.3], [['a','b'], 0.2]]
    d2 -- detection from second sensor considered of the format
    [[['a'], 0.5],[['b'], 0.3], [['a','b'], 0.2]]
    
    return:
    if valid mass file, return loaded detections in the following format
    [[['a'], 0.5], [['b'], 0.3], [['a','b'], 0.2]]
    
    if not valid mass file, return error
    """
    combined_masses = []
    
    # Combine by multiplication and generate all subgroup components, including the
    # Null-space created by this fusion
    for hypothesis_d1 in d1:
        for hypothesis_d2 in d2:
            joint_list = list(set(hypothesis_d1[0]).intersection(hypothesis_d2[0]))
            if len(joint_list) == 0:
                joint_list = ['null']
                    
            if joint_list in [x[0] for x in combined_masses]:
                for i in combined_masses:
                    if i[0] == joint_list:
                        i[1] += round(hypothesis_d1[1] * hypothesis_d2[1],4)


            else:
                combined_masses.append([joint_list, round(hypothesis_d1[1] * hypothesis_d2[1],4)])

    ##Scale by null accordingly if it exists/has been created during fusion
    if ['null'] in [i[0] for i in combined_masses]:
        K = 1 - round([i[1] for i in combined_masses if i[0] == ['null']][0],4)
    else: 
        K = 1
    combined_masses = [[i[0],round(i[1]/K,6)] for i in combined_masses if i[0] != ['null']]
    
    if round(check_sums(combined_masses),2) == 1:
        
        return combined_masses

    else:
        return """an error has occurred, mass fusion has not returned a result of 1. 
                For dempster-shafer to run, it must hold that masses sum to 1. Your
                mass fusion has returned: """+ str(round(check_sums(combined_masses),4))


def check_sums(detections):
    """
    Purpose: Confirm that the masses within the text file sum to 1. 
    
    keyword arguments: 
    detections -- loaded detection nested list of the following format
    [[['a'], 0.5],[['b'], 0.3], [['a','b'], 0.2]]
    
    return:
    validate -- return whether or not this detection list is valad (aka the masses sum to 1)
    """
    
    return sum([i[1] for i in detections])

    
def get_belief(mass):
    """
    Purpose: For a given set of masses, get the belief - 
    the lower bound of probability, such that it is the sum of all
    masses where B is a subset of A
    
    keyword arguements:
    
    mass -- the mass for the weights you would like to get DS 
    style belief for
    
    return: the belief for DS
    """
    beliefs = []
    
    for i in range(0, len(mass)):
        subset_calc = [k[1] for k in mass if set(k[0]).issubset(set(mass[i][0]))]
        belief = round(sum(subset_calc),4)
        beliefs.append([mass[i][0], belief])
        
    return beliefs


def get_plausibility(mass):
    """
    Purpose: For a given set of masses, get the belief - 
    the lower bound of probability, such that it is the sum of all
    masses where B is a subset of A
    
    keyword arguements:
    
    mass -- the mass for the weights you would like to get DS 
    style plausibility for
    
    return: the belief for DS
    """
    plausibilities = []
    
    for i in range(0, len(mass)):
        subset_calc = [k[1] for k in mass if set(k[0]).intersection(set(mass[i][0]))]
        plausibility = round(sum(subset_calc),4)
        plausibilities.append([mass[i][0], plausibility])
        
    return plausibilities


def get_output(mass):
    belief = get_belief(mass)
    plausibility = get_plausibility(mass)
    together = [[mass[i][0], mass[i][1], belief[i][1], plausibility[i][1]] for i in range(0, len(mass))]
    return pd.DataFrame(together, columns = ['hypothesis', 'mass', 'belief', 'plausibility'])


# Helpful math functions for preparing datasets for DST.
def powerset(list):
    """
    Purpose: For the given set of items, get the powerset of the items
    such that the powerset of set s is the set of all subsets of s,
    including the empty set and s itself, with space for hypothesis
    
    For More Information on the Algorithm's use of bitwise operations see:
    https://stackoverflow.com/questions/41629583/if-counter-1j-what-does-this-statement-mean-and-how-it-works/41629671

    
    keyword arguements:
    
    s -- the set for which to return the powerset
    
    return: the powerset of s and field for hypothesis
    """
    output = []

    n = len(list)
    for i in range(2**n):
        val = [list[j] for j in range(n) if (i & (1 << j))]
        initial = [val, 0.0]
        output.append(initial)
    return output


def class_range_output(dataset, hypotheses, field_index_names = []):
    """
    Purpose: For the given set of items, find the frame of discernment -
    for determining class membership by the attribute’s minimum and maximum values

    keyword arguements:
    
    dataset -- the set of which we will extract the attributes
    hypothesis -- the range of items in the dataset which are hypotehsized
    filed_index_names -- the names of the attriubte fields in the dataset
    
    return: the frame of discernment for the hypotheses
    """
    class_range = {}

    for hypothesis in hypotheses:
        field_range = {}
        for f in field_index_names:
            field_range[f] = (dataset[hypotheses == hypothesis][f].min(), dataset[hypotheses == hypothesis][f].max())
        class_range[hypothesis] = field_range
    return class_range


# Helpful functions for initializing hypotheses.
def hypothesis(hypotheses, class_range, field_name, value):
    """
    Purpose: For the given set of values within a class range,
    provide a grouping for the data, and classify it into a hypothesis
    
    keyword arguements:
    
    hypotheses -- the set of potential hypotheses
    class_range -- the range of discernment for the hypotheses
    field_name -- the name of the attribute for the hypotheses
    value -- the value of the attribute for the hypotheses

    return: the set of possible hypotheses
    """
    hset = []
    for c in hypotheses.unique():
        if (class_range[c][field_name][0] <= value and value < class_range[c][field_name][1]):
            hset.append(c)
    return hset


def hypothesis_counts(hypothesis_count, h):
    """
    Purpose: For the powerset of the hypotheses, iteratively count
    the likelihood of each hypothesis falling within a certain set

    keyword arguements:
    
    hypotheses_count -- the powerset of the hypotheses with the likelihood value
    h -- the set of possible hypotheses
    
    return: the likelihood value of each hypothesis
    """
    if len(h) == 1 or len(h) == 2:
        for i in range(0, len(hypothesis_count)):
            if(hypothesis_count[i][0] == h):
                hypothesis_count[i][1] += 0.9
                hypothesis_count[-1][1] += 0.1
    else:
        hypothesis_count[-1][1] += 1