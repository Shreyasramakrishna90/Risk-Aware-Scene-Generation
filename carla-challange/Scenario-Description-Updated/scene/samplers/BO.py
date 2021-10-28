#!/usr/bin/python3
import textx
import numpy as np
import lxml.etree
import lxml.builder
import sys
import glob
import os
from xml.etree import ElementTree
import xml.etree.ElementTree as ET
from textx.metamodel import metamodel_from_file
import utils
import csv
import argparse
from argparse import RawTextHelpFormatter
import pandas as pd
import random
from sklearn.utils import shuffle
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.preprocessing import StandardScaler
from itertools import product
from statistics import mean,median
from ast import literal_eval
#from scene_interpreter import sampling_rules, compositional_rules

weather_parameters = ['cloudiness','precipitation','precipitation_deposits','sun_altitude_angle','wind_intensity','sun_azimuth_angle','wetness','fog_distance','fog_density']

def sampling_rules(sample):
    """
    Describes the sampling rules for selecting the weather samples
    The weather samples can only gradually increase in steps rather than having big jumps
    """
    #print(sample)
    if sample[0] in weather_parameters:
        min,max = (int(sample[1])-5,int(sample[1])+5)
        if min < 0:
            min = 0
        if max > 100:
            max = 100
        step = 5.0
    elif sample[0] == "road_segments":
        if sample[1] == 10:
            print("I am the problem")
            min,max = (0,2)
        else:
            min,max = (int(sample[1]),int(sample[1])+2)
        step = 1.0

    return min, max, step


def vector_2d(array):
    return np.array(array).reshape((-1, 1))

def gaussian_process(parameters, scores, x1x2, parameter_length):
    parameters = np.array(parameters).reshape((-1,parameter_length))
    #print(parameters)
    scores = vector_2d(scores)
    #print(scores)
    x1x2 = np.array(x1x2).reshape((-1,parameter_length))
    # Train gaussian process
    kernel = C(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2))
    gp = GaussianProcessRegressor(kernel, n_restarts_optimizer=5000)
    gp.fit(parameters,scores)
    y_mean, y_std = gp.predict(x1x2, return_std=True)
    y_std = y_std.reshape((-1, 1))

    return y_mean, y_std

def next_parameter_by_ei(y_max,y_mean, y_std, x1x2):
    # Calculate expected improvement from 95% confidence interval
    expected_improvement = (y_mean + 1.96 * y_std) - y_max
    #expected_improvement = y_min - (y_mean - 1.96 * y_std)
    expected_improvement[expected_improvement < 0] = 0
    max_index = expected_improvement.argmax()
    # Select next choice
    next_parameter = x1x2[max_index]
    print(next_parameter)
    return next_parameter

def read_parameter_file(file):
    """
    Read csv files
    """
    data = []
    with open(file, newline='') as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) > 1:
                row_data = []
                for i in range(len(row)):
                    row_data.append(float(row[i]))
                data.append(row_data)
            elif len(row) == 1:
                data.append(row)

    return data


def read_previous_stats(collision_file,stats_file,scenario_score_file):
    """
    Read Stats file to return collisions, martingales and risk
    """
    collisions = []
    scenario_scores = []
    martingales = []
    risks = []
    collision = pd.read_csv(collision_file, usecols = [0], header=None, index_col=False)
    for index, row in collision.iterrows():
        collisions.append(int(row))
    martingale = pd.read_csv(stats_file, usecols = [0], header=None, index_col=False)
    for index, row in martingale.iterrows():
       martingales.append(float(row))
    scenario_score = pd.read_csv(scenario_score_file, usecols = [0], header=None, index_col=False)
    for index, row in scenario_score.iterrows():
        scenario_scores.append(int(row))
    risk = pd.read_csv(stats_file, usecols = [1], header=None, index_col=False)
    for index, row in risk.iterrows():
       risks.append(float(row))
    #print(risks)

    return collisions, martingales, risk, scenario_score


def cartesian_product(*arrays):
    la = len(arrays)
    dtype = np.result_type(*arrays)
    arr = np.empty([len(a) for a in arrays] + [la], dtype=dtype)
    for i, a in enumerate(np.ix_(*arrays)):
        arr[...,i] = a
    return arr.reshape(-1, la)

# def get_sample_choice(current_hyperparameters,simulation_run):
#     """
#     Get choices of the sample array
#     """
#     choices_array = []
#     distributions = []
#     for i in range(len(current_hyperparameters)):
#         if current_hyperparameters[i][0] == 'sun_altitude_angle':
#             step = 45
#         elif current_hyperparameters[i][0] == 'segment_number' or 'traffic':
#             step = 1
#         else:
#             step = 5
#         choice_list = np.arange(current_hyperparameters[i][1],current_hyperparameters[i][2],step)
#         choices_array.append(choice_list)
#         parameter_distribution = random.choice(choice_list)
#         distributions.append(int(parameter_distribution))
#     #print(distributions)
#
#     return choices_array, parameter_distribution, distributions

def get_vals(current_hyperparameters,trial_runs,simulation_run):
    """
    Narrowing the random search for better training the search space
    """
    if simulation_run <= trial_runs:
        print("1")
        print(current_hyperparameters[0])
        if current_hyperparameters[0] in weather_parameters:
            if current_hyperparameters[0] != "sun_altitude_angle":
                min,max = 0,25
            elif current_hyperparameters[0] == "sun_altitude_angle":
                min,max = 20,45

    elif simulation_run > trial_runs and simulation_run <= 2*trial_runs:
        if current_hyperparameters[0] in weather_parameters:
            if current_hyperparameters[0] != "sun_altitude_angle":
                min,max = 25,50
            elif current_hyperparameters[0] == "sun_altitude_angle":
                min,max = 0,20

    elif simulation_run > 2*trial_runs and simulation_run <= 3*trial_runs:
        if current_hyperparameters[0] in weather_parameters:
            if current_hyperparameters[0] != "sun_altitude_angle":
                min,max = 50,75
            elif current_hyperparameters[0] == "sun_altitude_angle":
                min,max = 45,75

    elif simulation_run > 3*trial_runs and simulation_run <= 4*trial_runs:
        if current_hyperparameters[0] in weather_parameters:
            if current_hyperparameters[0] != "sun_altitude_angle":
                min,max = 75,100
            elif current_hyperparameters[0] == "sun_altitude_angle":
                min,max = 75,90
    step = 5

    return min,max,step

def get_sample_choice(current_hyperparameters,simulation_run,previous_stats_file,exploration,trial_runs):
    """
    Get choices of the sample array
    """
    choices_array = []
    distributions = []
    previous_hyperparameters = []
    if simulation_run <= 0:
        for i in range(len(current_hyperparameters)):
            if current_hyperparameters[i][0] == 'sun_altitude_angle':
                parameter_distribution = 45.0
            else:
                parameter_distribution = 0.0
            distributions.append(int(parameter_distribution))
    elif simulation_run > 0:
        parameters = pd.read_csv(previous_stats_file + "/scene_parameters.csv", usecols = [0,1], header=None, index_col=False)
        for index, row in parameters.iterrows():
                previous_hyperparameters.append((row[0],int(row[1])))
        #print(previous_hyperparameters)
        if simulation_run > exploration:
            for i in range(len(current_hyperparameters)):
                for hype in previous_hyperparameters:
                    if hype[0] == current_hyperparameters[i][0]:
                            if hype[0] == "traffic_density" or hype[0] == "sensor_faults":
                                min, max = current_hyperparameters[i][1], current_hyperparameters[i][2]
                                step = 1.0
                            else:
                                min,max,step = sampling_rules(previous_hyperparameters[i])
                choice_list = np.arange(min,max,step)
                choices_array.append(choice_list)
                parameter_distribution = random.choice(choice_list)
                distributions.append(int(parameter_distribution))
                    #print(distributions)
        else:
            for i in range(len(current_hyperparameters)):
                #min, max = current_hyperparameters[i][1], current_hyperparameters[i][2]
                if current_hyperparameters[i][0] == 'road_segments' or current_hyperparameters[i][0] == 'traffic_density' or current_hyperparameters[i][0] == 'sensor_faults':
                    min, max = current_hyperparameters[i][1], current_hyperparameters[i][2]
                    step = 1
                else:
                    min,max,step = get_vals(current_hyperparameters[i],trial_runs,simulation_run)
                choice_list = np.arange(min,max,step)
                choices_array.append(choice_list)
                parameter_distribution = random.choice(choice_list)
                distributions.append(int(parameter_distribution))
    print(distributions)

    return choices_array, parameter_distribution, distributions

def Bayesian_Optimization_Search(current_hyperparameters,folder,simulation_run,root,y):
    """
    Bayesian optimization for scene selection
    """
    exploration = 80
    trial_runs = 20
    new_hyperparameters_list = []
    parameters = []
    collisions = []
    selected_parameters = []
    martingales = []
    data_folder = "simulation-data" + "/" + "simulation%d/"%(y)
    previous_stats_file = root + "scene%d"%(simulation_run-1)
    parameter_file = root + "sampled_parameters.csv"
    collision_file = data_folder + "collision_stats.csv"
    stats_file = data_folder + "ood_stats.csv"
    parameter_length = len(current_hyperparameters)
    scenario_score_file = data_folder + "scenario_score.csv"
    choices_array, parameter_distribution, distributions = get_sample_choice(current_hyperparameters,simulation_run,previous_stats_file,exploration,trial_runs)
    #new_hyperparameters_list.append((current_hyperparameters[i][0],int(parameter_distribution)))

    # if simulation_run > 0:
    #     process_data_from_previous_run(folder,simulation_run,data_folder,y)

    if simulation_run <= exploration:
        if simulation_run <= 0:
            new_parameter = distributions
            collisions = 0
        elif simulation_run > 0:

            parameters = read_parameter_file(parameter_file)
            #print("0")
            collisions, martingales, risk, scenario_score = read_previous_stats(collision_file,stats_file,scenario_score_file)
            #print("1")
            x1x2 = cartesian_product(*choices_array)
            print(len(x1x2))
            #x1x2 = np.array(list(product(*choices_array)))
            y_max = max(risk)
            #print("2")
            #new_parameters = Optimizer(parameters,risk)
            #y_mean, y_std = Optimizer(parameters, risk, x1x2, parameter_length)
            y_mean, y_std = gaussian_process(parameters, risk, x1x2, parameter_length)
            new_parameter = next_parameter_by_ei(y_max, y_mean, y_std, x1x2)
            new_parameter = distributions
            #print("1")

    elif simulation_run > exploration:
        parameters = read_parameter_file(parameter_file)
        #print(parameters)
        collisions, martingales, risk, scenario_score = read_previous_stats(collision_file,stats_file,scenario_score_file)
        x1x2 = cartesian_product(*choices_array)
        print(len(x1x2))
        #x1x2 = np.array(list(product(*choices_array))) #TODO: This step is expensive. Find a way to perform it once and reuse it
        y_max = max(risk)
        y_mean, y_std = gaussian_process(parameters, martingales, x1x2, parameter_length)
        new_parameter = next_parameter_by_ei(y_max, y_mean, y_std, x1x2)
        #print("2")
        #new_parameter = next_parameter_by_ei(y_max, y_mean, y_std, x1x2)


    for i in range(len(current_hyperparameters)):
        new_hyperparameters_list.append((current_hyperparameters[i][0],new_parameter[i]))
        selected_parameters.append(new_parameter[i])


    return selected_parameters, new_hyperparameters_list
