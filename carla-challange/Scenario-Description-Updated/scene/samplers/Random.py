#!/usr/bin/python3
import csv
import pandas as pd
import random
import numpy as np



def Random_Search(current_hyperparameters,folder,simulation_run,root,y):
    """
    The random sampler takes in the hyperparameters of the current step and returns a new hyperparameter set that is randomly sampled
    """
    new_hyperparameters_list = []
    choices_array = []
    distributions = []
    if simulation_run <= 0:
        for i in range(len(current_hyperparameters)):
            if current_hyperparameters[i][0] == 'sun_altitude_angle':
                parameter_distribution = 45.0
            else:
                parameter_distribution = 0.0
            distributions.append(int(parameter_distribution))
            new_hyperparameters_list.append((current_hyperparameters[i][0],int(parameter_distribution)))
    else:
        for i in range(len(current_hyperparameters)):
            if current_hyperparameters[i][0] == 'segment_number' or 'traffic' or 'camera_faults':
                step = 1
            else:
                step = 5
            #if current_hyperparameters[i][1] == current_hyperparameters[i][2]:
            #    parameter_distribution = current_hyperparameters[i][1]
            #else:
            choice_list = np.arange(current_hyperparameters[i][1],current_hyperparameters[i][2],step)
            choices_array.append(choice_list)
            parameter_distribution = random.choice(choice_list)
            distributions.append(int(parameter_distribution))
            new_hyperparameters_list.append((current_hyperparameters[i][0],int(parameter_distribution)))


    return distributions, new_hyperparameters_list


# def sampler(current_hyperparameters,folder,simulation_run,root,y):
#     """
#     The sampler code that takes in current step hyperparameters and returns new hyperparameter set
#     1. Bayesian Optimization
#     2. Optuna with Hyperband
#     3. Reinforcement Learning
#     """
#     new_hyperparameters_list = []
#     if simulation_run <= 20: #initial phase which randomly samples the hyperparameters
#         for i in range(len(current_hyperparameters)):
#             if current_hyperparameters[i][0] == 'sun_altitude_angle':
#                 step = 45
#             else:
#                 step = 1
#             choice_list = np.arange(current_hyperparameters[i][1],current_hyperparameters[i][2],step)
#             parameter_distribution = random.choice(choice_list)
#             #parameter_distribution = np.random.uniform(current_hyperparameters[i][1],current_hyperparameters[i][2])
#             new_hyperparameters_list.append((current_hyperparameters[i][0],int(parameter_distribution)))
#     else: #post initial phase that exploits the previous hyperparameters and results to get new hyperparameters
#         collision, martingale_value, previous_hyperparameters = get_data_from_previous_run(folder,simulation_run,root,y) #This is the result from the previous runs
#         print(previous_hyperparameters)
#         for i in range(len(current_hyperparameters)): #Replace this part with the optimization algorithm
#             parameter_distribution = np.random.uniform(current_hyperparameters[i][1],current_hyperparameters[i][2])
#             new_hyperparameters_list.append((current_hyperparameters[i][0],int(parameter_distribution)))
#
#     with open(folder + "/scene_parameters.csv", 'w') as csvfile: #Always save the selected hyperparameters for optimization algorithms
#         writer = csv.writer(csvfile, delimiter = ',')
#         writer.writerows(new_hyperparameters_list)
#
#
#     return new_hyperparameters_list
