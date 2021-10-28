#!/usr/bin/python3
import os
import sys
import numpy as np

weather_parameters = ['cloudiness','precipitation','precipitation_deposits','wind_intensity','sun_azimuth_angle','wetness','fog_distance','fog_density']

def next_prime():
    def is_prime(num):
        "Checks if num is a prime value"
        for i in range(2,int(num**0.5)+1):
            if(num % i)==0: return False
        return True

    prime = 3
    while(1):
        if is_prime(prime):
            yield prime
        prime += 2

def vdc(n, base=2):
    vdc, denom = 0, 1
    while n:
        denom *= base
        n, remainder = divmod(n, base)
        vdc += remainder/float(denom)
    return vdc

def normalize_vals(sample,parameter):
    if parameter in weather_parameters:
        min,max = 0,100
    elif parameter == "sun_altitude_angle":
        min,max = 0,90
    elif parameter == 'road_segments':
        min,max = 0,10
    elif parameter == 'traffic_density':
        min,max = 10,20
    elif parameter == 'sensor_faults':
        min,max = 0,15

    sample = sample * (max - min) + min



    return round(sample)

def halton(size, dim):
    seq = []
    primeGen = next_prime()
    next(primeGen)
    for d in range(dim):
        base = next(primeGen)
        seq.append([vdc(i, base) for i in range(size)])
    return seq

def Halton_Sequence(current_hyperparameters,folder,simulation_run,root,y,total_scenes):
    distributions = []
    new_hyperparameters_list = []
    parameters = []
    samples = halton(total_scenes, len(current_hyperparameters))
    for i in range(len(samples[0])):
        temp = []
        for j in range(len(samples)):
            temp.append(samples[j][i])
        parameters.append(temp)
    #print(parameters)
    #distributions = parameters[simulation_run]
    for index,hype in enumerate(current_hyperparameters):
        sample = normalize_vals(parameters[simulation_run][index],hype[0])
        distributions.append(sample)
        new_hyperparameters_list.append((hype[0],sample))
    #print(new_hyperparameters_list)

    return distributions, new_hyperparameters_list


# def Halton_Sequence(num,simulation_run):
#     distributions = []
#     new_hyperparameters_list = []
#
#     parameters = []
#     samples = halton(num, 3)
#     for i in range(len(samples[0])):
#         for j in range(len(samples)):
#             parameters.append(samples[j][i])
#     print(parameters)
#     distributions = parameters[simulation_run]
#     # for index,hype in enumerate(current_parameters):
#     #     new_hyperparameters_list.append((current_hyperparameters[i][0],float(distributions[index])))
#
#     # return distributions, new_hyperparameters_list
#
#
# if __name__ == '__main__':
#     num = 10
#     for simulation_run in range(10):
#         Halton_Sequence(num,simulation_run)


# if __name__ == '__main__':
#
#
#     cloud = []
#     precipitation = []
#     time_of_day = []
#     samples = halton(10, 3)
#     #print(len(samples))
#     for sample in samples[0]:
#         time_of_day.append(normalize_vals(sample,min=0,max=90))
#     for sample in samples[1]:
#         cloud.append(normalize_vals(sample,min=0,max=100))
#     for sample in samples[2]:
#         precipitation.append(normalize_vals(sample,min=0,max=100))
#
#     print(time_of_day)
#     print(cloud)
#     print(precipitation)



    #sample = sample * (max_ - min_) + min_
    #samples1 = halton(10, 1)
    #print(samples)
    #print(samples1)
    # for index,sample in enumerate(samples):
    #     print(sample)
    #print(len(halton(100, 100)))
    # [[ 0.5         0.33333333]
    #  [ 0.25        0.66666667]
    #  [ 0.75        0.11111111]
    #  [ 0.125       0.44444444]
    #  [ 0.625       0.77777778]]
