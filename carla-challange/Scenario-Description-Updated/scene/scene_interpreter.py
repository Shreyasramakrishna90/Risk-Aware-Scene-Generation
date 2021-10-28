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
from utils import scene_file_generator,parse_routes_file
import csv
import argparse
from argparse import RawTextHelpFormatter
import pandas as pd
import random
from ruamel import yaml
from samplers.BO import Bayesian_Optimization_Search
from samplers.Random import Random_Search
from samplers.Manual import ManualEntry
from samplers.Halton import Halton_Sequence

varying_entities = ['scene_length', 'weather', 'road_segments', 'traffic_density', 'sensor_faults']
weather_parameters = ['cloudiness','precipitation','precipitation_deposits','sun_altitude_angle','wind_intensity','sun_azimuth_angle','wetness','fog_distance','fog_density']

def sampling_rules(sample):
    """
    Describes the sampling rules for selecting the weather samples
    The weather samples can only gradually increase in steps rather than having big jumps
    """
    if sample[0] in weather_parameters:
        min, max = (int(sample[1])-10, int(sample[1])+10)
    elif sample[0] is "road_segments":
        min, max = (int(sample[1]),int(sample[1])+1)
    step = 1

    return min, max, step


def compositional_rules(sample):
    """
    Describes the compositional rules for selecting the road segments
    There are only a certain waypoint order that needs to be followed
    This order is got from the Carla map
    """
    if sample[0] is "road_segments":
        min, max = (int(sample[1]),int(sample[1])+1)


    return min, max

def write_sampler_results(root,folder,parameter_values,joined_parameters):
    """
    Parameters returned by the sampler
    """
    with open(root + "sampled_parameters.csv", 'a') as csvfile: #Always save the selected hyperparameters for optimization algorithms
        writer = csv.writer(csvfile, delimiter = ',')
        writer.writerow(parameter_values)


    with open(folder + "/scene_parameters.csv", 'w') as csvfile1: #Always save the selected hyperparameters for optimization algorithms
        writer = csv.writer(csvfile1, delimiter = ',')
        writer.writerows(joined_parameters)



def parameter_sampler(dynamic_parameters,static_parameters,folder,simulation_run,root,y,optimizer,sampler,total_scenes):
    """
    The sampler code that takes in current step hyperparameters and returns new hyperparameter set
    1. Manual Entry by User
    2. Random Search
    3. Grid Search
    4. Halton Search
    5. Bayesian Optimization Search
    6. Reinforcement Learning
    """
    if sampler == "Random":
        parameter_values, sampled_parameters = Random_Search(dynamic_parameters,folder,simulation_run,root,y)
    if sampler == "Halton":
        parameter_values, sampled_parameters = Halton_Sequence(dynamic_parameters,folder,simulation_run,root,y,total_scenes)
    elif sampler == "Bayesian_Optimization":
        parameter_values, sampled_parameters = Bayesian_Optimization_Search(dynamic_parameters,folder,simulation_run,root,y)

    joined_parameters = sampled_parameters + static_parameters

    write_sampler_results(root,folder,parameter_values,joined_parameters)

    #print(sampled_param_values)
    return joined_parameters


def get_distribution_values(parameter_name):
    """
    Hard coded min and max values for the parameters
    """
    # if(parameter_name == 'cloudiness' or parameter_name == 'precipitation' or parameter_name == 'precipitation_deposits' or parameter_name == 'sun_altitude_angle'):
    #     min,max = (0,100)
    # elif(parameter_name == 'wind_intensity' or parameter_name == 'sun_azimuth_angle' or parameter_name == 'wetness' or parameter_name == 'fog_distance' or parameter_name == 'fog_density'):
    #     min,max = (0,100)
    if parameter_name in weather_parameters:
        min,max = 0,100
    elif parameter_name == "sun_altitude_angle":
        min,max = 0,90
    elif parameter_name == 'road_segments':
        min,max = (0,10)
    elif parameter_name == 'traffic_density':
        min,max = (10,20)
    elif parameter_name == 'sensor_faults':
        min,max = (0,15)

    return min, max

def read_scene_parameters(scene_info,varying_scene_entities,varying_scene_parameters):
    """
    Read scene parameters and their distribution ranges entered by the user
    """
    static_parameters = []
    dynamic_parameters = []
    num_entities = len(scene_info.entities)
    for i in range(0,num_entities):
        entity_name = scene_info.entities[i].name
        if entity_name in varying_entities:
            num_parameters = len(scene_info.entities[i].properties)
            for j in range(0,num_parameters):
                parameter_name = scene_info.entities[i].properties[j].name
                parameter_type = scene_info.entities[i].properties[j].type.name
                if parameter_name in varying_scene_parameters:
                    #print(parameter_name)
                    if parameter_type == 'distribution':
                        parameter_min, parameter_max = get_distribution_values(parameter_name)
                        dynamic_parameters.append((parameter_name,parameter_min,parameter_max))
                else:
                    if parameter_name == 'sun_altitude_angle':
                        parameter_value = 45
                    else:
                        parameter_value = 0
                    static_parameters.append((parameter_name,parameter_value))

    return dynamic_parameters, static_parameters


def organize_parameters(param_vals):
    """
    Organize varying scene parameters
    """
    weather_list = []
    #camera_fault_type = 0
    for val in param_vals:
        if val[0] == 'time':
            scene_time = int(val[1])
        elif val[0] == 'traffic_density':
            traffic_info = int(val[1])
        elif val[0] == 'road_segments':
            road_segment = int(val[1])
            #print(road_segment)
        elif val[0] == 'sensor_faults':
            fault_type = int(val[1])
        else:
            weather_list.append(val)

    return weather_list,traffic_info,road_segment, fault_type


def set_scene_weather(scene_info,weather_list):
    """
    Weather parameters for the scene
    """
    weather = []
    for i, entry in enumerate(weather_list):
        if entry[0] == 'cloudiness':
            scene_info.entities[2].properties[0] = str(int(entry[1]))
        elif entry[0] == 'precipitation':
            scene_info.entities[2].properties[1] = str(int(entry[1]))
        elif entry[0] == 'precipitation_deposits':
            scene_info.entities[2].properties[2] = str(int(entry[1]))
        elif entry[0] == 'sun_altitude_angle':
            scene_info.entities[2].properties[3] = str(int(entry[1]))
        elif entry[0] == 'wind_intensity':
            scene_info.entities[2].properties[4] = str(int(entry[1]))
        elif entry[0] == 'sun_azimuth_angle':
            scene_info.entities[2].properties[5] = str(int(entry[1]))
        elif entry[0] == 'wetness':
            scene_info.entities[2].properties[6] = str(int(entry[1]))
        elif entry[0] == 'fog_distance':
            scene_info.entities[2].properties[7] = str(int(entry[1]))
        elif entry[0] == 'fog_density':
            scene_info.entities[2].properties[8] = str(int(entry[1]))
        weather.append(str(int(entry[1])))

    return weather

def set_scene_road_segment(global_route,road_segment):
    """
    Waypoints for the scene
    """
    list = []
    list.append(global_route[road_segment*2])
    list.append(global_route[road_segment*2+1])
    list.append(global_route[road_segment*2+2])

    return list

def write_weather_data(weather,data_file):
    """
    Write weather data into a file for simulator
    """
    with open(data_file, 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter = ',')
        writer.writerows(weather)

def get_scene_info(scenario_language_path,carla_route, num_scenes, varying_scene_entities, varying_scene_parameters, sampler, manual_scene_specification):
    """
    Invoke the scenario generation language and pull in the language grammer, scene parameters that require sampling
    """
    grammar = metamodel_from_file(scenario_language_path + 'carla.tx') #grammer for the scenario language
    scene_info = grammar.model_from_file(scenario_language_path + 'scene-model.carla') #scenario entities
    agent_info = grammar.model_from_file(scenario_language_path + 'agent-model.carla') #agent entities
    global_route,town = parse_routes_file(carla_route,False) #global route by reading one route from CARLA AD
    dynamic_parameters, static_parameters = read_scene_parameters(scene_info,varying_scene_entities, varying_scene_parameters) #Get the scene parameters that vary

    #print(scene_info.entities)
    return scene_info, dynamic_parameters, static_parameters,global_route,town

def read_yaml(yaml_file):
    """
    Read the input yaml file entered by the user
    """
    with open(yaml_file) as file:
        config = yaml.safe_load(file)

    return config

def decode_scene_description(scene_config):
    """
    Decode the scene description to extract scene related information
    """
    manual_scene_specification = []
    varying_entities = []
    varying_parameters = []
    sampler = []
    num_scenes = scene_config['Simulation Length']
    for entry in scene_config['Scene Description']:
        if entry == 'weather':
            val = 0
            for data in scene_config['Scene Description']['weather']:
                if scene_config['Scene Description']['weather'][data] is True:
                    varying_parameters.append(data)
                    val+=1
            if val > 0:
                varying_entities.append(entry)
        else:
            if scene_config['Scene Description'][entry] is True:
                varying_entities.append(entry)
                varying_parameters.append(entry)

    for entry in scene_config['Samplers']:
        if scene_config['Samplers'][entry] is True:
            sampler.append(entry)

    if len(sampler) > 1:
        print("Warning multiple samplers are selected in the specification file!!!!")
        sys.exit(1)
    if len(sampler) == 0:
        print("Warning no samplers are selected in the specification file!!!!")
        sys.exit(1)

    if sampler[0] == 'Manual':
        if not scene_config['Scene Specification']:
            print("Warning: Manual Scene Specification is not provided!!!")
            #quit()
            sys.exit(1)
        else:
            for entry in scene_config['Scene Specification']:
                manual_scene_specification.append(scene_config['Scene Specification'][entry])
                print(entry)

    print("-----------Running %s Optimizer---------------"%sampler[0])

    return num_scenes, varying_entities, varying_parameters, sampler[0], manual_scene_specification

def main(args,root,y):
    """
    Main that hosts the scenario generation in loop
    """
    scene_num = args.scene_num
    simulation_run = args.simulation_num
    optimizer = args.optimizer
    total_scenes = args.total_scenes
    distributions = []
    print("----------------------------------------------")
    print("Run%d complete"%scene_num)
    weather_data = []
    joined_parameter_list = []
    carla_route = '/carla-challange/leaderboard/data/routes/route_17.xml'
    folder = root + "scene%d"%simulation_run #folder to store all the xml generated
    scenario_language_path = 'Scenario-Description-Updated/scene/'
    os.makedirs(folder, exist_ok=True)
    data_file = folder+ "/weather_data.csv"
    scene_description = '/carla-challange/Scenario-Description-Updated/scene/scene_description.yml'
    #agent_description = '/carla-challange/Scenario-Description-Updated/scene/agent_description.yml'
    scene_config = read_yaml(scene_description)
    #agent_config = read_yaml(agent_description)
    num_scenes, varying_scene_entities, varying_scene_parameters, sampler, manual_scene_specification = decode_scene_description(scene_config)
    scene_info,dynamic_parameters, static_parameters,global_route,town = get_scene_info(scenario_language_path,carla_route,num_scenes, varying_scene_entities, varying_scene_parameters, sampler, manual_scene_specification)
    if sampler == 'Manual':
        joined_parameter_list = ManualEntry(manual_scene_specification,folder,simulation_run,root,y,dynamic_parameters, static_parameters)
    else:
        if len(dynamic_parameters) == 0:
            joined_parameter_list = static_parameters
            for entry in static_parameters:
                distributions.append(entry[1])
                #joined_parameter_list.append((entry[0],0))
            write_sampler_results(root,folder,distributions,joined_parameter_list)
        else:
            joined_parameter_list = parameter_sampler(dynamic_parameters,static_parameters,folder,simulation_run,root,y,optimizer,sampler,total_scenes)
        #joined_parameter_list = dynamic_parameters + static_parameters
    weather_list,traffic_info,road_segment, fault_type = organize_parameters(joined_parameter_list) #Organize the selected hyperparameters
    weather = set_scene_weather(scene_info,weather_list) #weather description
    weather_data.append(weather)
    road_segment_list = set_scene_road_segment(global_route,road_segment) #Choose the road segment
    scene_file_generator(scene_info,scene_num,folder,road_segment_list,town) #generated XML each round
    write_weather_data(weather_data,data_file) #write weather to file


def write_folder_number(y,path):
    """
    Write the folder number in which the routes are stored
    """
    file1 = open(path + "tmp.txt","w")
    file1.write(str(y))
    file1.close() #to change file access modes

def create_root_folder(sim_number):
    paths = []
    folders = ["routes"]
    for folder in folders:
        new_path = folder + "/"
        dirlist = [ item for item in os.listdir(new_path) if os.path.isdir(os.path.join(new_path,item)) ]
        folder_len = len(dirlist)
        if folder_len == 0:
            y = 0
        else:
            if sim_number == 0:
                y = folder_len
            else:
                y = folder_len - 1
        root = new_path + "simulation%d"%y + "/"
        os.makedirs(root, exist_ok=True) #creates a new dir everytime with max number
        paths.append(root)

    write_folder_number(y,new_path)

    return paths, y


if __name__ == '__main__':
        description = "CARLA Scene Generation\n"
        parser = argparse.ArgumentParser(description=description, formatter_class=RawTextHelpFormatter)
        parser.add_argument('--simulation_num', type=int, default=1, help='Type the simulation folder to store the data')
        parser.add_argument('--scene_num', type=int, default=1, help='Type the scene number to be executed')
        parser.add_argument('--optimizer', type=str, default="Random", help='Type the Optimizer to be used for scene selection')
        parser.add_argument('--total_scenes', type=int, help='Total number of scenes')
        args = parser.parse_args()
        #path = "/home/scope/Carla/carla-dockers/"
        root,y = create_root_folder(args.simulation_num)

        main(args,root[0],y)
