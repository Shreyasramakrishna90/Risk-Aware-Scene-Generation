import cv2
import glob
import numpy as np
import scipy as sp
from matplotlib import *
from pylab import *
import time
import os
from sklearn.utils import shuffle
from keras.models import model_from_json
from keras.losses import mse
import csv
import pandas as pd
import re


def extract_run_path(path):
    runs_path = []
    for run_data in glob.glob(path +"run*.csv"):
        runs_path.append(run_data)
    #runs_path.sort(reverse=True)
    # for i in range(1,runs+1):
    #     runs_path.append(path + 'run%d.csv'%i)
    print(runs_path)
    return runs_path

def extract_collision_data(path):
    collision_path = []
    for collision_data in glob.glob(path +"data*.txt"):
        collision_path.append(collision_data)
    print(collision_path)
    final_colisions = []
    for j in range(len(collision_path)):
        file1 = open(collision_path[j], 'r')
        Lines = file1.readlines()
        count = 0
        data = []
        collisions = []
        col = []
        for line in Lines:
                data.append(line.strip())
        for i in range(len(data)):
            number = []
            if(data[i]!= "" and data[i][0].isdigit()):
                for k in range(len(data[i])):
                    if(data[i][k].isdigit()):
                        number.append(data[i][k])
                    elif(data[i][k] == " "):
                        break
                collisions.append(number)
        for x in range(len(collisions)):
            col.append("".join(collisions[x]))
        final_colisions.append(col)
    print(final_colisions)

    return final_colisions[0]


def extract_fault_data(fault_data_path):
    fault_data = []
    with open(fault_data_path, 'r') as readFile:
        reader = csv.reader(readFile)
        next(reader)
        for row in reader:
            data = []
            data.append(row[0])
            data.append(float(row[1])/20)
            data.append((float(row[1]) + float(row[2]))/20)
            fault_data.append(data)
    print(fault_data)
    return fault_data

def extract_weather_data(weather_path):
    weather_data = []
    with open(weather_path, 'r') as readFile:
        reader = csv.reader(readFile)
        for row in reader:
            weather_data.append(row)
    print(weather_data)
    return weather_data

def plot(runs_path,weather_data,collision_times,fault_data,path):
        ground_truth = []
        stopping_distance = []
        speed = []
        steer = []
        mval = []
        steps = []
        time = []
        with open(path + "/run1.csv", 'r') as readFile:
            reader = csv.reader(readFile)
            next(reader)
            for row in reader:
                steps.append(float(row[0]))
                time.append(float(row[0])/20.0)
                #risk.append(float(row[2]))
                mval.append(float(row[1]))

        with open(path + "distance.csv", 'r') as readFile:
            reader = csv.reader(readFile)
            next(reader)
            for row in reader:
                ground_truth.append(float(row[0]))
                #time.append(float(row[0])/20.0)
                #risk.append(float(row[2]))
                #mval.append(float(row[1]))

        with open(path + "emergency_braking.csv", 'r') as readFile:
            reader = csv.reader(readFile)
            next(reader)
            for row in reader:
                stopping_distance.append(float(row[1]))
                speed.append(float(row[2]))
                steer.append(float(row[3]))


        fig, ax1 = plt.subplots()
        color = 'tab:red'
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('stopping_distance', color=color)
        ax1.set_ylim([0, 72])
        x=0
        if(len(collision_times)!=0):
            for xc in collision_times:
                ax1.axvline(x=float(xc),linewidth = 2, linestyle ="--", color ='green', label="collision" if x == 0 else "")
                x+=1
        #print(fault_data[0])
        if(fault_data[0][0]=="1"):
            ax1.axvspan(fault_data[0][1],fault_data[0][2], alpha=0.2, color = 'yellow', label = "fault duration")
        ax1.plot(time, stopping_distance, color=color, label= 'stopping distance')
        ax1.tick_params(axis='y', labelcolor=color)

        #ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

        color = 'tab:brown'
        #ax1.set_ylim([])
        ax1.plot(time, ground_truth, color=color, label = 'ground truth distance')

        # color = 'tab:orange'
        # ax1.plot(time, speed, color=color, label = 'speed')
        # color = 'tab:pink'
        # ax1.plot(time, steer, color=color, label = 'steer')

        ax3 = ax1.twinx()

        color = 'tab:blue'
        ax3.set_ylabel('Monitor_result', color=color)
        ax3.plot(time, mval, color=color, label = 'monitor results')
        ax3.set_ylim([-5, 40])
        ax3.tick_params(axis='y', labelcolor=color)
        ax3.set_title("Scene with Cloud:%s, Precip:%s, Precip-deposit:%s"%(weather_data[4],weather_data[2],weather_data[3]))
        fig.legend(loc=8, bbox_to_anchor=(0.5, -0.02),fancybox=True, shadow=True, ncol=5)
        #fig.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),fancybox=True, shadow=True, ncol=2)
        fig.subplots_adjust(bottom=0.5)
        fig.tight_layout()  # otherwise the right y-label is slightly clipped
        fig.savefig(path+'run.png', bbox_inches='tight')
        plt.cla()
    #plt.show()


if __name__ == '__main__':
    runs = int(input("Enter the simulation run to be plotted:"))
    for j in range(0,15):
        path = "/home/scope/Carla/braking-carla/leaderboard/data/my_data/simulation%d/failure_mode%d/"%(runs,j)
        weather_path =  path + "simulation_data.csv" #"/home/scope/Carla/ICCPS_CARLA_challenge/leaderboard/data/my_data/simulation%d/simulation_data.csv"%runs
        fault_data_path = path + "fault_data.csv" #"/home/scope/Carla/ICCPS_CARLA_challenge/leaderboard/data/my_data/simulation%d/fault_data.csv"%runs
        collision_times = extract_collision_data(path)
        print(collision_times)
        fault_data = extract_fault_data(fault_data_path)
        runs_path = extract_run_path(path)
        weather_data = extract_weather_data(weather_path)
        plot(runs_path,weather_data[1],collision_times,fault_data,path)
