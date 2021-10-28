import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import pandas as pd
from scipy.interpolate import make_interp_spline


def read_hyperparameter_data(file):
    """
    read hyperparameter files
    """
    data1 = []
    data2 = []
    data3 = []
    parameters = pd.read_csv(file, usecols = [0,1,2], header=None, index_col=False)
    x = 0
    for index, row in parameters.iterrows():
        #if x != 0:
        data1.append(int(row[0]))
        data2.append(int(row[1]))
        data3.append(int(row[2]))
        x+=1

    return data1, data2, data3

def read_simulation_data(file):
    """
    read simulation data. risk and martingale
    """
    data1 = []
    data2 = []
    length = []
    parameters = pd.read_csv(file, usecols = [0,1], header=None, index_col=False)
    for index, row in parameters.iterrows():
            data1.append(float(row[0]))
            data2.append(float(row[1]))
            length.append(index)

    return data1, data2, length


def exploration_plot(file,risk_file,store_plots):
    """
    3d plot of the weather parameters
    """
    combinations = [['cloud','precipitation','time-of-day'],['Risk','precipitation','time-of-day'],['Martingale','precipitation','time-of-day']]
    data1, data2, data3 = read_hyperparameter_data(file)
    sim_data1, sim_data2, index = read_simulation_data(risk_file)
    for index,combination in enumerate(combinations):
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        if 'Risk' in combination:
            x = sim_data2
            y = 'risk'
            z = 3.5
        elif 'Martingale' in combination:
            x = sim_data1
            y = 'martingale'
            z = 40
        else:
            x = data1
            y = 'cloud'
            z =100
        ax.scatter3D(data3, data2, x, c=None, cmap='hsv')
        ax.set_xlim(0,90)
        ax.set_ylim(0,100)
        ax.set_zlim(0,z)
        ax.set_xlabel(combination[2])
        ax.set_ylabel(combination[1])
        ax.set_zlabel(y)
        fig.savefig(store_plots +'fig%d.png'%index, bbox_inches='tight')
        plt.cla()

def smooth(scalars, weight):  # Weight between 0 and 1
    last = scalars[0]  # First value in the plot (first timestep)
    smoothed = list()
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point  # Calculate smoothed value
        smoothed.append(smoothed_val)                        # Save it
        last = smoothed_val                                  # Anchor the last smoothed value

    return smoothed

def simulation_performance(file,risk_file,store_plots):
    """
    plot 3d plots
    """
    plots = ['Likelihood of Collision', 'OOD Monitor Martingale']
    sim_data1, sim_data2, index = read_simulation_data(risk_file)
    for plot in plots:
        if plot == 'Likelihood of Collision':
            x = sim_data2
            y = 3.5
        else:
            x = sim_data1
            y = 40
        fig, ax = plt.subplots()
        color = 'tab:blue'
        ax.set_xlabel('Simulations')
        ax.set_ylabel(plot)
        ax.set_ylim([0, y])
        ax.plot(index, smooth(x, .9), color=color, label= 'stopping distance')
        #ax.tick_params(axis='y', labelcolor=color)
        fig.savefig(store_plots +'%s.png'%plot, bbox_inches='tight')
        plt.cla()

def simulation_performance_combined(file,risk_file,store_plots):
    """
    plot risk and martingale
    """
    sim_data1, sim_data2, index = read_simulation_data(risk_file)
    fig, ax = plt.subplots()
    color = 'tab:red'
    ax.set_xlabel('Simulations')
    ax.set_ylabel('Likelihood of Collision',color=color)
    ax.set_ylim([0, 2.5])
    ax.plot(index, smooth(sim_data2, .95), color=color, label= 'Likelihood of Collision')

    ax2 = ax.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('Monitor_result', color=color)
    ax2.plot(index, smooth(sim_data1, 0.95), color=color, label = 'monitor results')
    ax2.set_ylim([-5, 40])
    #ax.tick_params(axis='y', labelcolor=color)
    fig.legend(loc=8, bbox_to_anchor=(0.5, -0.02),fancybox=True, shadow=True, ncol=2)
    fig.subplots_adjust(bottom=0.5)
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    fig.savefig(store_plots +'combined.png', bbox_inches='tight')
    plt.cla()

def get_root_folder(path):
    dirlist = [ item for item in os.listdir(path) if os.path.isdir(os.path.join(path,item)) ]
     #has the list containing all the folder names
    #x=list(max(dirlist)) #finds the max number in the list and converts it into a list
    #y=x[-1]=int(x[-1])
    folder_len = len(dirlist) - 1

    return int(folder_len)


def plot_stats():
    print("----------------------")
    print("Plotting Stats")
    print("----------------------")
    route_root = "routes" + "/"
    data_root = "simulation-data" + "/"
    x = get_root_folder(route_root)
    data_file_location = data_root + "simulation%d"%x + "/"
    route_file_location = route_root + "simulation%d"%x + "/"
    store_plots = route_file_location + "plots" + "/"
    if not os.path.exists(store_plots):
        os.makedirs(store_plots)
    sampled_hyperparameter_file = route_file_location + "sampled_parameters.csv"
    risk_file = data_file_location + "ood_stats.csv"
    exploration_plot(sampled_hyperparameter_file, risk_file, store_plots)
    simulation_performance(sampled_hyperparameter_file,risk_file,store_plots)
    simulation_performance_combined(sampled_hyperparameter_file,risk_file,store_plots)


if __name__ == '__main__':
    plot_stats()
