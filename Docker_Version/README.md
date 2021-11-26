# Single Docker for CARLA Server-Client

This repo has a docker for the CARLA challange example with ReSonAte risk computation. Follow the steps below to build and run the dcker.

# Create Folders for Docker Volumes
Create three folders named ```routes```, ```simulation-data``` and ```images``` inside this directory. These folders are the data volumes for the carla client docker. Run the following

```
mkdir routes               #stores the scene information.
mkdir simulation-data      #stores the sensor information
mkdir images               #stores images if chosen by the user
```
Alternately, enter into this repo and execute this script ```./make_volume_folders.sh``` to set up these empty folders.

# Downloads

***Manual Downloads***

***Step 1***: Download [CARLA_0.9.9](https://github.com/carla-simulator/carla/releases/tag/0.9.9/) and put it inside the carla_client folder. Please pull CARLA_0.9.9.0 version. If you pull any other version, there will be a mismatch between the CARLA simulation API and the client API. 

***Step 2***: Pull the trained OOD detector weights from [here](https://vanderbilt365-my.sharepoint.com/:f:/g/personal/shreyas_ramakrishna_vanderbilt_edu/EvZqbV90bY1HmSCofd6A1m0BTlqrPBzOF1gy4vDvAt4KUQ?e=LnHlx6). Please download the folder (center-B-1.2-trial), unzip the folder and save it as "ood_detector_weights" into the following path "/carla_client/carla-challange/leaderboard/team_code/detector_code" .

***Step 3***: The preitrained LEC is got from [Learning By Cheating](https://github.com/bradyz/2020_CARLA_challenge). Download the weights from [here](https://vanderbilt365-my.sharepoint.com/:u:/g/personal/shreyas_ramakrishna_vanderbilt_edu/ETRBzI7Ai3VJt9zL7yPnJO4Bi5zYvgggreiY2CG68f8s8A?e=nGJIQl)

Unzip the weights file and save it as ***model.ckpt*** in the carla-challange/carla_project folder. 

***Automated Downloads (Preferred)***

Enter into this repo and execute this script ```./downloads.sh``` to download these three requirements automatically into the required folders.


# Docker Build

***Step 1***: Make sure NVIDIA Docker is set up. Please follow these [steps](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#installation-guide) if you have not already set it up.

***Step 2***: Run the following to build a docker using Ubuntu 18.04 image and then install all the libraries required to run the our carla client

```
./build_carla_client.sh
```
Please make sure the build is complete without errors. ```You will get sudo pip warning and that should be fine```

***Step 3***: After the docker is built. Set up xhost permission for the GUI to show up the simulation run. For this open a terminal on your local machine and run the following command 

```
xhost -local:docker
```
***Note***: Possible erors "Cannot connect to X server: 1" . The solution is to run **xhost +** in a terminal on your host computer.  [Reference](Reference: https://stackoverflow.com/questions/56931649/docker-cannot-connect-to-x-server)

# Docker Run

Now, run the carla client docker using the script below. 

***Not necessary in the latest commit*** ~~(Note: CARLA is mounted as a volume. So, please change the path of the CARLA folder in your run_carla_client file. ```-v /(Add Path to CARLA_0.9.9)/:/CARLA_0.9.9 \```)~~

```
./run_carla_client.sh
```  
This will take you into the carla client docker. When you are inside the docker run the following

```
./run_agent.sh
``` 
This script has a few variables that need to be set before execution. 

1. PORT: The simulator port (default:3000)
2. HAS_DISPLAY: 1 = display simulation run, 2 = headless mode (no display)
