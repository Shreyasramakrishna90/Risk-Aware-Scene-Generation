# CARLA Client Code

This is the CARLA client folder that has all the client related code for running the experiments. Most of the code is borrowed from the CARLA leaderboard challange https://carlachallenge.org/get-started/. We use this code and build the scene generation and sampling approaches on top. 


1. The carla_project has some CARLA utility files that is required to setup the AV in the simulator.

2. The leaderboard is the main folder that is used in this work. Two files that are important are: (1) leaderboard/leaderboard/leaderboard_evaluator.py - this is the main script that sets up the scene in the simulation and runs it. (2) image_agent.py - this is the LEC code that also hosts the [RESONATE Risk estimator](https://github.com/scope-lab-vu/Resonate) and the OOD detectors. 

3. The scenario_runner folder has all the functionalities for executing the [CARLA Autonomous Driving challange](https://carlachallenge.org/) setup can be found in this folder

4. The run_agent.sh script launches the simulation run in this setup. This is the access point that can be used to control the number of simulations to be run. It has two arguments that can be passed. (a) end - this is the number of scenes that need to be generated. (b) exploration_runs - In case of BO, this decides the amount of training (exploration) required to train the Gaussian Process Model. This can be ignored in this work as we provide the samplers ``warm start" using the previously generated random scenes. You can execute this script as follows.

```
./run_agent.sh 5    #this will generate 5 scenes
```
