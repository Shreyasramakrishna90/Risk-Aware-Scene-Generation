# CARLA Client Code

This is the CARLA client folder that has all the client-related code for running the experiments. Most of the code is borrowed from the CARLA leaderboard challenge https://carlachallenge.org/get-started/. We use this code and build the scene generation and sampling approaches on top. 


1. The carla_project has some CARLA utility files that are required to set up the AV in the simulator.

2. The leaderboard is the main folder that is used in this work. Two important files are (1) leaderboard/leaderboard/leaderboard_evaluator.py - this is the main script that sets up the scene in the simulation and runs it. (2) image_agent.py - this is the LEC code that also hosts the [RESONATE Risk estimator](https://github.com/scope-lab-vu/Resonate) and the OOD detectors. 

3. The scenario_runner folder has all the functionalities for executing the [CARLA Autonomous Driving challange](https://carlachallenge.org/) setup can be found in this folder

4. The run_agent.sh script launches the simulation run in this setup. This is the access point that can be used to control the number of simulations to be run. It has two arguments that can be passed. (a) end - this is the number of scenes that need to be generated. (b) exploration_runs - In the case of BO, this decides the amount of training (exploration) required to train the Gaussian Process Model. This can be ignored in this work as we provide the sampler's ``warm start" using the previously generated random scenes. You can execute this script as follows.

```
./run_agent.sh 5    #this will generate 5 scenes
```

# Scene Generation

We use a scenario description DSML written in [textX](https://textx.github.io/textX/stable/) to generate different temporal scene parameters (weather parameters, time-of-day,traffic density), spatial scene parameters (road segments) and agent sensor faults.

# Language

carla.tx -- Has the grammer for the scenario description language. 

scene-model.carla -- Has the entities of a CARLA scene. The entities like town description, weather, road segments are defined here.

scene-model.carla -- Has the entities of an agent in CARLA simulation. The entities like ego agent type, sensors and faults are defined here.

# Specification Files

scene_description.yml - This is the scene specification file that allows the user to select what scene parameters need to be sampled and what sampler is needed to be used for sampling the scene parameters. 

# Interpreter

scene_interpreter.py -- The interpreter takes in the specification files from the user, the scenario language, and the samplers to generate scene artifact files for the simulator to use. It generates a route file (XML) for the CARLA simulator to set up the environment and an agent file (XML) to set up the sensors for the CARLA agent  
