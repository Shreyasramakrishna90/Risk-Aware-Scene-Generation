# Scene Generation

We use a scenario description DSML written in [textX](https://textx.github.io/textX/stable/) to generate different temporal scene parameters (weather parameters, time-of-day,traffic density), spatial scene parameters (road segments) and agent sensor faults.

# Language

[carla.tx](https://github.com/Shreyasramakrishna90/risk-aware-scene-generation/blob/main/carla_client/carla-challange/Scenario-Description-Updated/scene/carla.tx) -- Has the grammer for the scenario description language. 

[scene-model.carla](https://github.com/Shreyasramakrishna90/risk-aware-scene-generation/blob/main/carla_client/carla-challange/Scenario-Description-Updated/scene/scene-model.carla) -- Has the entities of a CARLA scene. The entities like town description, weather, road segments are defined here.

[scene-model.carla](https://github.com/Shreyasramakrishna90/risk-aware-scene-generation/blob/main/carla_client/carla-challange/Scenario-Description-Updated/scene/agent-model.carla) -- Has the entities of an agent in CARLA simulation. The entities like ego agent type, sensors and faults are defined here.

# Specification Files

[scene_description.yml](https://github.com/Shreyasramakrishna90/risk-aware-scene-generation/blob/main/carla_client/carla-challange/Scenario-Description-Updated/scene/scene_description.yml) - This is a specification file that allows the user to select what scene parameters need to be sampled and what sampler is needed to be used for sampling the scene parameters. 

[agent_description.yml](https://github.com/Shreyasramakrishna90/risk-aware-scene-generation/blob/main/carla_client/carla-challange/Scenario-Description-Updated/scene/agent_description.yml) - This is a specification file that allows the user to select what sensors need to be added to the agent and what data from the agent needs to be recorded at runtime.
# Interpreter

[scene_interpreter.py](https://github.com/Shreyasramakrishna90/risk-aware-scene-generation/blob/main/carla_client/carla-challange/Scenario-Description-Updated/scene/scene_interpreter.py) -- The interpreter takes in the specification files from the user, the scenrario language and the samplers to generate scene artifact files for the simulator to use. 
It generates a route file (XML) for the CARLA simulator to set up the environment and an agent file (XML) to set up the sensors for the CARLA agent  


# Samplers

The goal of this work is to test different samplers for sequential scene generation. We have integrated and the user can select from the following samplers.

1. [Manual](https://github.com/Shreyasramakrishna90/risk-aware-scene-generation/blob/main/carla_client/carla-challange/Scenario-Description-Updated/scene/samplers/Manual.py) - The user needs to specify the scene parameters if they choose this option.
2. [Random](https://github.com/Shreyasramakrishna90/risk-aware-scene-generation/blob/main/carla_client/carla-challange/Scenario-Description-Updated/scene/samplers/Random.py) - A random sampler is set up, which takes in the parameters that the user wants to sample. The other parameters will take a default value.
3. [Grid](https://github.com/Shreyasramakrishna90/risk-aware-scene-generation/blob/main/carla_client/carla-challange/Scenario-Description-Updated/scene/samplers/Grid.py) - A grid sampler is set up to search across all the scene parameter samples within their value ranges.
4. [Halton](https://github.com/Shreyasramakrishna90/risk-aware-scene-generation/blob/main/carla_client/carla-challange/Scenario-Description-Updated/scene/samplers/Halton.py) - Halton sampler is used to draw the scene sample parameters.
5. [Bayesian Optimization](https://github.com/Shreyasramakrishna90/risk-aware-scene-generation/blob/main/carla_client/carla-challange/Scenario-Description-Updated/scene/samplers/Bayesian_optimization.py) - A Bayesian Optimizer is set up to sample the scene parameters. Compared to the earlier samplers, this performs a closed loop sample selection. That is, it takes the previous risk values to sample new scene parameters.
6. Reinforcement Learning - An Actor Critic network to search the samples.
```
