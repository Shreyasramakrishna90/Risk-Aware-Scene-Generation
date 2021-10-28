#!/usr/bin/env python
# Copyright (c) 2018-2019 Intel Corporation.
# authors: German Ros (german.ros@intel.com), Felipe Codevilla (felipe.alcm@gmail.com)
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
CARLA Challenge Evaluator Routes

Provisional code to evaluate Autonomous Agents for the CARLA Autonomous Driving challenge
"""
from __future__ import print_function

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import traceback
import argparse
from argparse import RawTextHelpFormatter
from datetime import datetime
from distutils.version import LooseVersion
import importlib
import os
import pkg_resources
import sys
import torchvision
import csv
from keras import backend as K
from numba import cuda
import pandas as pd
import carla
from srunner.scenariomanager.carla_data_provider import *
from srunner.scenariomanager.timer import GameTime
from srunner.scenarios.control_loss import *
from srunner.scenarios.follow_leading_vehicle import *
from srunner.scenarios.maneuver_opposite_direction import *
from srunner.scenarios.no_signal_junction_crossing import *
from srunner.scenarios.object_crash_intersection import *
from srunner.scenarios.object_crash_vehicle import *
from srunner.scenarios.opposite_vehicle_taking_priority import *
from srunner.scenarios.other_leading_vehicle import *
from srunner.scenarios.signalized_junction_left_turn import *
from srunner.scenarios.signalized_junction_right_turn import *
from srunner.scenarios.change_lane import *
from srunner.scenarios.cut_in import *

from leaderboard.scenarios.scenario_manager import ScenarioManager
from leaderboard.scenarios.route_scenario import RouteScenario
from leaderboard.autoagents.agent_wrapper import SensorConfigurationInvalid
from leaderboard.utils.statistics_manager import StatisticsManager
from leaderboard.utils.route_indexer import RouteIndexer
#ftom leaderboard.team_code.image_agent import ImageAgent


sensors_to_icons = {
    'sensor.camera.semantic_segmentation':        'carla_camera',
    'sensor.camera.rgb':        'carla_camera',
    'sensor.lidar.ray_cast':    'carla_lidar',
    'sensor.other.radar':       'carla_radar',
    'sensor.other.gnss':        'carla_gnss',
    'sensor.other.imu':         'carla_imu',
    'sensor.opendrive_map':     'carla_opendrive_map',
    'sensor.speedometer':       'carla_speedometer'
}


class LeaderboardEvaluator(object):

    """
    TODO: document me!
    """

    ego_vehicles = []

    # Tunable parameters
    client_timeout = 10.0  # in seconds
    wait_for_world = 20.0  # in seconds
    frame_rate = 20.0      # in Hz

    def __init__(self, args, statistics_manager):
        """
        Setup CARLA client and world
        Setup ScenarioManager
        """
        self.statistics_manager = statistics_manager
        self.sensors = []
        self._vehicle_lights = carla.VehicleLightState.Position | carla.VehicleLightState.LowBeam

        # First of all, we need to create the client that will send the requests
        # to the simulator. Here we'll assume the simulator is accepting
        # requests in the localhost at port 2000.
        self.client = carla.Client(args.host, int(args.port))
        if args.timeout:
            self.client_timeout = float(args.timeout)
        self.client.set_timeout(self.client_timeout)

        dist = pkg_resources.get_distribution("carla")
        if LooseVersion(dist.version) < LooseVersion('0.9.9'):
            raise ImportError("CARLA version 0.9.9 or newer required. CARLA version found: {}".format(dist))

        # Load agent
        module_name = os.path.basename(args.agent).split('.')[0]
        sys.path.insert(0, os.path.dirname(args.agent))
        self.module_agent = importlib.import_module(module_name)

        # Create the ScenarioManager
        self.manager = ScenarioManager(args.debug, args.challenge_mode, args.track, self.client_timeout)

        # Time control for summary purposes
        self._start_time = GameTime.get_time()
        self._end_time = None

    def __del__(self):
        """
        Cleanup and delete actors, ScenarioManager and CARLA world
        """

        self._cleanup(True)
        if hasattr(self, 'manager') and self.manager:
            del self.manager
        if hasattr(self, 'world') and self.world:
            del self.world

    def _cleanup(self, ego=False):
        """
        Remove and destroy all actors
        """

        self.client.stop_recorder()

        CarlaDataProvider.cleanup()
        CarlaActorPool.cleanup()

        for i, _ in enumerate(self.ego_vehicles):
            if self.ego_vehicles[i]:
                if ego:
                    self.ego_vehicles[i].destroy()
                self.ego_vehicles[i] = None
        self.ego_vehicles = []

        if hasattr(self, 'agent_instance') and self.agent_instance:
            self.agent_instance.destroy()
            self.agent_instance = None

    def _prepare_ego_vehicles(self, ego_vehicles, wait_for_ego_vehicles=False):
        """
        Spawn or update the ego vehicles
        """

        if not wait_for_ego_vehicles:
            for vehicle in ego_vehicles:
                self.ego_vehicles.append(CarlaActorPool.setup_actor(vehicle.model,
                                                                    vehicle.transform,
                                                                    vehicle.rolename,
                                                                    True,
                                                                    color=vehicle.color,
                                                                    vehicle_category=vehicle.category))
        else:
            ego_vehicle_missing = True
            while ego_vehicle_missing:
                self.ego_vehicles = []
                ego_vehicle_missing = False
                for ego_vehicle in ego_vehicles:
                    ego_vehicle_found = False
                    carla_vehicles = CarlaDataProvider.get_world().get_actors().filter('vehicle.*')
                    for carla_vehicle in carla_vehicles:
                        if carla_vehicle.attributes['role_name'] == ego_vehicle.rolename:
                            ego_vehicle_found = True
                            self.ego_vehicles.append(carla_vehicle)
                            break
                    if not ego_vehicle_found:
                        ego_vehicle_missing = True
                        break

            for i, _ in enumerate(self.ego_vehicles):
                self.ego_vehicles[i].set_transform(ego_vehicles[i].transform)

        # sync state
        CarlaDataProvider.get_world().tick()

    def _load_and_wait_for_world(self, args, town, ego_vehicles=None):
        """
        Load a new CARLA world and provide data to CarlaActorPool and CarlaDataProvider
        """

        self.world = self.client.load_world(town)
        settings = self.world.get_settings()
        settings.fixed_delta_seconds = 1.0 / self.frame_rate
        settings.synchronous_mode = True
        self.world.apply_settings(settings)

        CarlaActorPool.set_client(self.client)
        CarlaActorPool.set_world(self.world)
        CarlaDataProvider.set_world(self.world)

        spectator = CarlaDataProvider.get_world().get_spectator()
        spectator.set_transform(carla.Transform(carla.Location(x=0, y=0,z=20), carla.Rotation(pitch=-90)))

        # Wait for the world to be ready
        if self.world.get_settings().synchronous_mode:
            self.world.tick()
        else:
            self.world.wait_for_tick()

        if CarlaDataProvider.get_map().name != town:
            print("The CARLA server uses the wrong map!")
            print("This scenario requires to use map {}".format(town))
            return False

        return True

    def _load_and_run_scenario(self, args, config,data_folder,route_folder,k,model_path,fault_type,traffic_amount,image_folder,sensor_faults_file):
        """
        Load and run the scenario given by config
        """
        print("0")
        if not self._load_and_wait_for_world(args, config.town, config.ego_vehicles):
            self._cleanup()
            return

        agent_class_name = getattr(self.module_agent, 'get_entry_point')()
        try:
            self.agent_instance = getattr(self.module_agent, agent_class_name)(args.agent_config,data_folder,route_folder,k,model_path,fault_type,image_folder,sensor_faults_file)
            config.agent = self.agent_instance
            self.sensors = [sensors_to_icons[sensor['type']] for sensor in self.agent_instance.sensors()]
        except Exception as e:
            print("Could not setup required agent due to {}".format(e))
            self._cleanup()
            return

        # Prepare scenario
        #print("Preparing scenario: " + config.name)

        try:
            self._prepare_ego_vehicles(config.ego_vehicles, False)
            scenario = RouteScenario(traffic_amount, world=self.world, config=config, debug_mode=args.debug)

        except Exception as exception:
            print("The scenario cannot be loaded")
            if args.debug:
                traceback.print_exc()
            print(exception)
            self._cleanup()
            return

        # Set the appropriate weather conditions
        weather = carla.WeatherParameters(
            cloudiness=config.weather.cloudiness,
            precipitation=config.weather.precipitation,
            precipitation_deposits=config.weather.precipitation_deposits,
            wind_intensity=config.weather.wind_intensity,
            sun_azimuth_angle=config.weather.sun_azimuth_angle,
            sun_altitude_angle=config.weather.sun_altitude_angle,
            fog_density=config.weather.fog_density,
            fog_distance=config.weather.fog_distance,
            wetness=config.weather.wetness
        )

        self.world.set_weather(weather)

        # Set the appropriate road friction
        if config.friction is not None:
            friction_bp = self.world.get_blueprint_library().find('static.trigger.friction')
            extent = carla.Location(1000000.0, 1000000.0, 1000000.0)
            friction_bp.set_attribute('friction', str(config.friction))
            friction_bp.set_attribute('extent_x', str(extent.x))
            friction_bp.set_attribute('extent_y', str(extent.y))
            friction_bp.set_attribute('extent_z', str(extent.z))

            # Spawn Trigger Friction
            transform = carla.Transform()
            transform.location = carla.Location(-10000.0, -10000.0, 0.0)
            self.world.spawn_actor(friction_bp, transform)

        # night mode
        if config.weather.sun_altitude_angle < 0.0:
            for vehicle in scenario.ego_vehicles:
                vehicle.set_light_state(carla.VehicleLightState(self._vehicle_lights))

        try:
            # Load scenario and run it
            if args.record:
                self.client.start_recorder("/home/carla/data/collision_data.log".format(args.record, config.name))
            self.manager.load_scenario(scenario, self.agent_instance)    #scenarionmanager.py
            self.statistics_manager.set_route(config.name, config.index, scenario.scenario)
            distance_path = data_folder + "distance.csv"
            self.manager.run_scenario(distance_path)#This is where the scenario runs scenariomanager.py
            # Stop scenario
            self.manager.stop_scenario()
            print(self.client.show_recorder_collisions("/home/carla/data/collision_data.log", "v", "a"))

            #file1 = open("/home/carla/data/collision_data.txt", "a")
            #file1.writelines(self.client.show_recorder_collisions("/home/carla/data/collision_data.txt", "v", "a"))
            #file1.close()

            # register statistics
            current_stats_record = self.statistics_manager.compute_route_statistics(config,
                                                                                    self.manager.scenario_duration_system,
                                                                                    self.manager.scenario_duration_game)
            # save
            self.statistics_manager.save_record(current_stats_record, config.index, args.checkpoint)

            # Remove all actors
            scenario.remove_all_actors()

            settings = self.world.get_settings()
            settings.synchronous_mode = False
            settings.fixed_delta_seconds = None
            self.world.apply_settings(settings)

        except SensorConfigurationInvalid as e:
            self._cleanup(True)
            sys.exit(-1)
        except Exception as e:
            if args.debug:
                traceback.print_exc()
            print(e)

        self._cleanup()

    def read_simulation_param(self,route_folder,x):
        """
        Get the traffic and fault type information information from the route folder
        """
        hyperparameters = []
        parameters = pd.read_csv(route_folder + "/scene_parameters.csv", usecols = [0,1], header=None, index_col=False)
        for index, row in parameters.iterrows():
            if row[0] == x:
                hyperparameters.append(int(row[1]))

        return hyperparameters[0]

    def process_simulation_data(self,data_folder,path,ood_stats_file):
        """
        Get hyperparameters and collision data from previous run
        """
        stats = []
        #data_folder = "/home/scope/Carla/carla-dockers/simulation-data/simulation%d/scene%d"%(y,simulation_run-1)
        simulation_stats = pd.read_csv(data_folder + "/run1.csv", usecols = ['monitor_result','risk'], index_col=False)
        martingale_value = simulation_stats['monitor_result'].mean()
        risk_value = simulation_stats['risk'].mean()
        stats.append(round(float(martingale_value),2))
        stats.append(round(float(risk_value),2))
        with open(ood_stats_file, 'a') as csvfile: #Always save the selected hyperparameters for optimization algorithms
            writer = csv.writer(csvfile, delimiter = ',')
            writer.writerow(stats)

        return round(float(risk_value),2)

    def compute_scenario_score(self,risk,collision, infractions, out_of_lane,scenario_score_file):
        stats = []
        #scenario_score = 0.5 * risk + 0.25 * float(infractions) + 0.25 * float(out_of_lane)
        scenario_score = risk + float(infractions) + float(out_of_lane)
        stats.append(round(scenario_score,2))
        with open(scenario_score_file, 'a') as csvfile: #Always save the selected hyperparameters for optimization algorithms
            writer = csv.writer(csvfile, delimiter = ',')
            writer.writerow(stats)

        print("---------------------")
        print("Avg Scene Risk:%0.2f"%risk)
        print("Scene Score:%0.2f"%round(scenario_score,2))
        print("---------------------")


    def run(self, path,args,data_folder,route_folder,model_path,image_folder):
        """
        Run the challenge mode
        """
        #value = 0
        traffic_amount = self.read_simulation_param(route_folder,x='traffic_density')
        fault_type = self.read_simulation_param(route_folder,x='sensor_faults')
        #print("---------------------")
        #print("fault type:%d"%fault_type)
        #print("traffic Amount:%d"%traffic_amount)

        filename = data_folder + "simulation_data.csv"
        stats_file = path + "collision_stats.csv"
        ood_stats_file = path + "ood_stats.csv"
        scenario_score_file = path + "scenario_score.csv"
        sensor_faults_file = data_folder + "fault_data.csv"
        route_indexer = RouteIndexer(args.routes, args.scenarios, args.repetitions,route_folder)
        if args.resume:
            route_indexer.resume(args.checkpoint)
            self.statistics_manager.resume(args.checkpoint)
        else:
            self.statistics_manager.clear_record(args.checkpoint)
            k=0
        while route_indexer.peek():
            k+=1
            # setup
            config = route_indexer.next()
            # run
            self._load_and_run_scenario(args, config,data_folder,route_folder,k,model_path,fault_type,traffic_amount,image_folder,sensor_faults_file)

            self._cleanup(ego=True)

            route_indexer.save_state(args.checkpoint)

            # save global statistics
            global_stats_record = self.statistics_manager.compute_global_statistics(route_indexer.total)
            collision, infractions, out_of_lane = StatisticsManager.save_global_record(global_stats_record, self.sensors, args.checkpoint,filename,stats_file,config)
            risk = self.process_simulation_data(data_folder,path,ood_stats_file)
            self.compute_scenario_score(risk,collision, infractions, out_of_lane,scenario_score_file)


def create_root_folder(data_path,y):
    paths = []
    folders = ["simulation-data","routes","dummy_images"]
    for folder in folders:
        new_path = data_path + folder + "/" + "simulation%s"%y + "/"
        if not os.path.exists(new_path):
            os.makedirs(new_path, exist_ok=True) #creates a new dir everytime with max number
        paths.append(new_path)

    return paths

def read_folder_number(path):
    """
    Write the folder number in which the routes are stored
    """
    path = path + "routes" + "/"
    file1 = open(path + "tmp.txt", "r")
    y = file1.read()
    file1.close() #to change file access modes
    os.remove(path + "tmp.txt")

    return y


def main():
    description = "CARLA AD Leaderboard Evaluation: evaluate your Agent in CARLA scenarios\n"

    # general parameters
    parser = argparse.ArgumentParser(description=description, formatter_class=RawTextHelpFormatter)
    parser.add_argument('--host', default='localhost',
                        help='IP of the host server (default: localhost)')
    parser.add_argument('--port', default='3000', help='TCP port to listen to (default: 3000)')
    parser.add_argument('--debug', type=int, help='Run with debug output', default=0)
    parser.add_argument('--spectator', type=bool, help='Switch spectator view on?', default=True)
    parser.add_argument('--record', type=str, default='',
                        help='Use CARLA recording feature to create a recording of the scenario')
    parser.add_argument('--timeout', default="30.0",
                        help='Set the CARLA client timeout value in seconds')

    # simulation setup
    parser.add_argument('--challenge-mode', action="store_true", help='Switch to challenge mode?')
    parser.add_argument('--routes',
                        help='Name of the route to be executed. Point to the route_xml_file to be executed.',
                        required=True)
    parser.add_argument('--scenarios',
                        help='Name of the scenario annotation file to be mixed with the route.',
                        required=True)
    parser.add_argument('--repetitions',
                        type=int,
                        default=1,
                        help='Number of repetitions per route.')

    # agent-related options
    parser.add_argument("-a", "--agent", type=str, help="Path to Agent's py file to evaluate", required=True)
    parser.add_argument("--agent-config", type=str, help="Path to Agent's configuration file", default="")

    parser.add_argument("--track", type=str, default='SENSORS', help="Participation track: SENSORS, MAP")
    parser.add_argument('--resume', type=bool, default=False, help='Resume execution from last checkpoint?')
    parser.add_argument("--checkpoint", type=str,
                        default='./simulation_results.json',
                        help="Path to checkpoint used for saving statistics and resuming")
    parser.add_argument('--simulation_number', type=int, help='Type the simulation folder to store the data')
    parser.add_argument('--scene_number', type=int, default=1, help='Type the scene number to be executed')
    parser.add_argument('--project_path', type=str, help='Type the simulation folder to store the data')

    arguments = parser.parse_args()

    statistics_manager = StatisticsManager()

    try:
        data_path = arguments.project_path
        y = read_folder_number(data_path)
        paths = create_root_folder(data_path,y)
        #paths,y = create_root_folder(data_path,arguments.simulation_number)
        model_path = data_path + "carla-challange/leaderboard/detector_code/ood_detector_weights/"
        #model_path = "/home/scope/Carla/carla-dockers/carla-challange/leaderboard/team_code/detector_code/trial2/center-B-1.2/"#"/home/scope/Carla/carla-dockers/carla-challange/leaderboard/team_code/detector_code/trial2/center-B-1.2-trial/"#
        #model_path = "/home/scope/Carla/carla-dockers/carla-challange/leaderboard/team_code/detector_code/trial1/old/center-B-1.2/"
        data_folder = paths[0] + "scene%d"%arguments.simulation_number + '/'
        os.makedirs(data_folder, exist_ok=True)
        route_folder = paths[1] + "scene%d"%arguments.simulation_number + '/'
        os.makedirs(route_folder, exist_ok=True)
        image_folder = paths[2] + "scene%d"%arguments.simulation_number + '/'
        os.makedirs(image_folder, exist_ok=True)
        leaderboard_evaluator = LeaderboardEvaluator(arguments, statistics_manager)
        #for i in range(0,15):
        leaderboard_evaluator.run(paths[0],arguments,data_folder,route_folder,model_path,image_folder)
            #data_plotting(path,data_folder)

    except Exception as e:
        traceback.print_exc()
    finally:
        del leaderboard_evaluator


if __name__ == '__main__':
    main()
