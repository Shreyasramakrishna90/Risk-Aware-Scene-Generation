#!/usr/bin/env python

# Copyright (c) 2018-2020 Intel Corporation
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
This module provides the ScenarioManager implementations.
It must not be modified and is for reference only!
"""

from __future__ import print_function
import signal
import sys
import time
import datetime
import py_trees
import carla
import os
import csv
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from srunner.scenariomanager.timer import GameTime
from srunner.scenariomanager.watchdog import Watchdog

from leaderboard.autoagents.agent_wrapper import AgentWrapper, AgentError
from leaderboard.envs.sensor_interface import SensorReceivedNoData
from leaderboard.utils.result_writer import ResultOutputProvider


class ScenarioManager(object):

    """
    Basic scenario manager class. This class holds all functionality
    required to start, run and stop a scenario.

    The user must not modify this class.

    To use the ScenarioManager:
    1. Create an object via manager = ScenarioManager()
    2. Load a scenario via manager.load_scenario()
    3. Trigger the execution of the scenario manager.run_scenario()
       This function is designed to explicitly control start and end of
       the scenario execution
    4. If needed, cleanup with manager.stop_scenario()
    """


    def __init__(self, timeout, debug_mode=False):
        """
        Setups up the parameters, which will be filled at load_scenario()
        """
        self.scenario = None
        self.scenario_tree = None
        self.scenario_class = None
        self.ego_vehicles = None
        self.other_actors = None

        self._debug_mode = debug_mode
        self._agent = None
        self._running = False
        self._timestamp_last_run = 0.0
        self._timeout = float(timeout)

        # Used to detect if the simulation is down
        watchdog_timeout = max(5, self._timeout - 2)
        self._watchdog = Watchdog(watchdog_timeout)

        # Avoid the agent from freezing the simulation
        agent_timeout = watchdog_timeout - 1
        self._agent_watchdog = Watchdog(agent_timeout)

        self.scenario_duration_system = 0.0
        self.scenario_duration_game = 0.0
        self.start_system_time = None
        self.end_system_time = None
        self.end_game_time = None

        # Register the scenario tick as callback for the CARLA world
        # Use the callback_id inside the signal handler to allow external interrupts
        signal.signal(signal.SIGINT, self.signal_handler)

    def signal_handler(self, signum, frame):
        """
        Terminate scenario ticking when receiving a signal interrupt
        """
        self._running = False

    def cleanup(self):
        """
        Reset all parameters
        """
        self._timestamp_last_run = 0.0
        self.scenario_duration_system = 0.0
        self.scenario_duration_game = 0.0
        self.start_system_time = None
        self.end_system_time = None
        self.end_game_time = None

    def load_scenario(self, scenario, agent, rep_number):
        """
        Load a new scenario
        """

        GameTime.restart()
        self._agent = AgentWrapper(agent)
        self.scenario_class = scenario
        self.scenario = scenario.scenario
        self.scenario_tree = self.scenario.scenario_tree
        self.ego_vehicles = scenario.ego_vehicles
        self.other_actors = scenario.other_actors
        self.repetition_number = rep_number

        # To print the scenario tree uncomment the next line
        # py_trees.display.render_dot_tree(self.scenario_tree)

        self._agent.setup_sensors(self.ego_vehicles[0], self._debug_mode)

    def run_scenario(self,distance_path):
        """
        Trigger the start of the scenario and wait for it to finish/fail
        """
        self.start_system_time = time.time()
        self.start_game_time = GameTime.get_time()

        self._watchdog.start()
        self._running = True

        self.fields = [
                        "Ground Truth"
                    ]

        while self._running:

            ego_location = CarlaDataProvider.get_hero_actor().get_transform().location
            ego_direction = ego_location + CarlaDataProvider.get_hero_actor().get_transform().get_forward_vector()*15
            offset = 2
            obstacle = []
            dis = float("inf")
            for actor in self.other_actors:
                actor_loc =  actor.get_transform().location
                if ((min(ego_location.x,ego_direction.x)-offset) <= actor_loc.x <= (max(ego_location.x,ego_direction.x) +offset)) and ((min(ego_location.y,ego_direction.y)-offset) <= actor_loc.y <= (max(ego_location.y,ego_direction.y)+offset)):
                    obstacle.append(actor.get_transform().location.distance(ego_location))
            if len(obstacle) > 0:
                dis = min(obstacle)

            timestamp = None
            world = CarlaDataProvider.get_world()
            if world:
                snapshot = world.get_snapshot()
                if snapshot:
                    timestamp = snapshot.timestamp
                    #print(timestamp, datetime.datetime.now())
            if timestamp:
                self._tick_scenario(timestamp)

            dict = [{'Ground Truth': str(dis)}]
            file_exists = os.path.isfile(distance_path)
            with open(distance_path, 'a') as csvfile:
                # creating a csv dict writer object
                writer = csv.DictWriter(csvfile, fieldnames = self.fields)
                if not file_exists:
                    writer.writeheader()
                writer.writerows(dict)

        self._console_message()


    def _console_message(self):
        """
        Message that will be displayed via console
        """
        def get_symbol(value, desired_value, high=True):
            """
            Returns a tick or a cross depending on the values
            """
            tick = '\033[92m'+'O'+'\033[0m'
            cross = '\033[91m'+'X'+'\033[0m'

            multiplier = 1 if high else -1

            if multiplier*value >= desired_value:
                symbol = tick
            else:
                symbol = cross

            return symbol

        if self.scenario_tree.status == py_trees.common.Status.RUNNING:
            # If still running, all the following is None, so no point continuing
            print("\n> Something happened during the simulation. Was it manually shutdown?\n")
            return

        blackv = py_trees.blackboard.Blackboard()
        route_completed = blackv.get("RouteCompletion")
        collisions = blackv.get("Collision")
        outside_route_lanes = blackv.get("OutsideRouteLanes")
        stop_signs = blackv.get("RunningStop")
        red_light = blackv.get("RunningRedLight")
        in_route = blackv.get("InRoute")

        # If something failed, stop
        if [x for x in (collisions, outside_route_lanes, stop_signs, red_light, in_route) if x is None]:
            return

        if blackv.get("RouteCompletion") >= 99:
            route_completed = 100
        else:
            route_completed = blackv.get("RouteCompletion")
        outside_route_lanes = float(outside_route_lanes)

        route_symbol = get_symbol(route_completed, 100, True)
        collision_symbol = get_symbol(collisions, 0, False)
        outside_symbol = get_symbol(outside_route_lanes, 0, False)
        red_light_symbol = get_symbol(red_light, 0, False)
        stop_symbol = get_symbol(stop_signs, 0, False)
        #print(self.scenario_tree.status,py_trees.common.Status.FAILURE,py_trees.common.Status.SUCCESS)
        if self.scenario_tree.status == py_trees.common.Status.FAILURE:
            if not in_route:
                message = "> FAILED: The actor deviated from the route"
            else:
                message = "> FAILED: The actor didn't finish the route"
        elif self.scenario_tree.status == py_trees.common.Status.SUCCESS:
            if route_completed == 100:
                message = "> SUCCESS: Congratulations, route finished! "
            else:
                message = "> FAILED: The actor timed out "
        else: # This should never be triggered
            return

        if self.scenario_tree.status != py_trees.common.Status.RUNNING:
            print("\n" + message)
            print("> ")
            print("> Score: ")
            print("> - Route Completed [{}]:      {}%".format(route_symbol, route_completed))
            #print("> - Outside route lanes [{}]:  {}%".format(outside_symbol, outside_route_lanes))
            print("> - Collisions [{}]:           {} times".format(collision_symbol, math.floor(collisions)))
            #print("> - Red lights run [{}]:       {} times".format(red_light_symbol, red_light))
            #print("> - Stop signs run [{}]:       {} times\n".format(stop_symbol, stop_signs))


    def _tick_scenario(self, timestamp):
        """
        Run next tick of scenario and the agent and tick the world.
        """

        if self._timestamp_last_run < timestamp.elapsed_seconds and self._running:
            self._timestamp_last_run = timestamp.elapsed_seconds

            self._watchdog.update()
            # Update game time and actor information
            GameTime.on_carla_tick(timestamp)
            #to be called everytime a new tick is received
            CarlaDataProvider.on_carla_tick()

            try:
                ego_action = self._agent()

            # Special exception inside the agent that isn't caused by the agent
            except SensorReceivedNoData as e:
                raise RuntimeError(e)

            except Exception as e:
                raise AgentError(e)
            #print("num of ego:", len(self.ego_vehicles))
            self.ego_vehicles[0].apply_control(ego_action)

            # Tick scenario
            self.scenario_tree.tick_once()
            #print(timestamp)
            #self._debug_mode = True
            if self._debug_mode:
                print("\n")
                py_trees.display.print_ascii_tree(
                    self.scenario_tree, show_status=True)
                sys.stdout.flush()

            if self.scenario_tree.status != py_trees.common.Status.RUNNING:
                self._running = False

            #spectator = CarlaDataProvider.get_world().get_spectator()
            #ego_trans = self.ego_vehicles[0].get_transform()
            #spectator.set_transform(carla.Transform(ego_trans.location + carla.Location(z=50),
            #                                            carla.Rotation(pitch=-90)))

        if self._running and self.get_running_status():
            CarlaDataProvider.get_world().tick(self._timeout)

    def get_running_status(self):
        """
        returns:
           bool: False if watchdog exception occured, True otherwise
        """
        return self._watchdog.get_status()

    def stop_scenario(self):
        """
        This function triggers a proper termination of a scenario
        """
        self._watchdog.stop()

        self.end_system_time = time.time()
        self.end_game_time = GameTime.get_time()

        self.scenario_duration_system = self.end_system_time - self.start_system_time
        self.scenario_duration_game = self.end_game_time - self.start_game_time

        if self.get_running_status():
            if self.scenario is not None:
                self.scenario.terminate()

            if self._agent is not None:
                self._agent.cleanup()
                self._agent = None

            self.analyze_scenario()

    def analyze_scenario(self):
        """
        Analyzes and prints the results of the route
        """
        global_result = '\033[92m'+'SUCCESS'+'\033[0m'

        for criterion in self.scenario.get_criteria():
            if criterion.test_status != "SUCCESS":
                global_result = '\033[91m'+'FAILURE'+'\033[0m'

        if self.scenario.timeout_node.timeout:
            global_result = '\033[91m'+'FAILURE'+'\033[0m'

        ResultOutputProvider(self, global_result)
