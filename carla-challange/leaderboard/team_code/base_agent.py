import time
import csv
import cv2
import carla
import os
from leaderboard.autoagents import autonomous_agent
from team_code.planner import RoutePlanner
import numpy as np
import random
import utils

def get_fault_list(fault_type):
    if(fault_type == 3):
        return (2,3)
    if(fault_type == 4):
        return (1,3)
    if(fault_type == 5):
        return (1,2)
    if(fault_type == 6):
        return (1,2,3)
    if(fault_type == 10):
        return (9,10)
    if(fault_type == 11):
        return (8,10)
    if(fault_type == 12):
        return (8,9)
    if(fault_type == 13):
        return (8,9,10)
    else:
        return fault_type

class BaseAgent(autonomous_agent.AutonomousAgent):
    def setup(self, path_to_conf_file,data_folder,route_folder,k,model_path,fault_type):
        self.track = autonomous_agent.Track.SENSORS
        self.config_path = path_to_conf_file
        self.data_folder = data_folder
        self.scene_number = k
        #self.failure_mode = i
        self.value = 0
        self.filename =    self.data_folder + "/fault_data.csv"  #"/home/scope/Carla/ICCPS_CARLA_challenge/leaderboard/data/my_data/Simulation6/fault_data.csv"
        self.step = -1
        self.wall_start = time.time()
        self.initialized = False
        self.fault_type_list = None
        self.fault_scenario = 1 #random.randint(0,1)
        self.fault_step = random.randint(100,150)
        self.fault_time = random.randint(150,200)
        self.fields = ['fault_scenario',
                        'fault_step',
                        'fault_time',
                        'fault_type',
                        'fault_list',
                        'brightness_value'
                        ]

        if(self.fault_scenario == 0):
            self.fault_type = 0
            self.fault_type_list = -1
        elif(self.fault_scenario == 1):
            self.fault_type =  fault_type #15 #random.randint(0,13)
            self.fault_type_list = get_fault_list(self.fault_type)

        self.dict = [{'fault_scenario':self.fault_scenario,'fault_step':self.fault_step,'fault_time':self.fault_time,'fault_type':self.fault_type,'fault_list':self.fault_type_list,'brightness_value':self.value}]

        file_exists = os.path.isfile(self.filename)

        # writing to csv file
        with open(self.filename, 'a') as csvfile:
            # creating a csv dict writer object
            writer = csv.DictWriter(csvfile, fieldnames = self.fields)

            if not file_exists:
                # writing headers (field names)
                writer.writeheader()
            # writing data rows
            writer.writerows(self.dict)

    def _init(self):
        self._command_planner = RoutePlanner(7.5, 25.0, 257)
        self._command_planner.set_route(self._global_plan, True)

        self.initialized = True

    def _get_position(self, tick_data):
        gps = tick_data['gps']
        gps = (gps - self._command_planner.mean) * self._command_planner.scale

        return gps

    def sensors(self):
        return [
                {
                    'type': 'sensor.camera.rgb',
                    'x': 1.3, 'y': 0.0, 'z': 1.3,
                    'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                    'width': 256, 'height': 256, 'fov': 90,
                    'id': 'rgb'
                    },
                {
                    'type': 'sensor.camera.rgb',
                    'x': 1.2, 'y': -0.25, 'z': 1.3,
                    'roll': 0.0, 'pitch': 0.0, 'yaw': -45.0,
                    'width': 256, 'height': 256, 'fov': 90,
                    'id': 'rgb_left'
                    },
                {
                    'type': 'sensor.camera.rgb',
                    'x': 1.2, 'y': 0.25, 'z': 1.3,
                    'roll': 0.0, 'pitch': 0.0, 'yaw': 45.0,
                    'width': 256, 'height': 256, 'fov': 90,
                    'id': 'rgb_right'
                    },
                {
                    'type': 'sensor.other.imu',
                    'x': 0.0, 'y': 0.0, 'z': 0.0,
                    'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                    'sensor_tick': 0.05,
                    'id': 'imu'
                    },
                {
                    'type': 'sensor.other.radar',
                    'x': 2.8, 'y': 0.0, 'z': 1.0,
                    'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,'fov': 25,
                    'sensor_tick': 0.05,
                    'id': 'radar'
                    },
                {
                    'type': 'sensor.other.gnss',
                    'x': 0.0, 'y': 0.0, 'z': 0.0,
                    'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                    'sensor_tick': 0.01,
                    'id': 'gps'
                    },
                {
                    'type': 'sensor.speedometer',
                    'reading_frequency': 20,
                    'id': 'speed'
                    }
                ]

    def fault_list(self,rgb,rgb_left,rgb_right,points,gps):
        if(self.fault_type == 0):
            #print("Blur Image")
            rgb = rgb
        elif(self.fault_type == 1):
            #print("Blur Image")
            rgb = cv2.blur(rgb,(10,10))
        elif(self.fault_type == 2):
            #print("Blur Image")
            rgb_left = cv2.blur(rgb_left,(10,10))
        elif(self.fault_type == 3):
            #print("Blur Image")
            rgb_right = cv2.blur(rgb_right,(10,10))
        elif(self.fault_type == 4):
            #print("Blur Image")
            rgb_left = cv2.blur(rgb_left,(10,10))
            rgb_right = cv2.blur(rgb_right,(10,10))
        elif(self.fault_type == 5):
            #print("Blur Image")
            rgb = cv2.blur(rgb,(10,10))
            rgb_right = cv2.blur(rgb_right,(10,10))
        elif(self.fault_type == 6):
            #print("Blur Image")
            rgb = cv2.blur(rgb,(10,10))
            rgb_left = cv2.blur(rgb_left,(10,10))
        elif(self.fault_type == 7):
            #print("Blur Image")
            rgb = cv2.blur(rgb,(10,10))
            rgb_right = cv2.blur(rgb_right,(10,10))
            rgb_left = cv2.blur(rgb_left,(10,10))
        if(self.fault_type == 8):
            #print("Center Camera Image Occluded")
            h, w, _ = rgb.shape
            rgb = cv2.rectangle(rgb, (0, 0), (w // 4, h), (0, 0, 0), cv2.FILLED)
        elif(self.fault_type == 9):
            #print("Left Camera Image Occluded")
            h, w, _ = rgb_left.shape
            rgb_left = cv2.rectangle(rgb_left, (0, 0), (w // 4, h), (0, 0, 0), cv2.FILLED)
        elif(self.fault_type == 10):
            #print("Right Camera Image Occluded")
            h, w, _ = rgb_right.shape
            rgb_right = cv2.rectangle(rgb_right, (0, 0), (w // 2, h), (0, 0, 0), cv2.FILLED)
        elif(self.fault_type == 11):
            #print("Right & left Camera Images Occluded")
            h, w, _ = rgb_right.shape
            rgb_right = cv2.rectangle(rgb_right, (0, 0), (w // 4, h), (0, 0, 0), cv2.FILLED)
            rgb_left = cv2.rectangle(rgb_left, (0, 0), (w // 4, h), (0, 0, 0), cv2.FILLED)
        elif(self.fault_type == 12):
            #print("Right & center Camera Images Occluded")
            h, w, _ = rgb_right.shape
            rgb_right = cv2.rectangle(rgb_right, (0, 0), (w // 4, h), (0, 0, 0), cv2.FILLED)
            rgb = cv2.rectangle(rgb, (0, 0), (w // 4, h), (0, 0, 0), cv2.FILLED)
        elif(self.fault_type == 13):
            #print("center & left Camera Images Occluded")
            h, w, _ = rgb.shape
            rgb = cv2.rectangle(rgb, (0, 0), (w // 4, h), (0, 0, 0), cv2.FILLED)
            rgb_left = cv2.rectangle(rgb_left, (0, 0), (w // 4, h), (0, 0, 0), cv2.FILLED)
        elif(self.fault_type == 14):
            #print("All the Camera Images are Occluded")
            h, w, _ = rgb_right.shape
            rgb_right = cv2.rectangle(rgb_right, (0, 0), (w // 4, h), (0, 0, 0), cv2.FILLED)
            rgb_left = cv2.rectangle(rgb_left, (0, 0), (w // 4, h), (0, 0, 0), cv2.FILLED)
            rgb = cv2.rectangle(rgb, (0, 0), (w // 4, h), (0, 0, 0), cv2.FILLED)
        elif(self.fault_type == 15):
            self.value = 40
            #gps = gps
            #noise = np.random.normal(0, .001, gps.shape)
            #gps += noise

        return rgb, rgb_left, rgb_right, points, gps, self.value

    def add_brightness(self,rgb,rgb_left,rgb_right):
        hsv = cv2.cvtColor(rgb_right, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        lim = 255 - self.value
        v[v > lim] = 255
        v[v <= lim] += self.value
        final_hsv = cv2.merge((h, s, v))
        rgb_right = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
        hsv = cv2.cvtColor(rgb, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        lim = 255 - self.value
        v[v > lim] = 255
        v[v <= lim] += self.value
        final_hsv = cv2.merge((h, s, v))
        rgb = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
        hsv = cv2.cvtColor(rgb_left, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        lim = 255 - self.value
        v[v > lim] = 255
        v[v <= lim] += self.value
        final_hsv = cv2.merge((h, s, v))
        rgb_left = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)

        return rgb,rgb_left,rgb_right

    def tick(self, input_data):
        self.step += 1

        rgb = cv2.cvtColor(input_data['rgb'][1][:, :, :3], cv2.COLOR_BGR2RGB)
        rgb_left = cv2.cvtColor(input_data['rgb_left'][1][:, :, :3], cv2.COLOR_BGR2RGB)
        rgb_right = cv2.cvtColor(input_data['rgb_right'][1][:, :, :3], cv2.COLOR_BGR2RGB)
        gps = input_data['gps'][1][:2]
        speed = input_data['speed'][1]['speed']
        compass = input_data['imu'][1][-1]
        radar_data = (input_data['radar'][1])
        #print(radar_data)
        points = np.reshape(radar_data, (len(radar_data), 4))
        #print(points[0][0])

        if(self.fault_scenario == 1):
            if (self.step > self.fault_step and self.step<self.fault_step + self.fault_time):
                    rgb,rgb_left,rgb_right,points,gps,self.value=self.fault_list(rgb,rgb_left,rgb_right,points,gps)
                    rgb,rgb_left,rgb_right = self.add_brightness(rgb,rgb_left,rgb_right)
                #noise = np.random.normal(0, .1, gps.shape)
                #gps += noise

        return {
                'rgb': rgb,
                'rgb_left': rgb_left,
                'rgb_right': rgb_right,
                'gps': gps,
                'speed': speed,
                'compass': compass,
                'cloud':points,
                'fault_scenario': self.fault_scenario,
                'fault_step': self.fault_step,
                'fault_duration': self.fault_time,
                'fault_type': self.fault_type
                }
