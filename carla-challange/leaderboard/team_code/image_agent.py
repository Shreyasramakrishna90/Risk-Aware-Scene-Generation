import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)
import os
import cv2
import torch
import torchvision
import carla
import csv
import math
import pathlib
import datetime
import gc
import json
from numba import cuda
from PIL import Image, ImageDraw
import threading
from threading import Thread
from queue import Queue
import numpy as np
import keras
import tensorflow as tf
#import tensorflow.keras as keras
from keras.models import Model, model_from_json, load_model
from keras.backend.tensorflow_backend import set_session
from keras import backend as K
from sklearn.utils import shuffle
from keras.models import model_from_json
from sklearn.metrics import mean_squared_error
from carla_project.src.image_model import ImageModel
from carla_project.src.converter import Converter

from team_code.base_agent import BaseAgent
from team_code.pid_controller import PIDController
from detectors.anomaly_detector import occlusion_detector, blur_detector, assurance_monitor
from team_code.risk_calculation.fault_modes import FaultModes
from team_code.risk_calculation.bowtie_diagram import BowTie
from team_code.planner import RoutePlanner
from srunner.scenariomanager.carla_data_provider import CarlaActorPool
#from keras.models import load_model
from scipy.stats import norm
import scipy.integrate as integrate
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

DEBUG = int(os.environ.get('HAS_DISPLAY', 0))
os.environ["CUDA_VISIBLE_DEVICES"]="0"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def get_entry_point():
    return 'ImageAgent'


def debug_display(tick_data, target_cam, out, steer, throttle, brake, desired_speed, step):
    _rgb = Image.fromarray(tick_data['rgb'])
    _draw_rgb = ImageDraw.Draw(_rgb)
    _draw_rgb.ellipse((target_cam[0]-3,target_cam[1]-3,target_cam[0]+3,target_cam[1]+3), (255, 255, 255))

    for x, y in out:
        x = (x + 1) / 2 * 256
        y = (y + 1) / 2 * 144

        _draw_rgb.ellipse((x-2, y-2, x+2, y+2), (0, 0, 255))

    _combined = Image.fromarray(np.hstack([tick_data['rgb_left'], _rgb, tick_data['rgb_right']]))
    _draw = ImageDraw.Draw(_combined)
    _draw.text((5, 10), 'Steer: %.3f' % steer)
    _draw.text((5, 30), 'Throttle: %.3f' % throttle)
    _draw.text((5, 50), 'Brake: %s' % brake)
    _draw.text((5, 70), 'Speed: %.3f' % tick_data['speed'])
    _draw.text((5, 90), 'Desired: %.3f' % desired_speed)

    cv2.imshow('map', cv2.cvtColor(np.array(_combined), cv2.COLOR_BGR2RGB))
    cv2.waitKey(1)

def process_weather_data(weather_file,k):
    #print("problem1")
    weather = []
    lines = []
    with open(weather_file, 'r') as readFile:
        reader = csv.reader(readFile)
        for row in reader:
            weather.append(row)


    return weather[k-1],len(weather)

ENV_LABELS = ["precipitation",
              "precipitation_deposits",
              "cloudiness",
              "wind_intensity",
              "sun_azimuth_angle",
              "sun_altitude_angle",
              "fog_density",
              "fog_distance",
              "wetness"]
FAULT_LABEL = ["fault_type"]
MONITOR_LABELS = ["center_blur_dect",
                "left_blur_dect",
                "right_blur_dect",
                "center_occ_dect",
                "left_occ_dect",
                "right_occ_dect",
                "lec_martingale"]


def _location(x, y, z):
    return carla.Location(x=float(x), y=float(y), z=float(z))

def _orientation(yaw):
    return np.float32([np.cos(np.radians(yaw)), np.sin(np.radians(yaw))])


def get_collision(p1, v1, p2, v2):
    A = np.stack([v1, -v2], 1)
    b = p2 - p1

    if abs(np.linalg.det(A)) < 1e-3:
        return False, None

    x = np.linalg.solve(A, b)
    collides = all(x >= 0) and all(x <= 1)

    return collides, p1 + x[0] * v1

def _numpy(carla_vector, normalize=False):
    result = np.float32([carla_vector.x, carla_vector.y])

    if normalize:
        return result / (np.linalg.norm(result) + 1e-4)

    return result

class ImageAgent(BaseAgent):
    def setup(self, path_to_conf_file,data_folder,route_folder,k,model_path,fault_type,image_folder,sensor_faults_file):
        super().setup(path_to_conf_file,data_folder,route_folder,k,model_path,fault_type,image_folder,sensor_faults_file)
        self.converter = Converter()
        self.net = ImageModel.load_from_checkpoint(path_to_conf_file)
        self.data_folder = data_folder
        self.route_folder = route_folder
        self.image_folder = image_folder
        self.scene_number = k
        #self.failure_mode = i
        self.weather_file = self.route_folder + "/weather_data.csv"
        self.model_path = model_path #"/home/scope/Carla/autopilot_Carla_ad/leaderboard/team_code/detector_code/trial1/old/center-B-1.2/"
        #self.device = cuda.get_current_device()
        #self.device.reset()
        self.model_vae = None
        #torch.cuda.empty_cache()
        self.net.cuda()
        self.net.eval()
        self.run = 0
        self.risk = 0
        self.state = []
        self.monitors = []
        self.blur_queue = Queue(maxsize=1)
        self.occlusion_queue = Queue(maxsize=1)
        self.am_queue = Queue(maxsize=1)
        self.pval_queue = Queue(maxsize=1)
        self.sval_queue = Queue(maxsize=1)
        self.mval_queue = Queue(maxsize=1)
        self.calib_set = []
        self.result = []
        self.blur = []
        self.occlusion = []
        #self.rgb_detector = []
        self.detector_file = None
        self.detector_file = None
        K.clear_session()
        config = tf.ConfigProto()
        #config.gpu_options.allow_growth = True
        sess = tf.Session(config=tf.ConfigProto(device_count={'GPU': 0}))#tf.Session(config=config)
        set_session(sess)

        self.save_path = None#"/isis/Carla/iccps/images/"

        path = "/home/baiting1/Desktop/risk.git/images"
        if path:
            now = datetime.datetime.now()
            string = pathlib.Path(os.environ['ROUTES']).stem + '_'
            string += '_'.join(map(lambda x: '%02d' % x, (now.month, now.day, now.hour, now.minute, now.second)))

            self.save_path = pathlib.Path(path) / string
            self.save_path.mkdir(exist_ok=False)

        (self.save_path / 'rgb').mkdir()
        (self.save_path / 'rgb_left').mkdir()
        (self.save_path / 'rgb_right').mkdir()
        (self.save_path / 'topdown').mkdir()
        (self.save_path / 'measurements').mkdir()
        (self.save_path / 'rgb_detector').mkdir()

        with open(self.model_path + 'auto_model.json', 'r') as f:
            self.model_vae = model_from_json(f.read())
        self.model_vae.load_weights(self.model_path + 'auto_model.h5')

        self.model_vae._make_predict_function()
        self.fields = ['step',
                      'monitor_result',
                      'risk',
                       'rgb_blur',
                       #'rgb_blur_percent',
                       'rgb_left_blur',
                       #'rgb_left_blur_percent',
                       'rgb_right_blur',
                       #'rgb_right_blur_percent',
                       'rgb_occluded',
                       #'rgb_occluded_percent',
                       'rgb_left_occluded',
                       #'rgb_left_occluded_percent',
                       'rgb_right_occluded',
                       #'rgb_right_occluded_percent'
                     ]
        self.weather,self.run_number = process_weather_data(self.weather_file,self.scene_number)

        with open(self.model_path+ 'calibration.csv', 'r') as file:
            reader = csv.reader(file)
            for row in reader:
                self.calib_set.append(row)

    def _init(self):
        super()._init()
        self._turn_controller = PIDController(K_P=1.25, K_I=0.75, K_D=0.3, n=40)
        self._speed_controller = PIDController(K_P=5.0, K_I=0.5, K_D=1.0, n=40)
        self._waypoint_planner = RoutePlanner(4.0, 50)
        self._waypoint_planner.set_route(self._plan_gps_HACK, True)
        self._vehicle = CarlaActorPool.get_hero_actor()
        self._world = self._vehicle.get_world()

    def blur_detection(self,result):
        self.blur =[]
        fm1,rgb_blur = blur_detector(result['rgb'], threshold=20)
        fm2,rgb_left_blur = blur_detector(result['rgb_left'], threshold=20)
        fm3,rgb_right_blur = blur_detector(result['rgb_right'], threshold=20)
        self.blur.append(rgb_blur)
        #self.blur.append(fm1)
        self.blur.append(rgb_left_blur)
        #self.blur.append(fm2)
        self.blur.append(rgb_right_blur)
        #self.blur.append(fm3)
        self.blur_queue.put(self.blur)


    def occlusion_detection(self,result):
        self.occlusion = []
        percent1,rgb_occluded = occlusion_detector(result['rgb'], threshold=25)
        percent2,rgb_left_occluded = occlusion_detector(result['rgb_left'], threshold=25)
        percent3,rgb_right_occluded = occlusion_detector(result['rgb_right'], threshold=25)
        self.occlusion.append(rgb_occluded)
        #self.occlusion.append(percent1)
        self.occlusion.append(rgb_left_occluded)
        #self.occlusion.append(percent2)
        self.occlusion.append(rgb_right_occluded)
        #self.occlusion.append(percent3)
        self.occlusion_queue.put(self.occlusion)

    def integrand(self,k,p_anomaly):
        result = 1.0
        for i in range(len(p_anomaly)):
            result *= k*(p_anomaly[i]**(k-1.0))
        return result

    def mse(self, imageA, imageB):
        #err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
        #err /= float(imageA.shape[0] * imageA.shape[1])
        err = np.mean(np.power(imageA - imageB, 2), axis=1)
        return err

    def assurance_monitor(self,dist):
        if(self.step == 0):
            p_anomaly = []
            prev_value = []
        else:
            p_anomaly = self.pval_queue.get()
            prev_value = self.sval_queue.get()
        anomaly=0
        m=0
        delta = 20
        #threshold = 20
        sliding_window = 5
        threshold = 20.0
        for i in range(len(self.calib_set)):
            if(float(dist) <= float(self.calib_set[i][0])):
                anomaly+=1

        p_value = anomaly/len(self.calib_set)
        if(p_value<0.005):
            p_anomaly.append(0.005)
        else:
            p_anomaly.append(p_value)
        if(len(p_anomaly))>= sliding_window:
            p_anomaly = p_anomaly[-1*sliding_window:]
        m = integrate.quad(self.integrand,0.0,1.0,args=(p_anomaly))
        m_val = round(math.log(m[0]),2)
        #print("m_val:%f"%m_val)
        if(self.step==0):
            S = 0
            S_prev = 0
        else:
            S = max(0, prev_value[0]+prev_value[1]-delta)
        prev_value = []
        S_prev = S
        m_prev = m[0]
        prev_value.append(S_prev)
        prev_value.append(m_prev)
        self.pval_queue.put(p_anomaly)
        self.sval_queue.put(prev_value)
        self.mval_queue.put(m_val)

    def risk_computation(self,weather,blur_queue,am_queue,occlusion_queue,fault_scenario,fault_type,fault_time,fault_step):
        monitors = []
        faults = []
        faults.append(fault_type)
        blur = self.blur_queue.get()
        occlusion = self.occlusion_queue.get()
        mval = self.mval_queue.get()
        monitors = blur + occlusion
        state = {"enviornment": {}, "fault_modes": None, "monitor_values": {}}
        for i in range(len(weather)):
            label = ENV_LABELS[i]
            state["enviornment"][label] = weather[i]
        state["fault_modes"] = fault_type
        for j in range(len(monitors)):
            label = MONITOR_LABELS[j]
            state["monitor_values"][label] = monitors[j]
        state["monitor_values"]["lec_martingale"] = mval
        fault_modes = state["fault_modes"]
        environment = state["enviornment"]
        monitor_values = state["monitor_values"]

        fault_modes = state["fault_modes"]

        bowtie = BowTie()
        r_t1_top = bowtie.rate_t1(state) * (1 - bowtie.prob_b1(state,fault_modes))
        r_t2_top = bowtie.rate_t2(state) * (1 - bowtie.prob_b2(state,fault_modes))
        r_top = r_t1_top + r_t2_top
        r_c1 = r_top * (1 - bowtie.prob_b3(state,fault_modes))

        print("Dynamic Risk:%f"%r_c1)

        dict = [{'step':self.step, 'monitor_result':mval, 'risk':round(r_c1,2), 'rgb_blur':blur[0],'rgb_left_blur':blur[1],'rgb_right_blur':blur[2],
        'rgb_occluded':occlusion[0],'rgb_left_occluded':occlusion[1],'rgb_right_occluded':occlusion[2]}]

        if(self.step == 0):
            self.detector_file =  self.data_folder + "/run%d.csv"%(self.scene_number)

        file_exists = os.path.isfile(self.detector_file)
        with open(self.detector_file, 'a') as csvfile:
            # creating a csv dict writer object
            writer = csv.DictWriter(csvfile, fieldnames = self.fields)
            if not file_exists:
                writer.writeheader()
            writer.writerows(dict)

    def tick(self, input_data):

        result = super().tick(input_data)
        #print(self.step)
        result['rgb_detector'] = cv2.resize(result['rgb'],(224,224))
        result['rgb_detector_left'] = cv2.resize(result['rgb_left'],(224,224))
        result['rgb_detector_right'] = cv2.resize(result['rgb_right'],(224,224))
        result['rgb'] = cv2.resize(result['rgb'],(256,144))
        result['rgb_left'] = cv2.resize(result['rgb_left'],(256,144))
        result['rgb_right'] = cv2.resize(result['rgb_right'],(256,144))
        result['rgb_detector'] = cv2.cvtColor(result['rgb_detector_right'], cv2.COLOR_RGB2BGR)
        cv2.imshow('Agent', result['rgb_detector'])
        cv2.waitKey(1)
        #detection_image = result['rgb_detector'] / 255.
        detection_image = result['rgb_detector']/ 255.
        detection_image = np.reshape(detection_image, [-1, detection_image.shape[0],detection_image.shape[1],detection_image.shape[2]])
        #img = np.array([detection_image])
        predicted_reps = self.model_vae.predict_on_batch(detection_image)
        dist = np.square(np.subtract(np.array(predicted_reps),detection_image)).mean() #- 0.001
        BlurDetectorThread = Thread(target=self.blur_detection, args=(result,))
        BlurDetectorThread.daemon = True
        OccusionDetectorThread = Thread(target=self.occlusion_detection, args=(result,))
        OccusionDetectorThread.daemon = True
        AssuranceMonitorThread = Thread(target=self.assurance_monitor, args=(dist,)) #image,model,calibration_set,pval_queue,sval_queue
        AssuranceMonitorThread.daemon = True

        #Start threads for parallel computation
        AssuranceMonitorThread.start()
        BlurDetectorThread.start()
        OccusionDetectorThread.start()

        result['image'] = np.concatenate(tuple(result[x] for x in ['rgb', 'rgb_left', 'rgb_right']), -1)

        theta = result['compass']
        theta = 0.0 if np.isnan(theta) else theta
        theta = theta + np.pi / 2
        R = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta),  np.cos(theta)],
            ])

        gps = self._get_position(result)
        far_node, _ = self._command_planner.run_step(gps)
        target = R.T.dot(far_node - gps)
        target *= 5.5
        target += [128, 256]
        target = np.clip(target, 0, 256)

        if(self.weather[0]<="20.0" and self.weather[1]<="20.0" and self.weather[2]<="20.0"):
            result['cloud'][0] = result['cloud'][0]
        elif((self.weather[0]>"20.0" and self.weather[0]<"50.0") and (self.weather[1]>"20.0" and self.weather[1]<"50.0") and (self.weather[2]>"20.0" and self.weather[2]<"50.0")):
            noise = np.random.normal(0, 0.5, result['cloud'][0].shape)
            result['cloud'][0] += noise
        elif((self.weather[0]>"50.0" and self.weather[0]<"70.0") and (self.weather[1]>"50.0" and self.weather[1]<"70.0") and (self.weather[2]>"50.0" and self.weather[2]<"70.0")):
            noise = np.random.normal(0, 1.5, result['cloud'][0].shape)
            result['cloud'][0] += noise
        elif((self.weather[0]>"70.0" and self.weather[0]<"100.0") and (self.weather[1]>"70.0" and self.weather[1]<"100.0") and (self.weather[2]>"70.0" and self.weather[2]<"100.0")):
            noise = np.random.normal(0, 2, result['cloud'][0].shape)
            result['cloud'][0] += noise
        else:
            noise = np.random.normal(0, 0.5, result['cloud'][0].shape)
            result['cloud'][0] += noise

        result['target'] = target
        object_depth = []
        object_vel = []
        for i in range(len(result['cloud'])):
            object_depth.append(result['cloud'][i][0])
            object_vel.append(result['cloud'][i][3])

        index = np.argmin(object_depth)
        result['object_velocity'] = object_vel[index]
        result['object_distance'] = abs(min(object_depth))

        BlurDetectorThread.join()
        OccusionDetectorThread.join()
        AssuranceMonitorThread.join()

        self.risk_computation(self.weather,self.blur_queue,self.am_queue,self.occlusion_queue,self.fault_scenario,self.fault_type,self.fault_time,self.fault_step)

        del predicted_reps
        gc.collect()

        return result, far_node

    def NN_controller(self, tick_data):
        val = 0
        img = torchvision.transforms.functional.to_tensor(tick_data['image'])
        img = img[None].cuda()

        target = torch.from_numpy(tick_data['target'])
        target = target[None].cuda()

        with torch.no_grad():
            points, (target_cam, _) = self.net.forward(img, target)
        points_cam = points.clone().cpu()
        points_cam[..., 0] = (points_cam[..., 0] + 1) / 2 * img.shape[-1]
        points_cam[..., 1] = (points_cam[..., 1] + 1) / 2 * img.shape[-2]
        points_cam = points_cam.squeeze()
        points_world = self.converter.cam_to_world(points_cam).numpy()

        aim = (points_world[1] + points_world[0]) / 2.0
        angle = np.degrees(np.pi / 2 - np.arctan2(aim[1], aim[0])) / 90
        steer = self._turn_controller.step(angle)
        steer = np.clip(steer, -1.0, 1.0)

        desired_speed = np.linalg.norm(points_world[0] - points_world[1]) * 2.0
        # desired_speed *= (1 - abs(angle)) ** 2

        speed = tick_data['speed']
        stopping_distance = ((speed*speed)/13.72)

        if(tick_data['object_distance'] < 4.2):
            brake = True
            val=1
        else:
            brake = desired_speed < 0.2 or (speed / desired_speed) > 1.1

        delta = np.clip(desired_speed - speed, 0.0, 0.25)
        throttle = self._speed_controller.step(delta)
        throttle = np.clip(throttle, 0.0, 0.75)
        #throttle = 0.75
        throttle = throttle if not brake else 0.0
        return steer, brake, throttle, speed, stopping_distance, val, target_cam, points, desired_speed

    def auto(self, target, far_target, tick_data):
        pos = self._get_position(tick_data)
        theta = tick_data['compass']
        speed = tick_data['speed']

        # Steering.
        angle_unnorm = self._get_angle_to(pos, theta, target)
        angle = angle_unnorm / 90

        steer = self._turn_controller.step(angle)
        steer = np.clip(steer, -1.0, 1.0)
        steer = round(steer, 3)

        # Acceleration.
        angle_far_unnorm = self._get_angle_to(pos, theta, far_target)
        should_slow = abs(angle_far_unnorm) > 45.0 or abs(angle_unnorm) > 5.0
        target_speed = 4 if should_slow else 7.0

        brake = self._should_brake()
        target_speed = target_speed if not brake else 0.0

        delta = np.clip(target_speed - speed, 0.0, 0.25)
        throttle = self._speed_controller.step(delta)
        throttle = np.clip(throttle, 0.0, 0.75)

        if brake:
            steer *= 0.5
            throttle = 0.0

        return steer, throttle, brake, target_speed
        #far_node, far_command = self._command_planner.run_step(gps)

    def _get_angle_to(self, pos, theta, target):
        R = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta),  np.cos(theta)],
            ])

        aim = R.T.dot(target - pos)
        angle = -np.degrees(np.arctan2(-aim[1], aim[0]))
        angle = 0.0 if np.isnan(angle) else angle
        return angle

    def _should_brake(self):
        actors = self._world.get_actors()

        vehicle = self._is_vehicle_hazard(actors.filter('*vehicle*'))
        light = self._is_light_red(actors.filter('*traffic_light*'))
        walker = self._is_walker_hazard(actors.filter('*walker*'))

        return any(x is not None for x in [vehicle, light, walker])

    def _is_light_red(self, lights_list):
        if self._vehicle.get_traffic_light_state() != carla.libcarla.TrafficLightState.Green:
            affecting = self._vehicle.get_traffic_light()

            for light in self._traffic_lights:
                if light.id == affecting.id:
                    return affecting

        return None

    def _is_walker_hazard(self, walkers_list):
        z = self._vehicle.get_location().z
        p1 = _numpy(self._vehicle.get_location())
        v1 = 10.0 * _orientation(self._vehicle.get_transform().rotation.yaw)

        self._draw_line(p1, v1, z+2.5, (0, 0, 255))

        for walker in walkers_list:
            v2_hat = _orientation(walker.get_transform().rotation.yaw)
            s2 = np.linalg.norm(_numpy(walker.get_velocity()))

            if s2 < 0.05:
                v2_hat *= s2

            p2 = -3.0 * v2_hat + _numpy(walker.get_location())
            v2 = 8.0 * v2_hat

            self._draw_line(p2, v2, z+2.5)

            collides, collision_point = get_collision(p1, v1, p2, v2)

            if collides:
                return walker

        return None


    def _is_vehicle_hazard(self, vehicle_list):
        z = self._vehicle.get_location().z

        o1 = _orientation(self._vehicle.get_transform().rotation.yaw)
        p1 = _numpy(self._vehicle.get_location())
        s1 = max(7.5, 2.0 * np.linalg.norm(_numpy(self._vehicle.get_velocity())))
        v1_hat = o1
        v1 = s1 * v1_hat

        self._draw_line(p1, v1, z+2.5, (255, 0, 0))

        for target_vehicle in vehicle_list:
            if target_vehicle.id == self._vehicle.id:
                continue

            o2 = _orientation(target_vehicle.get_transform().rotation.yaw)
            p2 = _numpy(target_vehicle.get_location())
            s2 = max(5.0, 2.0 * np.linalg.norm(_numpy(target_vehicle.get_velocity())))
            v2_hat = o2
            v2 = s2 * v2_hat

            p2_p1 = p2 - p1
            distance = np.linalg.norm(p2_p1)
            p2_p1_hat = p2_p1 / (distance + 1e-4)

            self._draw_line(p2, v2, z+2.5, (255, 0, 0))

            angle_to_car = np.degrees(np.arccos(v1_hat.dot(p2_p1_hat)))
            angle_between_heading = np.degrees(np.arccos(o1.dot(o2)))

            if angle_between_heading > 60.0 and not (angle_to_car < 15 and distance < s1):
                continue
            elif angle_to_car > 30.0:
                continue
            elif distance > s1:
                continue

            return target_vehicle

        return None

    def _draw_line(self, p, v, z, color=(255, 0, 0)):
        if not DEBUG:
            return

        p1 = _location(p[0], p[1], z)
        p2 = _location(p[0]+v[0], p[1]+v[1], z)
        color = carla.Color(*color)

        #self._world.debug.draw_line(p1, p2, 0.25, color, 0.01)

    def set_global_plan(self, global_plan_gps, global_plan_world_coord):
        super().set_global_plan(global_plan_gps, global_plan_world_coord)
        self._plan_HACK = global_plan_world_coord
        self._plan_gps_HACK = global_plan_gps

    def save(self, far_node, near_command, steer, throttle, brake, target_speed, tick_data):
        frame = self.step // 5

        pos = self._get_position(tick_data)
        theta = tick_data['compass']
        speed = tick_data['speed']

        data = {
                'x': pos[0],
                'y': pos[1],
                'theta': theta,
                'speed': speed,
                'target_speed': target_speed,
                'x_command': far_node[0],
                'y_command': far_node[1],
                'command': near_command.value,
                'steer': steer,
                'throttle': throttle,
                'brake': brake,
                }
        (self.save_path / 'measurements' / ('%04d.json' % frame)).write_text(str(data))

        rgb_detector = cv2.resize(tick_data['rgb'],(224,224))

        Image.fromarray(tick_data['rgb']).save(self.save_path / 'rgb' / ('%04d.png' % frame))
        Image.fromarray(tick_data['rgb_left']).save(self.save_path / 'rgb_left' / ('%04d.png' % frame))
        Image.fromarray(tick_data['rgb_right']).save(self.save_path / 'rgb_right' / ('%04d.png' % frame))
        Image.fromarray(tick_data['topdown']).save(self.save_path / 'topdown' / ('%04d.png' % frame))
        Image.fromarray(rgb_detector).save(self.save_path / 'rgb_detector' / ('%04d.png' % frame))

    @torch.no_grad()
    def run_step(self, input_data, timestamp):
        if not self.initialized:
            self._init()
        val=0
        tick_data, far_node = self.tick(input_data)
        gps = self._get_position(tick_data)
        near_node, near_command = self._waypoint_planner.run_step(gps)
        #print(tick_data['topdown'],tick_data['rgb_left'], tick_data['rgb'], tick_data['rgb_right'],tick_data['gps'])
        steer, brake, throttle, speed, stopping_distance, val, target_cam, points, desired_speed = self.NN_controller(tick_data)
        # auto_steer, auto_throttle, auto_brake, auto_target_speed = self.auto(near_node, far_node, tick_data)
        # #print("autopilot:",auto_steer, auto_throttle, auto_brake)
        # #print("NNcontroller:",steer, throttle, brake)
        # if self.step % 5 == 0:
        #     self.save(far_node, near_command, auto_steer, auto_throttle, auto_brake, auto_target_speed, tick_data)
        #steer, brake, throttle = self.auto(tick_data)
        #steering, brake, throttle = auto_pilot_controller(tick_data)

        control = carla.VehicleControl()
        control.steer = steer
        control.throttle = throttle
        control.brake = float(brake)

        fields = ['step',
                'emergency_braking',
                'stopping_distance',
                'Radar_distance',
                'speed',
                'steer'
                ]

        dict = [{'step':self.step,'emergency_braking':val,'stopping_distance':abs(stopping_distance), 'Radar_distance':tick_data['object_distance'],'speed':speed,'steer':steer}]

        if(self.step == 0):
            self.braking_file = self.data_folder + "/emergency_braking.csv"

        file_exists = os.path.isfile(self.braking_file)
        with open(self.braking_file, 'a') as csvfile:
            # creating a csv dict writer object
            writer = csv.DictWriter(csvfile, fieldnames = fields)
            if not file_exists:
                writer.writeheader()
            writer.writerows(dict)

        if DEBUG:
            debug_display(
                    tick_data, target_cam.squeeze(), points.cpu().squeeze(),
                    steer, throttle, brake, desired_speed,
                    self.step)

        return control
