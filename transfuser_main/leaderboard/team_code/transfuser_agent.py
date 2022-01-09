import os
import json
import datetime
import pathlib
import time
import cv2
import carla
from collections import deque
import random
import torch
import carla
import gc
import os
import threading
from threading import Thread
import numpy as np
from PIL import Image
from queue import Queue
from keras.backend.tensorflow_backend import set_session
from leaderboard.autoagents import autonomous_agent
from transfuser.model import TransFuser
from transfuser.config import GlobalConfig
from transfuser.data import scale_and_crop_image, lidar_to_histogram_features, transform_2d_points
from team_code.planner import RoutePlanner
from keras import backend as K
import sys
sys.path.append('/home/baiting1/Desktop/risk.git/carla-challange/leaderboard')
from detectors.anomaly_detector import occlusion_detector, blur_detector, assurance_monitor
from team_code.risk_calculation.fault_modes import FaultModes
from team_code.risk_calculation.bowtie_diagram import BowTie
import tensorflow as tf
import math
from matplotlib import cm
from keras.models import Model, model_from_json, load_model
import csv
import scipy.integrate as integrate
#SAVE_PATH = os.environ.get('SAVE_PATH', None)
DEBUG = int(os.environ.get('HAS_DISPLAY', 0))

def get_entry_point():
	return 'TransFuserAgent'

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

class TransFuserAgent(autonomous_agent.AutonomousAgent):
	def setup(self, path_to_conf_file,data_folder,route_folder,k,model_path,fault_type,image_folder,sensor_faults_file):
		self.lidar_processed = list()
		self.track = autonomous_agent.Track.SENSORS
		self.config_path = path_to_conf_file
		self.step = -1
		self.wall_start = time.time()
		self.initialized = False

		self.input_buffer = {'rgb': deque(), 'rgb_left': deque(), 'rgb_right': deque(), 
							'rgb_rear': deque(), 'lidar': deque(), 'gps': deque(), 'thetas': deque()}

		self.config = GlobalConfig()
		self.net = TransFuser(self.config, 'cuda')
		self.net.load_state_dict(torch.load(os.path.join(path_to_conf_file, 'best_model.pth')))
		self.net.cuda()
		self.net.eval()

		#self.converter = Converter()
		self.data_folder = data_folder
		self.route_folder = route_folder
		self.image_folder = image_folder
		self.scene_number = k
		# self.failure_mode = i
		self.weather_file = self.route_folder + "/weather_data.csv"
		self.model_path = model_path  # "/home/scope/Carla/autopilot_Carla_ad/leaderboard/team_code/detector_code/trial1/old/center-B-1.2/"
		# self.device = cuda.get_current_device()
		# self.device.reset()
		self.model_vae = None
		# torch.cuda.empty_cache()
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
		# self.rgb_detector = []
		self.detector_file = None
		self.detector_file = None
		K.clear_session()
		config = tf.ConfigProto()
		# config.gpu_options.allow_growth = True
		sess = tf.Session(config=tf.ConfigProto(device_count={'GPU': 0}))  # tf.Session(config=config)
		set_session(sess)

		self.save_path = None

		path = "/home/baiting1/Desktop/risk.git/images"
		if path:
			now = datetime.datetime.now()
			string = pathlib.Path(os.environ['ROUTES']).stem + '_'
			string += '_'.join(map(lambda x: '%02d' % x, (now.month, now.day, now.hour, now.minute, now.second)))

			print (string)

			self.save_path = pathlib.Path(path) / string
			self.save_path.mkdir(exist_ok=False)

			#(self.save_path / 'rgb').mkdir(parents=True, exist_ok=False)
			#(self.save_path / 'lidar_0').mkdir(parents=True, exist_ok=False)
			#(self.save_path / 'lidar_1').mkdir(parents=True, exist_ok=False)
			#(self.save_path / 'meta').mkdir(parents=True, exist_ok=False)
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
		self.fields_risk = ['step',
					   'monitor_result',
					   'risk',
					   'rgb_blur',
					   # 'rgb_blur_percent',
					   'rgb_left_blur',
					   # 'rgb_left_blur_percent',
					   'rgb_right_blur',
					   # 'rgb_right_blur_percent',
					   'rgb_occluded',
					   # 'rgb_occluded_percent',
					   'rgb_left_occluded',
					   # 'rgb_left_occluded_percent',
					   'rgb_right_occluded',
					   # 'rgb_right_occluded_percent'
					   ]
		self.weather, self.run_number = process_weather_data(self.weather_file, self.scene_number)
		with open(self.model_path + 'calibration.csv', 'r') as file:
			reader = csv.reader(file)
			for row in reader:
				self.calib_set.append(row)

		self.track = autonomous_agent.Track.SENSORS
		self.config_path = path_to_conf_file
		self.data_folder = data_folder
		self.image_folder = image_folder
		# self.failure_mode = i
		self.value = 0
		self.filename = self.data_folder + "/fault_data.csv"  # "/home/scope/Carla/ICCPS_CARLA_challenge/leaderboard/data/my_data/Simulation6/fault_data.csv"
		self.step = -1
		self.wall_start = time.time()
		self.initialized = False
		self.fault_type_list = None
		self.fault_scenario = 1  # random.randint(0,1)
		self.fault_step = random.randint(100, 150)
		self.fault_time = random.randint(150, 200)
		self.fields = ['fault_scenario',
					   'fault_step',
					   'fault_time',
					   'fault_type',
					   'fault_list',
					   'brightness_value'
					   ]

		if (self.fault_scenario == 0):
			self.fault_type = 0
			self.fault_type_list = -1
		elif (self.fault_scenario == 1):
			self.fault_type = fault_type  # 15 #random.randint(0,13)
			self.fault_type_list = get_fault_list(self.fault_type)

		self.dict = [{'fault_scenario': self.fault_scenario, 'fault_step': self.fault_step, 'fault_time': self.fault_time,
					  'fault_type': self.fault_type, 'fault_list': self.fault_type_list, 'brightness_value': self.value}]

		file_exists = os.path.isfile(self.filename)

		# writing to csv file
		with open(self.filename, 'a') as csvfile:
			# creating a csv dict writer object
			writer = csv.DictWriter(csvfile, fieldnames=self.fields)

			if not file_exists:
				# writing headers (field names)
				writer.writeheader()
			# writing data rows
			writer.writerows(self.dict)


	def _init(self):
		self._route_planner = RoutePlanner(4.0, 50.0)
		self._route_planner.set_route(self._global_plan, True)

		self.initialized = True

	def _get_position(self, tick_data):
		gps = tick_data['gps']
		gps = (gps - self._route_planner.mean) * self._route_planner.scale

		return gps

	def sensors(self):
		return [
				{
					'type': 'sensor.camera.rgb',
					'x': 1.3, 'y': 0.0, 'z':2.3,
					'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
					'width': 400, 'height': 300, 'fov': 100,
					'id': 'rgb'
					},
				{
					'type': 'sensor.camera.rgb',
					'x': 1.3, 'y': 0.0, 'z':2.3,
					'roll': 0.0, 'pitch': 0.0, 'yaw': -60.0,
					'width': 400, 'height': 300, 'fov': 100,
					'id': 'rgb_left'
					},
				{
					'type': 'sensor.camera.rgb',
					'x': 1.3, 'y': 0.0, 'z':2.3,
					'roll': 0.0, 'pitch': 0.0, 'yaw': 60.0,
					'width': 400, 'height': 300, 'fov': 100,
					'id': 'rgb_right'
					},
				{
					'type': 'sensor.camera.rgb',
					'x': -1.3, 'y': 0.0, 'z':2.3,
					'roll': 0.0, 'pitch': 0.0, 'yaw': -180.0,
					'width': 400, 'height': 300, 'fov': 100,
					'id': 'rgb_rear'
					},
                {   
                    'type': 'sensor.lidar.ray_cast',
                    'x': 1.3, 'y': 0.0, 'z': 2.5,
                    'roll': 0.0, 'pitch': 0.0, 'yaw': -90.0,
                    'id': 'lidar'
                    },
				{
					'type': 'sensor.other.imu',
					'x': 0.0, 'y': 0.0, 'z': 0.0,
					'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
					'sensor_tick': 0.05,
					'id': 'imu'
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
					},
				{
					'type': 'sensor.other.radar',
					'x': 2.8, 'y': 0.0, 'z': 1.0,
					'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0, 'fov': 25,
					'sensor_tick': 0.05,
					'id': 'radar'
				}
				]

	def fault_list(self, rgb, rgb_left, rgb_right, points, gps):
		if (self.fault_type == 0):
			# print("Blur Image")
			rgb = rgb
		elif (self.fault_type == 1):
			# print("Blur Image")
			rgb = cv2.blur(rgb, (10, 10))
		elif (self.fault_type == 2):
			# print("Blur Image")
			rgb_left = cv2.blur(rgb_left, (10, 10))
		elif (self.fault_type == 3):
			# print("Blur Image")
			rgb_right = cv2.blur(rgb_right, (10, 10))
		elif (self.fault_type == 4):
			# print("Blur Image")
			rgb_left = cv2.blur(rgb_left, (10, 10))
			rgb_right = cv2.blur(rgb_right, (10, 10))
		elif (self.fault_type == 5):
			# print("Blur Image")
			rgb = cv2.blur(rgb, (10, 10))
			rgb_right = cv2.blur(rgb_right, (10, 10))
		elif (self.fault_type == 6):
			# print("Blur Image")
			rgb = cv2.blur(rgb, (10, 10))
			rgb_left = cv2.blur(rgb_left, (10, 10))
		elif (self.fault_type == 7):
			# print("Blur Image")
			rgb = cv2.blur(rgb, (10, 10))
			rgb_right = cv2.blur(rgb_right, (10, 10))
			rgb_left = cv2.blur(rgb_left, (10, 10))
		if (self.fault_type == 8):
			# print("Center Camera Image Occluded")
			h, w, _ = rgb.shape
			rgb = cv2.rectangle(rgb, (0, 0), (w // 4, h), (0, 0, 0), cv2.FILLED)
		elif (self.fault_type == 9):
			# print("Left Camera Image Occluded")
			h, w, _ = rgb_left.shape
			rgb_left = cv2.rectangle(rgb_left, (0, 0), (w // 4, h), (0, 0, 0), cv2.FILLED)
		elif (self.fault_type == 10):
			# print("Right Camera Image Occluded")
			h, w, _ = rgb_right.shape
			rgb_right = cv2.rectangle(rgb_right, (0, 0), (w // 2, h), (0, 0, 0), cv2.FILLED)
		elif (self.fault_type == 11):
			# print("Right & left Camera Images Occluded")
			h, w, _ = rgb_right.shape
			rgb_right = cv2.rectangle(rgb_right, (0, 0), (w // 4, h), (0, 0, 0), cv2.FILLED)
			rgb_left = cv2.rectangle(rgb_left, (0, 0), (w // 4, h), (0, 0, 0), cv2.FILLED)
		elif (self.fault_type == 12):
			# print("Right & center Camera Images Occluded")
			h, w, _ = rgb_right.shape
			rgb_right = cv2.rectangle(rgb_right, (0, 0), (w // 4, h), (0, 0, 0), cv2.FILLED)
			rgb = cv2.rectangle(rgb, (0, 0), (w // 4, h), (0, 0, 0), cv2.FILLED)
		elif (self.fault_type == 13):
			# print("center & left Camera Images Occluded")
			h, w, _ = rgb.shape
			rgb = cv2.rectangle(rgb, (0, 0), (w // 4, h), (0, 0, 0), cv2.FILLED)
			rgb_left = cv2.rectangle(rgb_left, (0, 0), (w // 4, h), (0, 0, 0), cv2.FILLED)
		elif (self.fault_type == 14):
			# print("All the Camera Images are Occluded")
			h, w, _ = rgb_right.shape
			rgb_right = cv2.rectangle(rgb_right, (0, 0), (w // 4, h), (0, 0, 0), cv2.FILLED)
			rgb_left = cv2.rectangle(rgb_left, (0, 0), (w // 4, h), (0, 0, 0), cv2.FILLED)
			rgb = cv2.rectangle(rgb, (0, 0), (w // 4, h), (0, 0, 0), cv2.FILLED)
		elif (self.fault_type == 15):
			self.value = 40
		# gps = gps
		# noise = np.random.normal(0, .001, gps.shape)
		# gps += noise

		return rgb, rgb_left, rgb_right, points, gps, self.value

	def add_brightness(self, rgb, rgb_left, rgb_right):
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

		return rgb, rgb_left, rgb_right

	def risk_computation(self, weather, blur_queue, am_queue, occlusion_queue, fault_scenario, fault_type, fault_time,
						 fault_step):
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
		r_t1_top = bowtie.rate_t1(state) * (1 - bowtie.prob_b1(state, fault_modes))
		r_t2_top = bowtie.rate_t2(state) * (1 - bowtie.prob_b2(state, fault_modes))
		r_top = r_t1_top + r_t2_top
		r_c1 = r_top * (1 - bowtie.prob_b3(state, fault_modes))

		print("Dynamic Risk:%f" % r_c1)

		dict = [{'step': self.step, 'monitor_result': mval, 'risk': round(r_c1, 2), 'rgb_blur': blur[0],
				 'rgb_left_blur': blur[1], 'rgb_right_blur': blur[2],
				 'rgb_occluded': occlusion[0], 'rgb_left_occluded': occlusion[1], 'rgb_right_occluded': occlusion[2]}]

		if (self.step == 0):
			self.detector_file = self.data_folder + "/run%d.csv" % (self.scene_number)

		file_exists = os.path.isfile(self.detector_file)
		with open(self.detector_file, 'a') as csvfile:
			# creating a csv dict writer object
			writer = csv.DictWriter(csvfile, fieldnames=self.fields_risk)
			if not file_exists:
				writer.writeheader()
			writer.writerows(dict)

	def blur_detection(self, result):
		self.blur = []
		fm1, rgb_blur = blur_detector(result['rgb'], threshold=20)
		fm2, rgb_left_blur = blur_detector(result['rgb_left'], threshold=20)
		fm3, rgb_right_blur = blur_detector(result['rgb_right'], threshold=20)
		self.blur.append(rgb_blur)
		# self.blur.append(fm1)
		self.blur.append(rgb_left_blur)
		# self.blur.append(fm2)
		self.blur.append(rgb_right_blur)
		# self.blur.append(fm3)
		self.blur_queue.put(self.blur)

	def occlusion_detection(self, result):
		self.occlusion = []
		percent1, rgb_occluded = occlusion_detector(result['rgb'], threshold=25)
		percent2, rgb_left_occluded = occlusion_detector(result['rgb_left'], threshold=25)
		percent3, rgb_right_occluded = occlusion_detector(result['rgb_right'], threshold=25)
		self.occlusion.append(rgb_occluded)
		# self.occlusion.append(percent1)
		self.occlusion.append(rgb_left_occluded)
		# self.occlusion.append(percent2)
		self.occlusion.append(rgb_right_occluded)
		# self.occlusion.append(percent3)
		self.occlusion_queue.put(self.occlusion)

	def assurance_monitor(self, dist):
		if (self.step == 0):
			p_anomaly = []
			prev_value = []
		else:
			p_anomaly = self.pval_queue.get()
			prev_value = self.sval_queue.get()
		anomaly = 0
		m = 0
		delta = 20
		# threshold = 20
		sliding_window = 5
		threshold = 20.0
		for i in range(len(self.calib_set)):
			if (float(dist) <= float(self.calib_set[i][0])):
				anomaly += 1

		p_value = anomaly / len(self.calib_set)
		if (p_value < 0.005):
			p_anomaly.append(0.005)
		else:
			p_anomaly.append(p_value)
		if (len(p_anomaly)) >= sliding_window:
			p_anomaly = p_anomaly[-1 * sliding_window:]
		m = integrate.quad(self.integrand, 0.0, 1.0, args=(p_anomaly))
		m_val = round(math.log(m[0]), 2)
		# print("m_val:%f"%m_val)
		if (self.step == 0):
			S = 0
			S_prev = 0
		else:
			S = max(0, prev_value[0] + prev_value[1] - delta)
		prev_value = []
		S_prev = S
		m_prev = m[0]
		prev_value.append(S_prev)
		prev_value.append(m_prev)
		self.pval_queue.put(p_anomaly)
		self.sval_queue.put(prev_value)
		self.mval_queue.put(m_val)

	def integrand(self, k, p_anomaly):
		result = 1.0
		for i in range(len(p_anomaly)):
			result *= k * (p_anomaly[i] ** (k - 1.0))
		return result

	def tick(self, input_data):
		self.step += 1

		rgb = cv2.cvtColor(input_data['rgb'][1][:, :, :3], cv2.COLOR_BGR2RGB)
		rgb_left = cv2.cvtColor(input_data['rgb_left'][1][:, :, :3], cv2.COLOR_BGR2RGB)
		rgb_right = cv2.cvtColor(input_data['rgb_right'][1][:, :, :3], cv2.COLOR_BGR2RGB)
		rgb_rear = cv2.cvtColor(input_data['rgb_rear'][1][:, :, :3], cv2.COLOR_BGR2RGB)
		gps = input_data['gps'][1][:2]
		speed = input_data['speed'][1]['speed']
		compass = input_data['imu'][1][-1]
		if (math.isnan(compass) == True): #It can happen that the compass sends nan for a few frames
			compass = 0.0
		lidar = input_data['lidar'][1][:, :3]

		radar_data = (input_data['radar'][1])
		# print(radar_data)
		points = np.reshape(radar_data, (len(radar_data), 4))
		# print(points[0][0])

		if (self.fault_scenario == 1):
			if (self.step > self.fault_step and self.step < self.fault_step + self.fault_time):
				rgb, rgb_left, rgb_right, points, gps, self.value = self.fault_list(rgb, rgb_left, rgb_right, points,
																					gps)
				rgb, rgb_left, rgb_right = self.add_brightness(rgb, rgb_left, rgb_right)


		result = {
				'rgb': rgb,
				'rgb_left': rgb_left,
				'rgb_right': rgb_right,
				'rgb_rear': rgb_rear,
				'lidar': lidar,
				'gps': gps,
				'speed': speed,
				'compass': compass,
				'cloud': points,
				'fault_scenario': self.fault_scenario,
				'fault_step': self.fault_step,
				'fault_duration': self.fault_time,
				'fault_type': self.fault_type
				}
		result['rgb_detector'] = cv2.resize(result['rgb'], (224, 224))
		result['rgb_detector_left'] = cv2.resize(result['rgb_left'], (224, 224))
		result['rgb_detector_right'] = cv2.resize(result['rgb_right'], (224, 224))
		result['rgb_detector'] = cv2.cvtColor(result['rgb_detector_right'], cv2.COLOR_RGB2BGR)
		cv2.imshow('Agent', result['rgb_detector'])
		cv2.waitKey(1)
		detection_image = result['rgb_detector'] / 255.
		detection_image = np.reshape(detection_image,
									 [-1, detection_image.shape[0], detection_image.shape[1], detection_image.shape[2]])
		predicted_reps = self.model_vae.predict_on_batch(detection_image)
		dist = np.square(np.subtract(np.array(predicted_reps), detection_image)).mean()  # - 0.001
		BlurDetectorThread = Thread(target=self.blur_detection, args=(result,))
		BlurDetectorThread.daemon = True
		OccusionDetectorThread = Thread(target=self.occlusion_detection, args=(result,))
		OccusionDetectorThread.daemon = True
		AssuranceMonitorThread = Thread(target=self.assurance_monitor,
										args=(dist,))  # image,model,calibration_set,pval_queue,sval_queue
		AssuranceMonitorThread.daemon = True

		# Start threads for parallel computation
		AssuranceMonitorThread.start()
		BlurDetectorThread.start()
		OccusionDetectorThread.start()

		pos = self._get_position(result)
		result['gps'] = pos
		next_wp, next_cmd = self._route_planner.run_step(pos)
		result['next_command'] = next_cmd.value

		theta = compass + np.pi/2
		R = np.array([
			[np.cos(theta), -np.sin(theta)],
			[np.sin(theta), np.cos(theta)]
			])

		local_command_point = np.array([next_wp[0]-pos[0], next_wp[1]-pos[1]])
		local_command_point = R.T.dot(local_command_point)
		result['target_point'] = tuple(local_command_point)

		if (self.weather[0] <= "20.0" and self.weather[1] <= "20.0" and self.weather[2] <= "20.0"):
			result['cloud'][0] = result['cloud'][0]
		elif ((self.weather[0] > "20.0" and self.weather[0] < "50.0") and (
				self.weather[1] > "20.0" and self.weather[1] < "50.0") and (
					  self.weather[2] > "20.0" and self.weather[2] < "50.0")):
			noise = np.random.normal(0, 0.5, result['cloud'][0].shape)
			result['cloud'][0] += noise
		elif ((self.weather[0] > "50.0" and self.weather[0] < "70.0") and (
				self.weather[1] > "50.0" and self.weather[1] < "70.0") and (
					  self.weather[2] > "50.0" and self.weather[2] < "70.0")):
			noise = np.random.normal(0, 1.5, result['cloud'][0].shape)
			result['cloud'][0] += noise
		elif ((self.weather[0] > "70.0" and self.weather[0] < "100.0") and (
				self.weather[1] > "70.0" and self.weather[1] < "100.0") and (
					  self.weather[2] > "70.0" and self.weather[2] < "100.0")):
			noise = np.random.normal(0, 2, result['cloud'][0].shape)
			result['cloud'][0] += noise
		else:
			noise = np.random.normal(0, 0.5, result['cloud'][0].shape)
			result['cloud'][0] += noise
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

		self.risk_computation(self.weather, self.blur_queue, self.am_queue, self.occlusion_queue, self.fault_scenario,
							  self.fault_type, self.fault_time, self.fault_step)

		del predicted_reps
		gc.collect()

		return result

	@torch.no_grad()
	def run_step(self, input_data, timestamp):
		if not self.initialized:
			self._init()

		tick_data = self.tick(input_data)

		if self.step < self.config.seq_len:
			rgb = torch.from_numpy(scale_and_crop_image(Image.fromarray(tick_data['rgb']), crop=self.config.input_resolution)).unsqueeze(0)
			self.input_buffer['rgb'].append(rgb.to('cuda', dtype=torch.float32))
			
			if not self.config.ignore_sides:
				rgb_left = torch.from_numpy(scale_and_crop_image(Image.fromarray(tick_data['rgb_left']), crop=self.config.input_resolution)).unsqueeze(0)
				self.input_buffer['rgb_left'].append(rgb_left.to('cuda', dtype=torch.float32))
				
				rgb_right = torch.from_numpy(scale_and_crop_image(Image.fromarray(tick_data['rgb_right']), crop=self.config.input_resolution)).unsqueeze(0)
				self.input_buffer['rgb_right'].append(rgb_right.to('cuda', dtype=torch.float32))

			if not self.config.ignore_rear:
				rgb_rear = torch.from_numpy(scale_and_crop_image(Image.fromarray(tick_data['rgb_rear']), crop=self.config.input_resolution)).unsqueeze(0)
				self.input_buffer['rgb_rear'].append(rgb_rear.to('cuda', dtype=torch.float32))

			self.input_buffer['lidar'].append(tick_data['lidar'])
			self.input_buffer['gps'].append(tick_data['gps'])
			self.input_buffer['thetas'].append(tick_data['compass'])

			control = carla.VehicleControl()
			control.steer = 0.0
			control.throttle = 0.0
			control.brake = 0.0
			
			return control

		gt_velocity = torch.FloatTensor([tick_data['speed']]).to('cuda', dtype=torch.float32)
		command = torch.FloatTensor([tick_data['next_command']]).to('cuda', dtype=torch.float32)

		tick_data['target_point'] = [torch.FloatTensor([tick_data['target_point'][0]]),
											torch.FloatTensor([tick_data['target_point'][1]])]
		target_point = torch.stack(tick_data['target_point'], dim=1).to('cuda', dtype=torch.float32)

		encoding = []
		rgb = torch.from_numpy(scale_and_crop_image(Image.fromarray(tick_data['rgb']), crop=self.config.input_resolution)).unsqueeze(0)
		self.input_buffer['rgb'].popleft()
		self.input_buffer['rgb'].append(rgb.to('cuda', dtype=torch.float32))
		
		if not self.config.ignore_sides:
			rgb_left = torch.from_numpy(scale_and_crop_image(Image.fromarray(tick_data['rgb_left']), crop=self.config.input_resolution)).unsqueeze(0)
			self.input_buffer['rgb_left'].popleft()
			self.input_buffer['rgb_left'].append(rgb_left.to('cuda', dtype=torch.float32))
			
			rgb_right = torch.from_numpy(scale_and_crop_image(Image.fromarray(tick_data['rgb_right']), crop=self.config.input_resolution)).unsqueeze(0)
			self.input_buffer['rgb_right'].popleft()
			self.input_buffer['rgb_right'].append(rgb_right.to('cuda', dtype=torch.float32))

		if not self.config.ignore_rear:
			rgb_rear = torch.from_numpy(scale_and_crop_image(Image.fromarray(tick_data['rgb_rear']), crop=self.config.input_resolution)).unsqueeze(0)
			self.input_buffer['rgb_rear'].popleft()
			self.input_buffer['rgb_rear'].append(rgb_rear.to('cuda', dtype=torch.float32))

		self.input_buffer['lidar'].popleft()
		self.input_buffer['lidar'].append(tick_data['lidar'])
		self.input_buffer['gps'].popleft()
		self.input_buffer['gps'].append(tick_data['gps'])
		self.input_buffer['thetas'].popleft()
		self.input_buffer['thetas'].append(tick_data['compass'])

		# transform the lidar point clouds to local coordinate frame
		ego_theta = self.input_buffer['thetas'][-1]
		ego_x, ego_y = self.input_buffer['gps'][-1]

		#Only predict every second step because we only get a LiDAR every second frame.
		if(self.step  % 2 == 0 or self.step <= 4):
			for i, lidar_point_cloud in enumerate(self.input_buffer['lidar']):
				curr_theta = self.input_buffer['thetas'][i]
				curr_x, curr_y = self.input_buffer['gps'][i]
				lidar_point_cloud[:,1] *= -1 # inverts x, y
				lidar_transformed = transform_2d_points(lidar_point_cloud,
						np.pi/2-curr_theta, -curr_x, -curr_y, np.pi/2-ego_theta, -ego_x, -ego_y)
				lidar_transformed = torch.from_numpy(lidar_to_histogram_features(lidar_transformed, crop=self.config.input_resolution)).unsqueeze(0)
				self.lidar_processed = list()
				self.lidar_processed.append(lidar_transformed.to('cuda', dtype=torch.float32))


			self.pred_wp = self.net(self.input_buffer['rgb'] + self.input_buffer['rgb_left'] + \
							   self.input_buffer['rgb_right']+self.input_buffer['rgb_rear'], \
							   self.lidar_processed, target_point, gt_velocity)

		steer, throttle, brake, metadata = self.net.control_pid(self.pred_wp, gt_velocity)
		self.pid_metadata = metadata

		if brake < 0.05: brake = 0.0
		if throttle > brake: brake = 0.0

		control = carla.VehicleControl()
		control.steer = float(steer)
		control.throttle = float(throttle)
		control.brake = float(brake)

		#if self.step % 10 == 0:
		#	self.save(tick_data)


		return control

	def save(self, tick_data):
		frame = self.step // 10
		#print(frame)
		Image.fromarray(tick_data['rgb']).save(self.save_path / 'rgb' / ('%04d.png' % frame))

		Image.fromarray(cm.gist_earth(self.lidar_processed[0].cpu().numpy()[0, 0], bytes=True)).save(self.save_path / 'lidar_0' / ('%04d.png' % frame))
		Image.fromarray(cm.gist_earth(self.lidar_processed[0].cpu().numpy()[0, 1], bytes=True)).save(self.save_path / 'lidar_1' / ('%04d.png' % frame))


		outfile = open(self.save_path / 'meta' / ('%04d.json' % frame), 'w')
		json.dump(self.pid_metadata, outfile, indent=4)
		outfile.close()

	def destroy(self):
		del self.net

