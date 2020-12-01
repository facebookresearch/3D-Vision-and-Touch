#Copyright (c) Facebook, Inc. and its affiliates.
#All rights reserved.
#This source code is licensed under the license found in the
#LICENSE file in the root directory of this source tree.


from scipy.spatial.transform import Rotation as R
import os
from glob import glob
from tqdm import tqdm
import scipy.io as sio
import random
from PIL import Image
import numpy as np
import torch
from torchvision import transforms

preprocess = transforms.Compose([
	transforms.Resize((256, 256)),
	transforms.ToTensor()
])

# class used for obtaining an instance of the dataset for training vision chart prediction
# to be passed to a pytorch dataloader
# input:
#	- classes: list of object classes used
# 	- args: set of input parameters from the training file
# 	- set_type: the set type used
# 	- sample_num: the size of the point cloud to be returned in a given batch
class mesh_loader_vision(object):
	def __init__(self, classes, args, set_type='train', sample_num=3000):

		# initialization of data locations
		self.args = args
		self.surf_location = '../data/surfaces/'
		self.img_location = '../data/images/'
		self.touch_location = '../data/scene_info/'
		self.sheet_location = '../data/sheets/'
		self.sample_num = sample_num
		self.set_type = set_type
		self.set_list = np.load('../data/split.npy', allow_pickle='TRUE').item()

		names = [[f.split('/')[-1], f.split('/')[-2]] for f in glob((f'{self.img_location}/*/*'))]
		self.names = []
		self.classes_names = [[] for _ in classes]
		np.random.shuffle(names)
		for n in tqdm(names):
			if n[1] in classes:
				if os.path.exists(self.surf_location + n[1] + '/' + n[0] + '.npy'):
					if os.path.exists(self.touch_location + n[1] + '/' + n[0]):
						if n[0] + n[1] in self.set_list[self.set_type]:
							if n[0] +n[1] in self.set_list[self.set_type]:
								self.names.append(n)
								self.classes_names[classes.index(n[1])].append(n)

		print(f'The number of {set_type} set objects found : {len(self.names)}')

	def __len__(self):
		return len(self.names)

	# select the object and grasps for training
	def get_training_instance(self):
		# select an object and and a principle grasp randomly
		class_choice = random.choice(self.classes_names)
		object_choice = random.choice(class_choice)
		obj, obj_class = object_choice
		# select the remaining grasps and shuffle the select grasps
		num_choices = [0, 1, 2, 3, 4]
		nums = []
		for i in range(self.args.num_grasps):
			choice = random.choice(num_choices)
			nums.append(choice)
			del (num_choices[num_choices.index(choice)])
		random.shuffle(nums)
		return obj, obj_class, nums[-1], nums

	# select the object and grasps for validating
	def get_validation_examples(self, index):
		# select an object and a principle grasp
		obj, obj_class = self.names[index]
		orig_num = 0
		# select the remaining grasps deterministically
		nums = [(orig_num + i) % 5 for i in range(self.args.num_grasps)]
		return obj, obj_class, orig_num, nums

	# load surface point cloud
	def get_gt_points(self, obj_class, obj):
		samples = np.load(self.surf_location +obj_class + '/' + obj + '.npy')
		if self.args.eval:
			np.random.seed(0)
		np.random.shuffle(samples)
		gt_points = torch.FloatTensor(samples[:self.sample_num])
		gt_points *= .5 # scales the models to the size of shape we use
		gt_points[:, -1] += .6 # this is to make the hand and the shape the right releative sizes
		return gt_points

	# load vision signal
	def get_images(self, obj_class, obj, grasp_number):
		# load images
		img_occ = Image.open(f'{self.img_location}/{obj_class}/{obj}/{grasp_number}.png')
		img_unocc = Image.open(f'{self.img_location}/{obj_class}/{obj}/unoccluded.png')
		# apply pytorch image preprocessing
		img_occ = preprocess(img_occ)
		img_unocc = preprocess(img_unocc)
		return torch.FloatTensor(img_occ), torch.FloatTensor(img_unocc)

	# load touch sheet mask indicating toch success
	def get_touch_info(self, obj_class, obj, grasps):
		sheets, successful = [], []
		# cycle though grasps and load touch sheets
		for grasp in grasps:
			sheet_location = self.sheet_location + f'{obj_class}/{obj}/sheets_{grasp}_finger_num.npy'
			hand_info = np.load(f'{self.touch_location}/{obj_class}/{obj}/{grasp}.npy', allow_pickle=True).item()
			sheet, success = self.get_touch_sheets(sheet_location, hand_info)
			sheets.append(sheet)
			successful += success
		return torch.cat(sheets), successful

	# load the touch sheet
	def get_touch_sheets(self, location, hand_info):
		sheets = []
		successful = []
		touches = hand_info['touch_success']
		finger_pos = torch.FloatTensor(hand_info['cam_pos'])
		# cycle through fingers in the grasp
		for i in range(4):
			sheet = np.load(location.replace('finger_num', str(i)))
			# if the touch was unsuccessful
			if not touches[i] or sheet.shape[0] == 1:
				sheets.append(finger_pos[i].view(1, 3).expand(25, 3)) # save the finger position instead in every vertex
				successful.append(False) # binary mask for unsuccessful touch
			# if the touch was successful
			else:
				sheets.append(torch.FloatTensor(sheet)) # save the sheet
				successful.append(True) # binary mask for successful touch
		sheets = torch.stack(sheets)
		return sheets, successful

	def __getitem__(self, index):
		if self.set_type == 'train':
			obj, obj_class, grasp_number, grasps = self.get_training_instance()
		else:
			obj, obj_class, grasp_number, grasps = self.get_validation_examples(index)
		data = {}

		# meta data
		data['names'] = obj, obj_class, grasp_number
		data['class'] = obj_class

		# load sampled ground truth points
		data['gt_points'] = self.get_gt_points(obj_class, obj)

		# load images
		data['img_occ'], data['img_unocc'] = self.get_images(obj_class, obj, grasp_number)

		# get touch information
		data['sheets'], data['successful'] = self.get_touch_info(obj_class, obj, grasps)

		return data

	def collate(self, batch):
		data = {}
		data['names'] = [item['names'] for item in batch]
		data['class'] = [item['class'] for item in batch]
		data['sheets'] = torch.cat([item['sheets'].unsqueeze(0) for item in batch])
		data['gt_points'] = torch.cat([item['gt_points'].unsqueeze(0) for item in batch])
		data['img_occ'] = torch.cat([item['img_occ'].unsqueeze(0) for item in batch])
		data['img_unocc'] = torch.cat([item['img_unocc'].unsqueeze(0) for item in batch])
		data['successful'] = [item['successful'] for item in batch]

		return data


# class used for obtaining an instance of the dataset for training touch chart prediction
# to be passed to a pytorch dataloader
# input:
#	- classes: list of object classes used
# 	- args: set of input parameters from the training file
# 	- set_type: the set type used
# 	- num: if specified only returns a given grasp number
#	- all: if True use all objects, regarless of set type
#	- finger: if specified only returns a given finger number
class mesh_loader_touch(object):
	def __init__(self, classes, args, set_type='train', produce_sheets = False):

		# initialization of data locations
		self.args = args
		self.surf_location = '../data/surfaces/'
		self.img_location = '../data/images/'
		self.touch_location = '../data/scene_info/'
		self.sheet_location = '../data/remake_sheets/'
		self.set_type = set_type
		self.set_list = np.load('../data/split.npy', allow_pickle='TRUE').item()
		self.empty =  torch.FloatTensor(np.load('../data/empty_gel.npy'))
		self.produce_sheets = produce_sheets



		names = [[f.split('/')[-1], f.split('/')[-2]] for f in glob((f'{self.img_location}/*/*'))]
		self.names = []
		import os
		for n in tqdm(names):
			if n[1] in classes:
				if os.path.exists(self.surf_location + n[1]  + '/' + n[0] + '.npy'):
					if os.path.exists(self.touch_location + n[1] + '/' + n[0]):
						if self.produce_sheets or (n[0] + n[1]) in self.set_list[self.set_type]:
							if produce_sheets:
								for i in range(5):
									for j in range(4):
											self.names.append(n + [i, j])
							else:
								for i in range(5):
									hand_info = np.load(f'{self.touch_location}/{n[1]}/{n[0]}/{i}.npy',
														allow_pickle=True).item()
									for j in range(4):
										if hand_info['touch_success'][j]:
											self.names.append(n + [i, j])

		print(f'The number of {set_type} set objects found : {len(self.names)}')

	def __len__(self):
		return len(self.names)

	def standerdize_point_size(self, points):
		if points.shape[0] == 0:
			return torch.zeros((self.args.num_samples, 3))
		np.random.shuffle(points)
		points = torch.FloatTensor(points)
		while points.shape[0] < self.args.num_samples :
			points = torch.cat((points, points, points, points))
		perm = torch.randperm(points.shape[0])
		idx = perm[:self.args.num_samples ]
		return points[idx]

	def get_finger_transforms(self, hand_info, finger_num, args):
		rot = hand_info['cam_rot'][finger_num]
		rot = R.from_euler('xyz', rot, degrees=False).as_matrix()
		rot_q = R.from_matrix(rot).as_quat()
		pos = hand_info['cam_pos'][finger_num]
		return torch.FloatTensor(rot_q), torch.FloatTensor(rot), torch.FloatTensor(pos)


	def __getitem__(self, index):
		obj, obj_class, num, finger_num = self.names[index]

		# meta data
		data = {}
		data['names'] = [obj, num , finger_num]
		data['class'] = obj_class

		# hand infomation
		hand_info = np.load(f'{self.touch_location}/{obj_class}/{obj}/{num}.npy', allow_pickle=True).item()
		data['rot'], data['rot_M'], data['pos'] = self.get_finger_transforms(hand_info, finger_num, self.args)
		data['good_touch'] = hand_info['touch_success']

		# simulated touch information
		scene_info = np.load(f'{self.touch_location}/{obj_class}/{obj}/{num}.npy', allow_pickle=True).item()
		data['depth'] = torch.clamp(torch.FloatTensor(scene_info['depth'][finger_num]).unsqueeze(0), 0, 1)
		data['sim_touch']  = torch.FloatTensor(np.array(scene_info['gel'][finger_num]) / 255.).permute(2, 0, 1).contiguous().view(3, 100, 100)
		data['empty'] = torch.FloatTensor(self.empty / 255.).permute(2, 0, 1).contiguous().view(3, 100, 100)

		# point cloud information
		data['samples'] = self.standerdize_point_size(scene_info['points'][finger_num])
		data['num_samples'] = scene_info['points'][finger_num].shape

		# where to save sheets
		data['save_dir'] = f'{self.sheet_location}/{obj_class}/{obj}/sheets_{num}_{finger_num}.npy'
		return data



	def collate(self, batch):
		data = {}
		data['names'] = [item['names'] for item in batch]
		data['class'] = [item['class'] for item in batch]
		data['samples'] = torch.cat([item['samples'].unsqueeze(0) for item in batch])
		data['sim_touch'] = torch.cat([item['sim_touch'].unsqueeze(0) for item in batch])
		data['empty'] = torch.cat([item['empty'].unsqueeze(0) for item in batch])
		data['depth'] = torch.cat([item['depth'].unsqueeze(0) for item in batch])
		data['ref'] = {}
		data['ref']['rot'] = torch.cat([item['rot'].unsqueeze(0) for item in batch])
		data['ref']['rot_M'] = torch.cat([item['rot_M'].unsqueeze(0) for item in batch])
		data['ref']['pos'] = torch.cat([item['pos'].unsqueeze(0) for item in batch])
		data['good_touch'] = [item['good_touch'] for item in batch]
		data['save_dir'] = [item['save_dir'] for item in batch]
		data['num_samples'] = [item['num_samples'] for item in batch]

		return data
