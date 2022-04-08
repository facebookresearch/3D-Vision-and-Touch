#Copyright (c) Facebook, Inc. and its affiliates.
#All rights reserved.

#This source code is licensed under the license found in the
#LICENSE file in the root directory of this source tree.

import os
import models

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import tensorflow as tf
import tensorboard as tb
from tqdm import tqdm
import sys

sys.path.insert(0, "../")
import utils
import data_loaders



class Engine():
	def __init__(self, args):

		# set seeds
		np.random.seed(args.seed)
		torch.manual_seed(args.seed)
		torch.cuda.manual_seed(args.seed)

		# set initial data values
		self.epoch = 0
		self.best_loss = 10000
		self.args = args
		self.last_improvement = 0
		self.num_samples = 10000
		self.classes = ['0001', '0002']
		self.checkpoint_dir = os.path.join('experiments/checkpoint/', args.exp_type, args.exp_id)



	def __call__(self) -> float:
		# initial data
		if  self.args.GEOmetrics:
			self.adj_info, initial_positions = utils.load_mesh_vision(self.args, f'../data/sphere.obj')
		else:
			self.adj_info, initial_positions = utils.load_mesh_vision(self.args, f'../data/vision_sheets.obj')
		self.encoder = models.Encoder(self.adj_info, Variable(initial_positions.cuda()), self.args)
		self.encoder.cuda()
		params = list(self.encoder.parameters())
		self.optimizer = optim.Adam(params, lr=self.args.lr, weight_decay=0)

		writer = SummaryWriter(os.path.join('experiments/tensorboard/', self.args.exp_type ))
		train_loader, valid_loaders = self.get_loaders()

		if self.args.eval:
			if self.args.pretrained != 'no':
				self.load_pretrained()
			else:
				self.load('')
			with torch.no_grad():
				self.validate(valid_loaders, writer)
			exit()
		# training loop
		for epoch in range(3000):
			self.epoch = epoch
			self.train(train_loader, writer)
			with torch.no_grad():
				self.validate(valid_loaders, writer)
			self.check_values()


	def get_loaders(self):
		train_data = data_loaders.mesh_loader_vision(self.classes, self.args, set_type='train', sample_num=self.num_samples)
		train_loader = DataLoader(train_data, batch_size=self.args.batch_size, shuffle=True, num_workers=16, collate_fn=train_data.collate)
		valid_loaders = []
		set_type = 'test' if self.args.eval else 'valid'
		for c in self.classes:
			valid_data = data_loaders.mesh_loader_vision(c, self.args, set_type=set_type, sample_num=self.num_samples)
			valid_loaders.append( DataLoader(valid_data, batch_size=self.args.batch_size, shuffle=False, num_workers=16, collate_fn=valid_data.collate))

		return train_loader, valid_loaders

	def train(self, data, writer):

		total_loss = 0
		iterations = 0
		self.encoder.train()
		for k, batch in enumerate(tqdm(data)):
			self.optimizer.zero_grad()

			# initialize data
			img_occ = batch['img_occ'].cuda()
			img_unocc = batch['img_unocc'].cuda()
			gt_points = batch['gt_points'].cuda()

			# inference
			# self.encoder.img_encoder.pooling(img_unocc, gt_points, debug=True)
			verts = self.encoder(img_occ, img_unocc, batch)

			# losses
			loss = utils.chamfer_distance(verts, self.adj_info['faces'], gt_points, num=self.num_samples)
			loss = self.args.loss_coeff * loss.mean()

			# backprop
			loss.backward()
			self.optimizer.step()

			# log
			message = f'Train || Epoch: {self.epoch}, loss: {loss.item():.2f}, b_ptp:  {self.best_loss:.2f}'
			tqdm.write(message)
			total_loss += loss.item()
			iterations += 1.

		writer.add_scalars('train_loss', {self.args.exp_id : total_loss / iterations}, self.epoch)




	def validate(self, data, writer):
		total_loss = 0
		# local losses at different distances from the touch sites

		self.encoder.eval()
		for v, valid_loader in enumerate(data):
			num_examples = 0
			class_loss = 0
			for k, batch in enumerate(tqdm(valid_loader)):
				# initialize data
				img_occ = batch['img_occ'].cuda()
				img_unocc = batch['img_unocc'].cuda()
				gt_points = batch['gt_points'].cuda()
				batch_size = img_occ.shape[0]
				obj_class = batch['class'][0]

				# model prediction
				verts = self.encoder(img_occ, img_unocc, batch)

				# losses
				loss = utils.chamfer_distance(verts, self.adj_info['faces'], gt_points, num=self.num_samples)

				loss = self.args.loss_coeff * loss.mean() * batch_size

				# logs
				num_examples += float(batch_size)
				class_loss += loss

			print_loss = (class_loss / num_examples)
			message = f'Valid || Epoch: {self.epoch}, class: {obj_class}, f1: {print_loss:.2f}'
			tqdm.write(message)
			total_loss += (print_loss / float(len(self.classes)))

		print('*******************************************************')
		print(f'Validation Accuracy: {total_loss}')
		print('*******************************************************')

		writer.add_scalars('valid_ptp', {self.args.exp_id: total_loss}, self.epoch)
		self.current_loss = total_loss

	def save(self, label):
		if not os.path.exists(self.checkpoint_dir):
			os.makedirs(self.checkpoint_dir)
		torch.save(self.encoder.state_dict(), self.checkpoint_dir + '/encoder_vision' + label)
		torch.save(self.optimizer.state_dict(), self.checkpoint_dir + '/optim_vision' + label)

	def check_values(self):
		if self.best_loss >= self.current_loss:
			improvement =  self.best_loss -self.current_loss
			self.best_loss = self.current_loss
			print(f'Saving Model with a {improvement} improvement')
			self.save('')
			self.last_improvement = 0
		else:
			self.last_improvement += 1
			if self.last_improvement == self.args.patience:
				print(f'Over {self.args.patience} steps since last imporvement')
				print('Exiting now')
				exit()

		if self.epoch % 10 == 0:
			print(f'Saving Model at epoch {self.epoch}')
			self.save(f'_recent')
		print('*******************************************************')

	def load(self, label):
		self.encoder.load_state_dict(torch.load(self.checkpoint_dir + '/encoder_vision' + label))
		self.optimizer.load_state_dict(torch.load(self.checkpoint_dir + '/optim_vision' + label))

	def load_pretrained(self):
		pretrained_location = 'experiments/checkpoint/pretrained/' + self.args.pretrained
		self.encoder.load_state_dict(torch.load(pretrained_location))



if __name__ == '__main__':
	import argparse

	parser = argparse.ArgumentParser()
	parser.add_argument('--seed', type=int, default=0, help='Setting for the random seed.')
	parser.add_argument('--GEOmetrics', type=int, default=0, help='use GEOMemtrics setup instead')
	parser.add_argument('--lr', type=float, default=0.0003, help='Initial learning rate.')
	parser.add_argument('--eval', action='store_true', default=False, help='Evaluate the trained model on the test set.')
	parser.add_argument('--batch_size', type=int, default=16, help='Size of the batch.')
	parser.add_argument('--exp_id', type=str, default='Eval', help='The experiment name')
	parser.add_argument('--exp_type', type=str, default='Test', help='The experiment group')
	parser.add_argument('--use_occluded', action='store_true', default=False, help='To use the occluded image.')
	parser.add_argument('--use_unoccluded', action='store_true', default=False, help='To use the unoccluded image.')
	parser.add_argument('--use_touch', action='store_true', default=False, help='To use the touch information.')
	parser.add_argument('--patience', type=int, default=30, help='How many epochs without imporvement before training stops.')
	parser.add_argument('--loss_coeff', type=float, default=9000., help='Coefficient for loss term.')
	parser.add_argument('--num_img_blocks', type=int, default=6, help='Number of image block in the image encoder.')
	parser.add_argument('--num_img_layers', type=int, default=3, help='Number of image layer in each blocl in the image encoder.')
	parser.add_argument('--size_img_ker', type=int, default=5, help='Size of the image kernel in each Image encoder layer')
	parser.add_argument('--num_gcn_layers', type=int, default=20, help='Number of GCN layer in the mesh deformation network.')
	parser.add_argument('--hidden_gcn_layers', type=int, default=300, help='Size of the feature vector for each  GCN layer in the mesh deformation network.')
	parser.add_argument('--num_grasps', type=int, default=1, help='Number of grasps in each instance to train with')
	parser.add_argument('--pretrained', type=str, default='no', help='String indicating which pretrained model to use.',
						choices=['no', 'empty', 'touch', 'touch_unoccluded', 'touch_occluded', 'unoccluded', 'occluded'])
	args = parser.parse_args()

	# update args for pretrained models
	args = utils.pretrained_args(args)

	trainer = Engine(args)
	trainer()

