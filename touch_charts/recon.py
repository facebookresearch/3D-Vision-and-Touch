#Copyright (c) Facebook, Inc. and its affiliates.
#All rights reserved.

#This source code is licensed under the license found in the
#LICENSE file in the root directory of this source tree.

import models
from torch.utils.tensorboard import SummaryWriter
import torch
import numpy as np
import torch.optim as optim
import os
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
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
		self.classes = ['bottle', 'knife', 'cellphone', 'rifle']
		self.checkpoint_dir = os.path.join('experiments/checkpoint/', args.exp_type, args.exp_id)
		self.log_dir = f'experiments/results/{self.args.exp_type}/{self.args.exp_id}/'
		if not os.path.exists(self.log_dir):
			os.makedirs(self.log_dir)


	def __call__(self) -> float:

		self.encoder = models.Encoder(self.args)
		self.encoder.cuda()
		params = list(self.encoder.parameters())
		self.optimizer = optim.Adam(params, lr=self.args.lr, weight_decay=0)
		writer = SummaryWriter(os.path.join('experiments/tensorboard/', args.exp_type ))

		train_loader, valid_loaders = self.get_loaders()

		if self.args.eval:
			self.load('')
			with torch.no_grad():
				self.validate(valid_loaders, writer)
			exit()
		for epoch in range(self.args.epochs):
			self.epoch = epoch
			self.train(train_loader, writer)
			with torch.no_grad():
				self.validate(valid_loaders, writer)
			self.check_values()


	def get_loaders(self):
		# training data
		train_data = data_loaders.mesh_loader_touch(self.classes, self.args, set_type='train')
		train_loader = DataLoader(train_data, batch_size=self.args.batch_size, shuffle=True, num_workers=16, collate_fn=train_data.collate)

		# validation data
		valid_loaders = []
		set_type = 'test' if self.args.eval else 'valid'
		for c in self.classes:
			valid_data = data_loaders.instance_loader(c, self.args, set_type=set_type)
			valid_loaders.append(
				DataLoader(valid_data, batch_size=self.args.batch_size, shuffle=False, num_workers=16, collate_fn=valid_data.collate))
		return train_loader, valid_loaders


	def train(self, data, writer):

		total_loss = 0
		iterations = 0
		self.encoder.train()
		for k, batch in enumerate(tqdm(data)):
			self.optimizer.zero_grad()

			# initialize data
			sim_touch = batch['sim_touch'].cuda()
			depth = batch['depth'].cuda()
			ref_frame = batch['ref']
			gt_points = batch['samples'].cuda()

			# inference
			pred_depth, pred_points = self.encoder(sim_touch, depth, ref_frame, empty = batch['empty'].cuda())

			# losses
			loss = point_loss = self.args.loss_coeff * utils.point_loss(pred_points, gt_points)
			total_loss += point_loss.item()

			# backprop
			loss.backward()
			self.optimizer.step()

			# log
			message = f'Train || Epoch: {self.epoch},  loss: {loss.item():.5f} '
			message += f'|| best_loss:  {self.best_loss :.5f}'
			tqdm.write(message)
			iterations += 1.

		writer.add_scalars('train', {self.args.exp_id: total_loss / iterations}, self.epoch)



	def validate(self, data, writer):
		total_loss = 0
		self.encoder.eval()

		# loop through every class
		for v, valid_loader in enumerate(data):
			num_examples = 0
			class_loss = 0

			# loop through every batch
			for k, batch in enumerate(tqdm(valid_loader)):

				# initialize data
				sim_touch = batch['sim_touch'].cuda()
				depth = batch['depth'].cuda()
				ref_frame = batch['ref']
				gt_points = batch['samples'].cuda()
				obj_class = batch['class'][0]
				batch_size = gt_points.shape[0]

				# inference
				pred_depth, pred_points = self.encoder( sim_touch, depth, ref_frame, empty = batch['empty'].cuda())

				# losses
				point_loss = self.args.loss_coeff * utils.point_loss(pred_points, gt_points)

				# log
				num_examples += float(batch_size)
				class_loss += point_loss * float(batch_size)


			# log
			class_loss = (class_loss / num_examples)
			message = f'Valid || Epoch: {self.epoch}, class: {obj_class}, loss: {class_loss:.5f}'
			message += f' ||  best_loss: {self.best_loss:.5f}'
			tqdm.write(message)
			total_loss += (class_loss / float(len(self.classes)))

		# log
		print('*******************************************************')
		print(f'Total validation loss: {total_loss}')
		print('*******************************************************')
		if not self.args.eval:
			writer.add_scalars('valid', {self.args.exp_id: total_loss}, self.epoch)
		self.current_loss = total_loss

	def save(self, label):
		if not os.path.exists(self.checkpoint_dir):
			os.makedirs(self.checkpoint_dir)
		torch.save(self.encoder.state_dict(), self.checkpoint_dir + '/encoder_touch' + label)
		torch.save(self.optimizer.state_dict(), self.checkpoint_dir + '/optim_touch' + label)

	def check_values(self):
		if self.best_loss >= self.current_loss:
			improvement = self.best_loss - self.current_loss
			self.best_loss = self.current_loss
			print(f'Saving Model with a {improvement} improvement in point loss')
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
		self.encoder.load_state_dict(torch.load(self.checkpoint_dir + '/encoder_touch' + label))
		self.optimizer.load_state_dict(torch.load(self.checkpoint_dir + '/optim_touch' + label))


if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument('--seed', type=int, default=0, help='Setting for the random seed.')
	parser.add_argument('--epochs', type=int, default=300, help='Number of epochs to use.')
	parser.add_argument('--lr', type=float, default=0.00003, help='Initial learning rate.')
	parser.add_argument('--eval', action='store_true', default=False, help='Evaluate the trained model on the test set.')
	parser.add_argument('--batch_size', type=int, default=32, help='Size of the batch.')
	parser.add_argument('--num_samples', type=int, default=4000, help='Number of points in the predicted point cloud.')
	parser.add_argument('--patience', type=int, default=70, help='How many epochs without imporvement before training stops.')
	parser.add_argument('--loss_coeff', type=float, default=9000., help='Coefficient for loss term.')
	parser.add_argument('--exp_id', type=str, default='test', help='The experiment name')
	parser.add_argument('--exp_type', type=str, default='test', help='The experiment group')
	args = parser.parse_args()

	trainer = Engine(args)
	trainer()

