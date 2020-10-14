#Copyright (c) Facebook, Inc. and its affiliates.
#All rights reserved.

#This source code is licensed under the license found in the
#LICENSE file in the root directory of this source tree.

import models
import os
import torch
import numpy as np
import torch.optim as optim
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

		self.classes = ['knife', 'bottle','cellphone', 'rifle']
		self.args = args
		self.verts, self.faces = utils.load_mesh_touch(f'../data/initial_sheet.obj')

	def __call__(self) -> float:
		self.encoder = models.Encoder(self.args)
		self.encoder.load_state_dict(torch.load(self.args.save_directory))
		self.encoder.cuda()
		self.encoder.eval()

		train_data = data_loaders.mesh_loader_touch(self.classes, self.args, produce_sheets=True)
		train_loader = DataLoader(train_data, batch_size=1, shuffle=False,
					num_workers=16, collate_fn=train_data.collate)


		for k, batch in enumerate(tqdm(train_loader, smoothing=0)):
			# initialize data
			sim_touch = batch['sim_touch'].cuda()
			depth = batch['depth'].cuda()
			ref_frame = batch['ref']

			# predict point cloud
			with torch.no_grad():
				pred_depth, sampled_points = self.encoder(sim_touch, depth, ref_frame, empty = batch['empty'].cuda())

			# optimize touch chart
			for points, dir in zip(sampled_points,batch['save_dir'] ):
				if not os.path.exists(dir[:-14]):
					os.makedirs(dir[:-14])

				# if not a successful touch
				if torch.abs(points).sum() == 0 :
					np.save(dir, np.zeros(1))
					continue

				# make initial mesh match touch sensor when touch occurred
				initial = self.verts.clone().unsqueeze(0)
				pos = ref_frame['pos'].cuda().view(1, -1)
				rot = ref_frame['rot_M'].cuda().view(1, 3, 3)
				initial = torch.bmm(rot, initial.permute(0, 2, 1)).permute(0, 2, 1)
				initial += pos.view(1, 1, 3)
				initial = initial[0]

				# set up optimization
				updates = torch.zeros(self.verts.shape, requires_grad=True, device="cuda")
				optimizer = optim.Adam([updates], lr=0.003, weight_decay=0)
				last_improvement = 0
				best_loss = 10000

				while True:
					# update
					optimizer.zero_grad()
					verts = initial + updates

					# losses
					surf_loss = utils.chamfer_distance(verts.unsqueeze(0), self.faces, points.unsqueeze(0), num =self.args.num_samples)
					edge_lengths = utils.batch_calc_edge(verts.unsqueeze(0), self.faces)
					loss = self.args.surf_co * surf_loss + 70 * edge_lengths

					# optimize
					loss.backward()
					optimizer.step()

					# check results
					if loss < 0.0006:
						break
					if best_loss > loss :
						best_loss = loss
						best_verts = verts.clone()
						last_improvement = 0
					else:
						last_improvement += 1
						if last_improvement > 50:
							break

				np.save(dir, best_verts.data.cpu().numpy())



if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument('--seed', type=int, default=0, help='Random seed.')
	parser.add_argument('--save_directory', type=str, default='../pretrained/touch_charts/encoder_touch', help='Random seed.')
	parser.add_argument('--num_samples', type=int, default=4000, help='Number of points in the predicted point cloud.')
	parser.add_argument('--model_location', type=str, default="../data/initial_sheet.obj")
	parser.add_argument('--surf_co', type=float, default=9000.)
	args = parser.parse_args()
	trainer = Engine(args)
	trainer()







