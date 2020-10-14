#Copyright (c) Facebook, Inc. and its affiliates.
#All rights reserved.
#This source code is licensed under the license found in the
#LICENSE file in the root directory of this source tree.

import numpy as np
import torch
import time
import sys
sys.path.insert(0, "../")


wait = True
while wait:
	try:
		from third_party_code import ChamferDistance
		chamfer_dist = ChamferDistance()
		wait = False
	except FileExistsError:
		time.sleep(15)
		print('trying again')
		continue


# loads the initial mesh and stores vertex, face, and adjacency matrix information
# input:
# 	- args: arguments from the training file
# 	- obj_name: name of the initial mesh object file fot eh vision charts
# output:
# 	- adj_info: the adjacency matrix, and faces for the combination of vision and touch charts
# 	- verts: the set of vertices for the initial vision charts
def load_mesh_vision(args, obj_name):
	# load obj file
	obj = import_obj(obj_name)
	verts = np.array(obj.vertices)
	verts = torch.FloatTensor(verts).cuda()
	faces = torch.LongTensor(np.array(obj.faces) - 1).cuda()
	# get adjacency matrix infomation
	adj_info = adj_init(verts, faces, args)
	return adj_info, verts


# loads object file
# involves identifying face and vertex infomation in .obj file
# needs to be triangulated to work
class import_obj(object):
	def __init__(self, file):
		self.vertices = []
		self.faces = []
		with open(file) as f :
			for line in f:
				line = line.replace('//', '/')
				line = line.replace('\n', '')
				if line[:2] == "v ":
					self.vertices.append([float(v) for v in line.split(" ")[1:]])
				elif line[0] == "f":
					self.faces.append([int(s.split('/')[0]) for s in line.split(' ')[1:]])


# normalizes symetric, binary adj matrix such that sum of each row is 1
def normalize_adj(mx):
	rowsum = mx.sum(1)
	r_inv = (1. / rowsum).view(-1)
	r_inv[r_inv != r_inv] = 0.
	mx = torch.mm(torch.eye(r_inv.shape[0]).to(mx.device) * r_inv, mx)
	return mx

# defines the adjacecny matrix for an object
def adj_init(verts, faces, args):
	# get generic adjacency matrix for vision charts
	adj = calc_adj(faces)
	adj_info = {}
	if args.use_touch:
		# this combines the adjacency information of touch and vision charts
		# the output adj matrix has the first k rows corresponding to vision charts, and the last |V| - k
		# corresponding to touch charts. Similarly the first l faces are correspond to vision charts, and the
		# remaining correspond to touch charts
		adj, faces = adj_fuse_touch(verts, faces, adj, args)

	adj = normalize_adj(adj)
	adj_info['adj'] = adj
	adj_info['faces'] = faces
	return adj_info



# combines graph for vision and touch charts to define a fused adjacency matrix
# input:
# 	- verts: vertices of the vision charts
# 	- faces: faces of the vision charts
# 	- adj: adjacency matric for the vision charts
# 	- args: arguements from the training file
# output:
# 	- adj: adjacency matrix from the combination of touch and vision charts
# 	- faces: combination of vision and touch chart faces
def adj_fuse_touch(verts, faces, adj, args):
	verts = verts.data.cpu().numpy()
	hash = {}

	# find vertices which have the same 3D position
	for e, v in enumerate(verts):
		if v.tobytes() in hash:
			hash[v.tobytes()].append(e)
		else:
			hash[v.tobytes()] = [e]

	# load object information for generic touch chart
	sheet = import_obj('../data/initial_sheet.obj')
	sheet_verts = torch.FloatTensor(np.array(sheet.vertices)).cuda()
	sheet_faces = torch.LongTensor(np.array(sheet.faces) - 1).cuda()
	sheet_adj = calc_adj(sheet_faces)

	# central vertex for each touch chart that will communicate with all vision charts
	central_point = 4
	central_points = [central_point + (i * sheet_adj.shape[0]) + adj.shape[0] for i in range(4 * args.num_grasps)]

	# define and fill new adjacency matrix with vision and touch charts
	new_dim = adj.shape[0] + (4 * args.num_grasps * sheet_adj.shape[0])
	new_adj = torch.zeros((new_dim, new_dim)).cuda()
	new_adj[: adj.shape[0], :adj.shape[0]] = adj.clone()
	for i in range(4 * args.num_grasps):
		start = adj.shape[0] + (sheet_adj.shape[0] * i)
		end = adj.shape[0] + (sheet_adj.shape[0] * (i + 1))
		new_adj[start: end, start:end] = sheet_adj.clone()
	adj = new_adj

	# define new faces with vision and touch charts
	all_faces = [faces]
	for i in range(4 * args.num_grasps):
		temp_sheet_faces = sheet_faces.clone() + verts.shape[0]
		temp_sheet_faces += i * sheet_verts.shape[0]
		all_faces.append(temp_sheet_faces)
	faces = torch.cat(all_faces)

	# update adjacency matrix to allow communication between vision and touch charts
	for key in hash.keys():
		cur_verts = hash[key]
		if len(cur_verts) > 1:
			for v1 in cur_verts:
				for v2 in cur_verts: # vertices on the boundary of vision charts can communicate
					adj[v1, v2] = 1
				if args.use_touch:
					for c in central_points: # touch and vision charts can communicate
						adj[v1, c] = 1
						adj[c, v1] = 1

	return adj, faces

# computes adjacemcy matrix from face information
def calc_adj(faces):
	v1 = faces[:, 0]
	v2 = faces[:, 1]
	v3 = faces[:, 2]
	num_verts = int(faces.max())
	adj = torch.eye(num_verts + 1).to(faces.device)

	adj[(v1, v2)] = 1
	adj[(v1, v3)] = 1
	adj[(v2, v1)] = 1
	adj[(v2, v3)] = 1
	adj[(v3, v1)] = 1
	adj[(v3, v2)] = 1

	return adj




# sample points from a batch of meshes
# implemented from:
# https://github.com/EdwardSmith1884/GEOMetrics/blob/master/utils.py
# MIT License
# input:
# 	- verts: vertices of the mesh to sample from
#	- faces: faces of the mesh to sample from
#	- num: number of point to sample
# output:
#	- points: points sampled on the surface of the mesh
def batch_sample(verts, faces, num=10000):
	dist_uni = torch.distributions.Uniform(torch.tensor([0.0]).cuda(), torch.tensor([1.0]).cuda())
	batch_size = verts.shape[0]

	# calculate area of each face
	x1, x2, x3 = torch.split(torch.index_select(verts, 1, faces[:, 0]) - torch.index_select(verts, 1, faces[:, 1]), 1,
							 dim=-1)
	y1, y2, y3 = torch.split(torch.index_select(verts, 1, faces[:, 1]) - torch.index_select(verts, 1, faces[:, 2]), 1,
							 dim=-1)
	a = (x2 * y3 - x3 * y2) ** 2
	b = (x3 * y1 - x1 * y3) ** 2
	c = (x1 * y2 - x2 * y1) ** 2
	Areas = torch.sqrt(a + b + c) / 2
	Areas = Areas.squeeze(-1) / torch.sum(Areas, dim=1)  # percentage of each face w.r.t. full surface area

	# define distrubtions of relative face surface areas
	choices = None
	for A in Areas:
		if choices is None:
			choices = torch.multinomial(A, num, True)  # list of faces to be sampled from
		else:
			choices = torch.cat((choices, torch.multinomial(A, num, True)))

	# select the faces to be used
	select_faces = faces[choices].view(verts.shape[0], 3, num)
	face_arange = verts.shape[1] * torch.arange(0, batch_size).cuda().unsqueeze(-1).expand(batch_size, num)
	select_faces = select_faces + face_arange.unsqueeze(1)
	select_faces = select_faces.view(-1, 3)
	flat_verts = verts.view(-1, 3)

	# sample one point from each
	xs = torch.index_select(flat_verts, 0, select_faces[:, 0])
	ys = torch.index_select(flat_verts, 0, select_faces[:, 1])
	zs = torch.index_select(flat_verts, 0, select_faces[:, 2])
	u = torch.sqrt(dist_uni.sample_n(batch_size * num))
	v = dist_uni.sample_n(batch_size * num)
	points = (1 - u) * xs + (u * (1 - v)) * ys + u * v * zs
	points = points.view(batch_size, num, 3)

	return points


# compute the local chamfer distance metric on the ground truth mesh at different distances away from the touch sites
# input:
# 	- samples: point cloud from surface of predicted charts
# 	- batch: current batch information
# 	- losses: the current losses across the test set
# output:
#	- losses: updates losses across the test set
#	- num_examples: the number of times the losses were updated
def calc_local_chamfer(samples, batch, losses):
	batch_size = samples.shape[0]
	# a grid of point projected towards the surface of the object, starting from the same position and orientation
	# as the touch sensor when the touch occurred, but 5 times its size
	planes = batch['radius'].cuda().view(batch_size, 4, 100, 100, 3)
	# mask indicating which point hit the surface of the object, ie, tho ones we care about
	masks = batch['radius_masks'].cuda().view(batch_size, 4, 100, 100)
	successful = batch['successful']

	num_examples = 0
	# for every grasps
	for pred, gt, mask, success in zip(samples, planes, masks, successful):
		# for every ring size around each touch site
		for i in range(5):
			# for every touch
			for j in range(4):
				if not success[j]:
					continue

				# select the right ring of points, ie 1 x size of sensor ... 5 x size of sensor
				dim_mask = torch.zeros(mask[j].shape).clone()
				dim_mask[40 - i * 10: 60 + i * 10, 40 - i * 10: 60 + i * 10] = 1
				dim_mask[50 - i * 10: 50 + i * 10, 50 - i * 10: 50 + i * 10] = 0

				# select point which are on the objects surface
				dim_mask[mask[j] == 0] = 0
				gt_masked = gt[j][dim_mask == 1]
				if (gt_masked.shape[0] == 0):
					continue

				# compute the local loss between the selected points and the predicted surface
				indices, _ = chamfer_dist(gt_masked.unsqueeze(0), pred.unsqueeze(0))
				pred_counter = pred[indices.long()[0]]
				loss = (torch.sum((pred_counter - gt_masked) ** 2, dim=1)).mean()
				losses[i] += loss
				if i == 0:
					num_examples += 1.

	return losses, num_examples

# sets up arugments for the pretrained models
def pretrained_args(args):
	if args.pretrained == 'touch':
		args.num_gcn_layers = 15
		args.hidden_gcn_layers = 250
		args.use_occluded = False
		args.use_unoccluded = False
		args.use_touch = True

	elif args.pretrained == 'touch_unoccluded':
		args.num_img_blocks = 5
		args.num_img_layers = 5
		args.size_img_ker = 5
		args.num_gcn_layers = 25
		args.hidden_gcn_layers = 200
		args.use_occluded = False
		args.use_unoccluded = True
		args.use_touch = True

	elif args.pretrained == 'touch_occluded':
		args.num_img_blocks = 5
		args.num_img_layers = 5
		args.size_img_ker = 5
		args.num_gcn_layers = 20
		args.hidden_gcn_layers = 250
		args.use_occluded = True
		args.use_unoccluded = False
		args.use_touch = True

	elif args.pretrained == 'unoccluded':
		args.num_img_blocks = 4
		args.num_img_layers = 3
		args.size_img_ker = 5
		args.num_gcn_layers = 20
		args.hidden_gcn_layers = 250
		args.use_occluded = False
		args.use_unoccluded = True
		args.use_touch = False


	elif args.pretrained == 'occluded':
		args.num_img_blocks = 5
		args.num_img_layers = 5
		args.size_img_ker = 5
		args.num_gcn_layers = 20
		args.hidden_gcn_layers = 200
		args.use_occluded = True
		args.use_unoccluded = False
		args.use_touch = False

	return args





# implemented from:
# https://github.com/EdwardSmith1884/GEOMetrics/blob/master/utils.py
# MIT License
# loads the initial mesh and returns vertex, and face information
def load_mesh_touch(obj='386.obj'):
	obj = import_obj(obj)
	verts = np.array(obj.vertices)
	verts = torch.FloatTensor(verts).cuda()
	faces = torch.LongTensor(np.array(obj.faces) - 1).cuda()
	return verts, faces




# returns the chamfer distance between a mesh and a point cloud
# input:
# 	- verts: vertices of the mesh
#	- faces: faces of the mesh
# 	- gt_points: point cloud to operate over
# output:
#	- cd: computed chamfer distance
def chamfer_distance(verts, faces, gt_points, num=1000):
	batch_size = verts.shape[0]

	# sample from faces and calculate pairs
	pred_points = batch_sample(verts, faces, num=num)
	id_p, id_g = chamfer_dist(gt_points, pred_points)

	# calculate chamfer distance
	pred_points = pred_points.view(-1, 3)
	gt_points = gt_points.contiguous().view(-1, 3)
	points_range = num * torch.arange(0, batch_size).cuda().unsqueeze(-1).expand(batch_size, num)
	id_p = (id_p.long() + points_range).view(-1)
	id_g = (id_g.long() + points_range).view(-1)
	pred_counters = torch.index_select(pred_points, 0, id_p)
	gt_counters = torch.index_select(gt_points, 0, id_g)

	dist_1 = torch.mean(torch.sum((gt_counters - pred_points) ** 2, dim=1).view(batch_size, -1), dim=-1)
	dist_2 = torch.mean(torch.sum((pred_counters - gt_points) ** 2, dim=1).view(batch_size, -1), dim=-1)
	cd = (dist_1 + dist_2)

	return cd



# implemented from:
# https://github.com/EdwardSmith1884/GEOMetrics/blob/master/utils.py
# MIT License
# compute the edgle lengths of a batch of meshes
def batch_calc_edge(verts, faces):
	# get vertex locations of faces
	p1 = torch.index_select(verts, 1, faces[:, 0])
	p2 = torch.index_select(verts, 1, faces[:, 1])
	p3 = torch.index_select(verts, 1, faces[:, 2])

	# get edge lengths
	e1 = p2 - p1
	e2 = p3 - p1
	e3 = p2 - p3

	edge_length = (torch.sum(e1 ** 2, -1).mean() + torch.sum(e2 ** 2, -1).mean() + torch.sum(e3 ** 2, -1).mean()) / 3.

	return edge_length


# returns the chamfer distance between two point clouds
# input:
# 	- gt_points: point cloud 1 to operate over
#	- pred_points: point cloud 2 to operate over
# output:
#	- cd: computed chamfer distance
def point_loss(gt_points, pred_points):
	batch_size = pred_points.shape[0]
	num = pred_points.shape[-2]

	id_p, id_g = chamfer_dist(gt_points, pred_points)
	pred_points = pred_points.view(-1, 3)
	gt_points = gt_points.contiguous().view(-1, 3)
	points_range = num * torch.arange(0, batch_size).cuda().unsqueeze(-1).expand(batch_size, num)

	id_p = (id_p.long() + points_range).view(-1)
	id_g = (id_g.long() + points_range).view(-1)
	pred_counters = torch.index_select(pred_points, 0, id_p)
	gt_counters = torch.index_select(gt_points, 0, id_g)

	dist_1 = torch.mean(torch.sum((gt_counters - pred_points) ** 2, dim=1))
	dist_2 = torch.mean(torch.sum((pred_counters - gt_points) ** 2, dim=1))
	cd = (dist_1 + dist_2)

	return cd


