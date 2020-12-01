#Copyright (c) Facebook, Inc. and its affiliates.
#All rights reserved.

#This source code is licensed under the license found in the
#LICENSE file in the root directory of this source tree.

import torch.nn as nn
import torch
import torch.nn.functional as F

# implemented from:
# https://github.com/MicrosoftLearning/dev290x-v2/blob/master/Mod04/02-Unet/unet_pytorch/model.py
# MIT License
class DoubleConv(nn.Module):
	def __init__(self, in_channels, out_channels, mid_channels=None):
		super().__init__()
		if not mid_channels:
			mid_channels = out_channels
		self.double_conv = nn.Sequential(
			nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
			nn.BatchNorm2d(mid_channels),
			nn.ReLU(inplace=True),
			nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
			nn.BatchNorm2d(out_channels),
			nn.ReLU(inplace=True)
		)
	def forward(self, x):
		return self.double_conv(x)

# implemented from:
# https://github.com/MicrosoftLearning/dev290x-v2/blob/master/Mod04/02-Unet/unet_pytorch/model.py
# MIT License
class Down(nn.Module):
	def __init__(self, in_channels, out_channels):
		super().__init__()
		self.maxpool_conv = nn.Sequential(
			nn.MaxPool2d(2),
			DoubleConv(in_channels, out_channels)
		)
	def forward(self, x):
		return self.maxpool_conv(x)

# implemented from:
# https://github.com/MicrosoftLearning/dev290x-v2/blob/master/Mod04/02-Unet/unet_pytorch/model.py
# MIT License
class Up(nn.Module):
	def __init__(self, in_channels, out_channels):
		super().__init__()
		self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2)
		self.conv = DoubleConv(in_channels, out_channels)

	def forward(self, x1, x2):
		x1 = self.up(x1)
		diffY = torch.tensor([x2.size()[2] - x1.size()[2]])
		diffX = torch.tensor([x2.size()[3] - x1.size()[3]])
		x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
						diffY // 2, diffY - diffY // 2])
		x = torch.cat([x2, x1], dim=1)
		output = self.conv(x)
		return output

# implemented from:
# https://github.com/MicrosoftLearning/dev290x-v2/blob/master/Mod04/02-Unet/unet_pytorch/model.py
# MIT License
class OutConv(nn.Module):
	def __init__(self, in_channels, out_channels):
		super(OutConv, self).__init__()
		self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

	def forward(self, x):
		return self.conv(x)



class Encoder(nn.Module):
	def __init__(self, args, dim = 100):
		super(Encoder, self).__init__()
		self.args = args

		# settings
		n_channels = 3
		n_classes = 1

		# downscale the image
		self.inc = DoubleConv(n_channels, 64)
		self.down1 = Down(64, 128)
		self.down2 = Down(128, 256)
		self.down3 = Down(256, 512)

		# upscale the image
		self.down4 = Down(512, 1024)
		self.up1 = Up(1024, 512)
		self.up2 = Up(512, 256)
		self.up3 = Up(256, 128)
		self.up4 = Up(128, 64)
		self.outc = OutConv(64, n_classes)

		# define a plane of the same size, and shape at the touch sensor
		width = .0218 - 0.00539
		y_z = torch.arange(dim).cuda().view(dim, 1).expand(dim, dim).float()
		y_z = torch.stack((y_z, y_z.permute(1, 0))).permute(1, 2, 0)
		plane = torch.cat((torch.zeros(dim, dim, 1).cuda(), y_z), dim=-1)
		self.orig_plane = (plane / float(dim) - .5) * width

	# update the plane with the predicted depth information
	def project_depth(self, depths, pos, rot, dim=100):
		# reshape the plane to have the same position and orientation as the touch sensor when the touch occurred
		batch_size = depths.shape[0]
		planes = self.orig_plane.view(1 , -1 , 3).expand(batch_size, -1, 3)
		planes = torch.bmm(rot, planes.permute(0, 2, 1)).permute(0, 2, 1)
		planes += pos.view(batch_size, 1, 3)

		# add the depth in the same direction as the normal of the sensor plane
		init_camera_vector = torch.FloatTensor((1, 0, 0)).cuda().view(1, 3, 1) .expand(batch_size, 3, 1 )
		camera_vector = torch.bmm(rot, init_camera_vector).permute(0, 2, 1)
		camera_vector = F.normalize(camera_vector, p=2, dim=-1).view(batch_size, 1, 1, 3).expand(batch_size, dim, dim, 3)
		depth_update = depths.unsqueeze(-1) * camera_vector
		local_depth = (planes + depth_update.view(batch_size, -1, 3)).view(batch_size, -1, 3)

		return local_depth

	def forward(self, gel, depth, ref_frame, empty, producing_sheet = False):
		# get initial data
		batch_size = ref_frame['pos'].shape[0]
		pos = ref_frame['pos'].cuda().view(batch_size, -1)
		rot_m = ref_frame['rot_M'].cuda().view(-1, 3, 3)

		# U-Net prediction
		# downscale the image
		x1 = self.inc(gel)
		x2 = self.down1(x1)
		x3 = self.down2(x2)
		x4 = self.down3(x3)
		x5 = self.down4(x4)
		# upscale the image
		x = self.up1(x5, x4)
		x = self.up2(x, x3)
		x = self.up3(x, x2)
		x = self.up4(x, x1)
		pred_depth =(self.outc(x))
		# scale the prediction
		pred_depth = F.sigmoid(pred_depth) * 0.1

		# we only want to use the points in the predicted point cloud if they correspond to pixels in the touch signal
		# which are "different" enough from the an untouched touch signal, otherwise the do not correspond to any
		# geometry of the object which is deforming the touch sensor's surface.
		diff = torch.sqrt((((gel.permute(0, 2, 3, 1) - empty.permute(0, 2, 3, 1)).view(batch_size, -1, 3)) **2).sum(dim = -1))
		useful_points = diff > 0.001
		# project the depth values into 3D points
		projected_depths = self.project_depth(pred_depth.squeeze(1), pos, rot_m).view(batch_size, -1, 3)

		pred_points = []
		for points, useful in zip(projected_depths, useful_points):
			# select only useful points
			orig_points = points.clone()
			points = points[useful]
			if points.shape[0] == 0:
				if producing_sheet:
					pred_points.append(torch.zeros((self.args.num_samples, 3)).cuda())
					continue
				else:
					points = orig_points

			# make the number of points in each element of a batch consistent
			while points.shape[0] < self.args.num_samples:
				points = torch.cat((points, points, points, points))
			perm = torch.randperm(points.shape[0])
			idx = perm[:self.args.num_samples]
			points = points[idx]
			pred_points.append(points)
		pred_points = torch.stack(pred_points)


		return pred_depth, pred_points










