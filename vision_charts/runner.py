import os
import submitit
import argparse
import recon


parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=0, help='Setting for the random seed.')
parser.add_argument('--geo', type=int, default=0, help='use_geomtrics')
parser.add_argument('--lr', type=float, default=0.0003, help='Initial learning rate.')
parser.add_argument('--eval', action='store_true', default=False, help='Evaluate the trained model on the test set.')
parser.add_argument('--batch_size', type=int, default=25, help='Size of the batch.')
parser.add_argument('--exp_id', type=str, default='Eval', help='The experiment name')
parser.add_argument('--exp_type', type=str, default='Test', help='The experiment group')
parser.add_argument('--use_occluded', action='store_true', default=False, help='To use the occluded image.')
parser.add_argument('--use_unoccluded', action='store_true', default=False, help='To use the unoccluded image.')
parser.add_argument('--use_touch', action='store_true', default=False, help='To use the touch information.')
parser.add_argument('--patience', type=int, default=70, help='How many epochs without imporvement before training stops.')
parser.add_argument('--loss_coeff', type=float, default=9000., help='Coefficient for loss term.')
parser.add_argument('--num_img_blocks', type=int, default=6, help='Number of image block in the image encoder.')
parser.add_argument('--num_img_layers', type=int, default=3, help='Number of image layer in each blocl in the image encoder.')
parser.add_argument('--size_img_ker', type=int, default=5, help='Size of the image kernel in each Image encoder layer')
parser.add_argument('--num_gcn_layers', type=int, default=20, help='Number of GCN layer in the mesh deformation network.')
parser.add_argument('--hidden_gcn_layers', type=int, default=300, help='Size of the feature vector for each  GCN layer in the mesh deformation network.')
parser.add_argument('--num_grasps', type=int, default=1, help='Number of grasps in each instance to train with')
parser.add_argument('--pretrained', type=str, default='no', help='String indicating which pretrained model to use.',
					choices=['no', 'touch', 'touch_unoccluded', 'touch_occluded', 'unoccluded', 'occluded'])
parser.add_argument('--visualize', action='store_true', default=False)
args = parser.parse_args()

trainer = recon.Engine(args)
submitit_logs_dir = os.path.join('experiments','logs', args.exp_type, args.exp_id )

executor = submitit.SlurmExecutor(submitit_logs_dir, max_num_timeout=3)
if args.eval:
	time = 30
else:
	time = 60*48
executor.update_parameters(
	num_gpus=1,
	partition='',
	cpus_per_task=16,
	mem=500000,
	time=time,
	job_name=args.exp_id,
	signal_delay_s=300,
	)
executor.submit(trainer)
