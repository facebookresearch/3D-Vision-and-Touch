import os

import submitit
import argparse
import produce_sheets


parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=0, help='Random seed.')
parser.add_argument('--start', type=int, default=0, help='Random seed.')
parser.add_argument('--end', type=int, default=10000000, help='Random seed.')
parser.add_argument('--save_directory', type=str, default='experiments/checkpoint/pretrained/encoder_touch',
					help='Location of the model used to produce sheets')
parser.add_argument('--num_samples', type=int, default=4000, help='Number of points in the predicted point cloud.')
parser.add_argument('--model_location', type=str, default="../data/initial_sheet.obj")
parser.add_argument('--surf_co', type=float, default=9000.)
args = parser.parse_args()

trainer = produce_sheets.Engine(args)
submitit_logs_dir = os.path.join('experiments','sheet_logs_again',str(args.start))

executor = submitit.SlurmExecutor(submitit_logs_dir, max_num_timeout=3)
time = 360
executor.update_parameters(
	num_gpus=1,
	partition='',
	cpus_per_task=16,
	mem=500000,
	time=time,
	job_name=str(args.start),
	signal_delay_s=300,
	)
executor.submit(trainer)
