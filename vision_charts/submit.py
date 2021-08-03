#Copyright (c) Facebook, Inc. and its affiliates.
#All rights reserved.
#This source code is licensed under the license found in the
#LICENSE file in the root directory of this source tree.

import numpy as np
import os
from tqdm import tqdm

def call(command):
	os.system(command)

param_namer = {'--seed': 'seed', '--num_gcn_layers': 'ngl', '--hidden_gcn_layers': 'hgl', '--num_img_blocks': 'nib',
			   '--num_img_layers': 'nil', '--num_grasps': 'grasps', '--geo': 'geo'}


commands = []
ex_type = 'Comparison'
eval = False

def add_commands(forced_params, string, params, exp_id_start):
	for f in forced_params:
		string += f' {f}'
	number = []
	keys = list(params.keys())
	for param_name in keys:
		number.append(len(params[param_name]))
	numbers = np.where(np.zeros(number) == 0 )
	numbers = np.stack(numbers).transpose()
	commands = []
	for n in numbers :
		exp_id = exp_id_start
		command = string
		for e, k in enumerate(n):
			param_name = keys[e]
			param_value =  params[param_name][k]
			command += f' {param_name} {param_value}'
			exp_id += f'_{param_namer[param_name]}_{param_value}'
		if eval:
			command += ' --eval'
		command += f' --exp_id {exp_id}'
		commands.append(command)
	return commands



######################
###### empty #########
######################
params = {'--seed': [0,1,2,3,4,5]}
exp_id_start = '@empty'
string = f'CUDA_VISIBLE_DEVICES=0  python runner.py  --exp_type {ex_type}'
forced_params = []
commands += add_commands(forced_params, string, params, exp_id_start)


######################
###### occluded ######
######################
params = {'--num_gcn_layers': [15, 20, 25], '--hidden_gcn_layers': [150, 200,  250], '--num_img_blocks': [4,5],
		  '--num_img_layers': [3, 5]}
exp_id_start = '@occluded'
string = f'CUDA_VISIBLE_DEVICES=0  python runner.py  --exp_type {ex_type}'
forced_params = ['--use_occluded']
commands += add_commands(forced_params, string, params, exp_id_start)

######################
###### unoccluded ####
######################
params = {'--num_gcn_layers': [15, 20, 25], '--hidden_gcn_layers': [150, 200,  250], '--num_img_blocks': [4,5],
		  '--num_img_layers': [3, 5]}
exp_id_start = '@unoccluded'
string = f'CUDA_VISIBLE_DEVICES=0  python runner.py  --exp_type {ex_type}'
forced_params = ['--use_unoccluded']
commands += add_commands(forced_params, string, params, exp_id_start)

########################
#### touch ######
########################
params = {'--num_gcn_layers': [15, 20, 25], '--hidden_gcn_layers': [150, 200,  250], '--num_grasps': [1, 2, 3, 4, 5]}
exp_id_start = '@touch'
string = f'CUDA_VISIBLE_DEVICES=0  python runner.py  --exp_type {ex_type}'
forced_params = ['--use_touch', ]
commands += add_commands(forced_params, string, params, exp_id_start)


##############################
##### occluded + touch #######
##############################
params = {'--num_gcn_layers': [15, 20, 25], '--hidden_gcn_layers': [150, 200,  250], '--num_img_blocks': [4,5],
		  '--num_img_layers': [3, 5]}
exp_id_start = '@occluded_touch'
string = f'CUDA_VISIBLE_DEVICES=0  python runner.py  --exp_type {ex_type}'
forced_params = ['--use_occluded', '--use_touch']
commands += add_commands(forced_params, string, params, exp_id_start)


##############################
### touch + unoccluded ######
##############################
params = {'--num_gcn_layers': [15, 20, 25], '--hidden_gcn_layers': [150, 200,  250], '--num_img_blocks': [4,5],
		  '--num_img_layers': [3, 5],'--num_grasps': [1, 2, 3, 4, 5] }
exp_id_start = '@unoccluded_touch'
string = f'CUDA_VISIBLE_DEVICES=0  python runner.py  --exp_type {ex_type}'
forced_params = ['--use_unoccluded', '--use_touch', ]
commands += add_commands(forced_params, string, params, exp_id_start)



for i in range(len(commands)):
	commands[i] +=  f'_command_{i}@'



from multiprocessing import Pool
pool = Pool(processes=10)
pbar = tqdm(pool.imap_unordered(call, commands), total=len(commands))
pbar.set_description(f"calling submitit")
for _ in pbar:
	pass















