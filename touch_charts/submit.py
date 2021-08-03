#Copyright (c) Facebook, Inc. and its affiliates.
#All rights reserved.
#This source code is licensed under the license found in the
#LICENSE file in the root directory of this source tree.

import numpy as np
import os
from tqdm import tqdm



interval = 1300
commands_to_run = []
for i in range(200):
	commands_to_run += [f'python runner.py --save_director experiments/checkpoint/pretrained/encoder_touch '
						f'--start {interval*i } --end {interval*i + interval}']


def call(command):
	os.system(command)
from multiprocessing import Pool
pool = Pool(processes=10)
pbar = tqdm(pool.imap_unordered(call, commands_to_run), total=len(commands_to_run))
pbar.set_description(f"calling submitit")
for _ in pbar:
	pass















