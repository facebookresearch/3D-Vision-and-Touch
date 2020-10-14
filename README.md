<!---
Copyright (c) Facebook, Inc. and its affiliates.
All rights reserved.
This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
-->
## Image-to-Set Prediction
Companion code for [E.J. Smith, et al.: 3D Shape Reconstruction from Vision and Touch](https://arxiv.org/abs/2007.03778).

This repository contains a code base and dataset for learning to fuse vision and touch signals from the 
grasp interaction of a simulated robotic hand and 3D obejct for 3D shape reconstruction. The code comes with pre-defined train/valid/test splits
over the dataset released, pretrained models, and training and evaluation scripts.

<p align="center">
  <img  src="images/Vision_Touch.png" width="570"  >
</p>

If you find this code useful in your research, please consider citing with the following BibTeX entry:

```
@misc{VisionTouch,
Author = {Edward J. Smith and Roberto Calandra and Adriana Romero and Georgia Gkioxari and David Meger and Jitendra Malik and Michal Drozdzal},
Title = {3D Shape Reconstruction from Vision and Touch},
Year = {2020},
journal = {arXiv:1911.05063},
}
```

### Installation

This code uses Python 3.6.9 , PyTorch 1.4.0. and cuda version 10.1

- Installing pytorch:
```bash
$ pip install torch==1.4.0
```

- Install dependencies
```bash
$ pip install -r requirements.txt
```

### Dataset
- To be released soon 

### Training

#### Touch Chart Prediction 

To train a model to predict touch charts, ie local geometry at each touch site, first move into the touch chart directory: 
```bash
$ cd touch_charts
``` 
To begin training call: 
```bash
$ python recon.py --exp_type <exp_type> --exp_id <exp_id> 
``` 
where  ```<exp_type>``` and  ```<exp_id>``` are the experiment type and id you wish to specify. There are a number of other arguments 
for changing the default parameters of this training, call with  ```--help``` to view them. 

Checkpoints will be saved under a directory ```"experiments/checkpoint/<exp_type>/<exp_id>/"```,  specified by ```--exp_type``` and  ```--exp_id```. 

To check training progress with Tensorboard:
```bash
$ tensorboard --logdir=experiments/tensorboard/<exp_type>/  --port=6006
```

The training above will only predict a point cloud of each signal. To optimize a mesh sheet to match this predicted point cloud 
and produce a predicted touch chart at every touch site call the following: 
```bash
$ python produce_sheets.py.py --save_directory experiments/checkpoint/<exp_type>/<exp_id>/encoder_touch
```

where ```<exp_type>``` and ```<exp_id>``` are the same settings as when training. By default the script uses the pretrained model provided to perform this optimization.
Regardless of the model used, this will take some time to complete. If you would rather just use the sheets produced using the pretrained model, ignore the previous step, return the to main directory, and call the following: 
```bash
$ mv data/pretrained_sheets data/sheets
```

#### Global Prediction

To train a model to deform vision charts around touch charts to produced a full surface prediction, first move into the vision chart directory: 
```bash
$ cd vision_charts
``` 
To begin training call: 
```bash
$ python recon.py --exp_type <exp_type> --exp_id <exp_id> 
``` 
where  ```<exp_type>``` and  ```<exp_id>``` are the experiment type and id you wish to specify. There are a number of other arguments for changing the default parameters of this training, call with  ```--help``` to view them. 

Checkpoints will be saved under a directory ```"experiments/checkpoint/<exp_type>/<exp_id>/"```,  specified by ```--exp_type``` and  ```--exp_id```. 

To check training progress with Tensorboard:
```bash
$ tensorboard --logdir=experiments/tensorboard/<exp_type>/  --port=6006
```

### Evaluation

#### Touch Chart Prediction 

Perform evaluation of the touch chart prediction, from the touch chart directory as follows:
```bash
$ python recon.py --eval --exp_type <exp_type> --exp_id <exp_id> 
``` 
where  ```<exp_type>``` and  ```<exp_id>``` are the experiment type and id specified during training. 

#### Global Prediction

Perform evaluation of the global prediction, from the vision chart directory as follows:
```bash
$ python recon.py --eval --exp_type <exp_type> --exp_id <exp_id> 
``` 
where  ```<exp_type>``` and  ```<exp_id>``` are the experiment type and id specified during training. 


### License

This codebase and dataset is released under MIT license, see [LICENSE](LICENSE.md) for details.
