# Manipulation-Oriented Object Perception in Clutter through Affordance Coordinate Frames
Created by Xiaotong Chen, Kaizhi Zheng, Zhen Zeng, Shreshtha Basu, James Cooney, Jana Pavlasek and Odest Chadwicke Jenkins from 
[Laboratory For Progress, the University of Michigan](https://progress.eecs.umich.edu/index.html).  
This is a pytorch implementation of the paper: [Paper link](https://arxiv.org/abs/2010.08202), [Video link](https://youtu.be/7P9_O9wveYk).

## Requirement
The code has been tested with 
* CUDA 10.1
* Python 3.6
* PyTorch==1.1 & Torchvision==0.3

## Dataset & Checkpoints
The synthetic dataset (containing 20k images, 53.9G) : [link](https://drive.google.com/file/d/1y4wfpTqvFQ_D6JAU_1V7J6rbv-uZH6ob/view?usp=sharing). You can download the files and store them under data/.

Our synthetic dataset is generated by NDDS, which is a plugin for Unreal Engine 4. You can find the usage in this [repo](https://github.com/NVIDIA/Dataset_Synthesizer).

You can download the models used for training and test at [here](https://drive.google.com/file/d/1MjNUxhO12YMb1KeFSTLqjuGXjslhmqYW/view?usp=sharing).

The following checkpoints can be downloaded at [here](https://drive.google.com/file/d/1LFV-xtbbSeSaXJmkubbop_v4bELmfDiO/view?usp=sharing):
* endpoints_attention.pth: Attention Module + Endpoints head
* scatters_attention.pth: Attention Module + Scatters head
* norm_attention.pth: Attention Module + Norm vector head

You can store these checkpoints under weights/.
## Running deep network for perception
### Training
```
# Train a new (endpoints) model from pretrained COCO weight
python3 train.py
```

### Test
```
# Load the checkpoint and the default output is shown in output/.
python3 test.py --resume endpoints_attention
```

## Running robot experiment in CoppeliaSim platform

Download CoppeliaSim simulator at [here](https://www.coppeliarobotics.com/downloads) (tested in version 4.1.0).
The simulator is free of 3rd libraries, to run it
```
cd PATH_TO_CoppeliaSim
./coppeliaSim.sh
```

Open the `simulator/sim_env.ttt` scene in `File/Open Scene...`, you will be able to see a tabletop scene setting.

Run robot experiment after setting appropriate parameters in `simulator/main.py`
```
python3 simulator/main.py
```

If you want to use random texture for objects, we provide some sample textures [here](https://drive.google.com/file/d/1FwILIEZ9VrX7HSvWRNmS04xOw3jmajTy/view?usp=sharing). 
Extract and put this folder to path `simulator/texture`. Feel free to add or delete textures in the folder.

The robot manipulation code has two parts, one is in `simulator/robot.py`, including API calls to Coppeliasim, interface to perception network,
calculation of end-effector poses for grasping and manipulation, and utility functions. Another is in Coppeliasim scene file `remoteApiCommandServer` file and written in lua, 
including API calls to manipulator IK, motion planning, scene object setup, grasping, etc.

For more debug information related to robot simulator, refer to CoppeliaSim [official website](https://www.coppeliarobotics.com/)
 and [forum](https://forum.coppeliarobotics.com/).
