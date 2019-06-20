Progressive Encoder and Decoder Networks for Single Image Super Resolution

We provide PyTorch implementation for SISR project. 
The current script works well with PyTorch 0.4.1+. 

Prerequisites:
- Linux or macOS
- Python 3
- CPU or NVIDIA GPU + CUDA CuDNN

Getting Started:

Installation:
- Clone this repo: git clone https://github.com/masi-aslahi/Supre-resolution.git
- cd Supre-resolution
- Install Pytorch (https://pytorch.org) and other dependencies such as visdom, torchvision, and dominate.

Train/Test:
- Download DIV2K dataset (http://www.vision.ee.ethz.ch/ntire17).
- To view training results and loss plots, run "python -m visdom.server" and click the URL http://localhost:8097.
- Train a model:
	python train.py --dataroot ./dataset/training_directory --name model_name -- model SR --which_model_netG any_from_the_network_file (i.e. upsample_4x) --which_direction AtoB

- Test the model:
	python test.py --dataroot ./dataset/testset_directory --name model_name --model SR --which_model_netG any_chosen_for_training (i.e. upsample_4x) --which_direction AtoB
	
