import torch
import torchvision
from torchvision import datasets
from utils import (
	create_image_dataloader, 
	train, 
	create_writer
)
from models import (
	create_effnetb0,
	create_effnetb2,
	create_effnetb3
)
import pss_experiments

def main():
	pss_experiments.execute()

if __name__ == "__main__":
	main()
