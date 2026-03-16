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
from food101_experiments import execute

def main():
	execute()
	
	pass

if __name__ == "__main__":
	main()
