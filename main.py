import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
import torchvision
import torchsummary
from utils import (
	create_image_dataloader, 
	set_seeds, 
	train, 
	create_writer
)
from models import (
	create_effnetb0,
	create_effnetb2
)

def main():
	device = torch.accelerator.current_accelerator() if torch.accelerator.is_available() else torch.cpu.current_device()

	num_epochs = [5, 10]
	model_factories = [create_effnetb0, create_effnetb2]
	train_dataloader_sources = {"pss_10": "data/pss_10/train", "pss_20": "data/pss_20/train"}

	experiment_number = 0

	for epochs in num_epochs:
		for factory in model_factories:
			for source_name, source in train_dataloader_sources.items():
				experiment_number += 1  

				model, weights = factory(
					out_features=3,
					device=device,
					seed=42
				)
				
				print(f"[INFO] Experiment number: {experiment_number}")
				print(f"[INFO] Model: {model.name}")
				print(f"[INFO] DataLoader: {source_name}")
				print(f"[INFO] Number of epochs: {epochs}")
				
				transform = weights.transforms()
				
				train_dataloader, data_classes = create_image_dataloader(
					dir=source,
					transform=transform,
					batch_size=32,
					num_workers=1
				)

				test_dataloader, _ = create_image_dataloader(
					dir="data/pss_10/train",
					transform=transform,
					batch_size=32,
					num_workers=1
				)
				
				print(f"Classes: {data_classes}")
				torchsummary.summary(model=model, input_size=(3, 224, 225))

				loss_fn = nn.CrossEntropyLoss()
				optimizer = torch.optim.Adam(
					lr=0.01, 
					params=model.parameters()
				)

				summary_writer = create_writer(
					experiment_name=source_name,
					model_name=model.name,
					extra=f"{epochs}_epochs"
				)

				results = train(
					model=model,
					train_dataloader=train_dataloader,
					test_dataloader=test_dataloader,
					optimizer=optimizer,
					loss_fn=loss_fn,
					epochs=epochs,
					device=device,
					writer=summary_writer
				)

if __name__ == "__main__":
	main()
