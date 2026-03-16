import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import datasets
from utils import (
	save_model, 
	train, 
	create_writer
)
from models import (
	create_effnetb0,
	create_effnetb2,
	create_effnetb3
)

def execute():
	device = torch.accelerator.current_accelerator() if torch.accelerator.is_available() else torch.cpu.current_device()

	num_epochs = [5, 10]
	model_factories = [create_effnetb0, create_effnetb2, create_effnetb3]

	for epochs in num_epochs:
		for factory in model_factories:
			model, weights = factory(
				out_features=101,
				device=device,
				seed=42
			)
			
			train_dataset = datasets.Food101(
				root="data",
				split="train",
				transform=weights.transforms()
			)
			test_dataset = datasets.Food101(
				root="data",
				split="test",
				transform=weights.transforms()
			)
			classes = train_dataset.classes

			train_dataloader = DataLoader(
				dataset=train_dataset,
				batch_size=32,
				num_workers=1,
				shuffle=True
			)
			
			test_dataloader = DataLoader(
				dataset=test_dataset,
				batch_size=32,
				num_workers=1,
				shuffle=True
			)

			loss_fn = nn.CrossEntropyLoss()
			optimizer = torch.optim.Adam(
				lr=0.001,
				params=model.parameters()
			)

			writer = create_writer(
				experiment_name="food101",
				model_name=model.name,
				extra=f"{epochs}_epochs"
			)
			
			train(
				model=model,
				train_dataloader=train_dataloader,
				test_dataloader=test_dataloader,
				optimizer=optimizer,
				loss_fn=loss_fn,
				epochs=epochs,
				device=device,
				writer=writer
			)

			save_model(
				model=model,
				target_dir=f"artifacts/models/food101/{epochs}",
				model_name=model.name
			)
	
	pass