import torch
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import os

def save_model(model: torch.nn.Module,
               target_dir: str,
               model_name: str):
  # Create target directory
  target_dir_path = Path(target_dir)
  target_dir_path.mkdir(parents=True,
                        exist_ok=True)

  # Create model save path
  assert model_name.endswith(".pth") or model_name.endswith(".pt"), "model_name should end with '.pt' or '.pth'"
  model_save_path = target_dir_path / model_name

  # Save the model state_dict()
  print(f"[INFO] Saving model to: {model_save_path}")
  torch.save(obj=model.state_dict(),
             f=model_save_path)

def set_seeds(seed: int):
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)

def create_writer(
    experiment_name: str,
    model_name: str,
    extra: str | None = None
) -> SummaryWriter:
  
  timestamp = datetime.now().strftime("%Y-%m-%d")

  if extra:
    log_dir = os.path.join("runs", timestamp, experiment_name, model_name, extra)
  else:
    log_dir = os.path.join("runs", timestamp, experiment_name, model_name)

  print(f"[INFO] Created SummaryWriter which will save results to: {log_dir}.")
  return SummaryWriter(log_dir=log_dir)