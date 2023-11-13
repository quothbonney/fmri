import torch
import random
from pathlib import Path


def validate_context_and_fix(c_ix, t_ix, batch_size):
  for i in range(batch_size):
    for context in c_ix[i]:
      if torch.equal(context, t_ix[i]):
        dim = random.choice([0, 1])
        shift = random.choice([-1, 1])
        context[dim] += shift
  return c_ix


def validate_context(c_ix, t_ix, batch_size):
  for i in range(batch_size):
    for context in c_ix[i]:
      if torch.equal(context, t_ix[i]):
        return False
  return True


# Stolen charitably from https://www.learnpytorch.io/05_pytorch_going_modular/
def save_model(model: torch.nn.Module,
               target_dir: str,
               model_name: str):
  """Saves a PyTorch model to a target directory.

  Args:
    model: A target PyTorch model to save.
    target_dir: A directory for saving the model to.
    model_name: A filename for the saved model. Should include
      either ".pth" or ".pt" as the file extension.
  Example usage:
    save_model(model=model_0,
               target_dir="models",
               model_name="05_going_modular_tingvgg_model.pth")
  """
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
