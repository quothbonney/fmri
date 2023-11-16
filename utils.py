import torch
import random
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import io
from PIL import Image
import torchvision.transforms as transforms

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

def plot_image_and_context(targets_ix, context_ix, out, sol, context):

    vmin = -0.5
    vmax = 6
    m=4

# Create a figure and a gridspec layout with 2 rows and 6 columns
    fig = plt.figure()
    gs = gridspec.GridSpec(2, 6)


# Create the first larger subplot spanning the first two columns of the first row
    t_data = targets_ix[m].numpy()
    c_data = context_ix[m].numpy()

    ax1 = plt.subplot(gs[0, :2])
    ax1.set_title(f'Prediction (t={t_data[1]}, z={t_data[0]})')
    ax1.imshow(out[m].reshape((64,64)).detach().cpu(), cmap='magma', vmin=vmin, vmax=vmax)
    ax1.axis('off')


    ax2 = plt.subplot(gs[0, 2:4])
    ax2.set_title('Ground Truth')
    ax2.imshow(sol[m].detach().cpu(), cmap='magma', vmin=vmin, vmax=vmax)
    ax2.axis('off')

### Second row
    ax3 = plt.subplot(gs[1, 0])
    ax3.set_title(f'C1 (t={c_data[0, 1]}, z={c_data[0, 0]})')
    ax3.imshow(context[m, 0], cmap='magma', vmin=vmin, vmax=vmax)
    ax3.axis('off')

    ax4 = plt.subplot(gs[1, 1])
    ax4.set_title(f'C2 (t={c_data[1, 1]}, z={c_data[1, 0]})')
    ax4.imshow(context[m, 1], cmap='magma', vmin=vmin, vmax=vmax)
    ax4.axis('off')

    ax5 = plt.subplot(gs[1, 2])
    ax5.set_title(f'C3 (t={c_data[2, 1]}, z={c_data[2, 0]})')
    ax5.imshow(context[m, 2], cmap='magma', vmin=vmin, vmax=vmax)
    ax5.axis('off')

    ax6 = plt.subplot(gs[1, 3])
    ax6.set_title(f'C4 (t={c_data[3, 1]}, z={c_data[3, 0]})')
    ax6.imshow(context[m, 3], cmap='magma', vmin=vmin, vmax=vmax)
    ax6.axis('off')

    for ax in [ax3, ax4, ax5, ax6]:
        ax.title.set_fontsize(10)  # Set the desired title font size
    plt.tight_layout()
    
    buf = io.BytesIO()
    plt.savefig(buf, format='jpeg')
    buf.seek(0)
    image = Image.open(buf)
    tensor_image = transforms.ToTensor()(image)
    plt.close(fig)


    return tensor_image


# Show the plot

#plt.savefig("pred3.png", dpi=300)
    plt.show()
