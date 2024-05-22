import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import ml_collections
import torch
from torch import multiprocessing as mp
import accelerate
import utils
from datasets import get_dataset
from dpm_solver_pp import NoiseScheduleVP, DPM_Solver
from absl import logging
import builtins
import einops
import libs.autoencoder
import libs.clip
from torchvision.utils import save_image
import torchvision.utils as vutils
from torchvision.transforms import ToPILImage
import seaborn as sns
import matplotlib.pyplot as plt


def vis(config):
   
    accelerator = accelerate.Accelerator()
    device = accelerator.device
    logging.info(f'Process {accelerator.process_index} using device: {device}')

    config = ml_collections.FrozenConfigDict(config)
    if accelerator.is_main_process:
        utils.set_logger(log_level='info')
    else:
        utils.set_logger(log_level='error')
        builtins.print = lambda *args: None

    nnet = utils.get_nnet(**config.nnet)
    nnet = accelerator.prepare(nnet)
    logging.info(f'load nnet from {config.nnet_path}')
    accelerator.unwrap_model(nnet).load_state_dict(torch.load(config.nnet_path, map_location='cpu'))
    nnet.eval()

    # Left weight layer names
    left_layer_names = [
        'in_blocks.0.l.weight',
        'in_blocks.4.l.weight',
        'mid_block.l.weight',
        'out_blocks.3.l.weight',
        'out_blocks.7.l.weight'
    ]

    # Right weight layer names with '.l' replaced by '.r'
    right_layer_names = [name.replace('.l.', '.r.') for name in left_layer_names]

    # Custom titles for the figures
    left_titles = ['Left First Stage Layer 1', 'Left First Stage Layer 5', 'Left First Stage Layer 9', 'Left First Stage Layer 13', 'Left First Stage Layer 17']
    right_titles = ['Right First Stage Layer 1', 'Right First Stage Layer 5', 'Right First Stage Layer 9', 'Right First Stage Layer 13', 'Right First Stage Layer 17']

    def get_nested_attr(obj, attr):
        attrs = attr.split('.')
        for at in attrs:
            if at.isdigit():
                obj = obj[int(at)]
            else:
                obj = getattr(obj, at)
        return obj

    def collect_weights(layer_names):
        all_weights = []
        for layer_name in layer_names:
            try:
                # Access weights using the nested lookup function
                layer = get_nested_attr(nnet, layer_name)
                weights = layer.data.cpu().numpy()
                print(f"{layer_name} shape: {weights.shape}")  # Debugging line to check the shape
                all_weights.append(weights)
            except Exception as e:
                print(f"Error accessing {layer_name}: {e}")
                all_weights.append(None)  # Append None if there is an error
        return all_weights

    def normalize_weights(weights):
        # Normalize the weights to a range of [0, 1]
        min_val = weights.min()
        max_val = weights.max()
        return (weights - min_val) / (max_val - min_val)

    # Collect weights for both sets of layer names
    left_weights = collect_weights(left_layer_names)
    right_weights = collect_weights(right_layer_names)

    # Apply normalization to enhance visualization
    left_weights = [normalize_weights(w) if w is not None else None for w in left_weights]
    right_weights = [normalize_weights(w) if w is not None else None for w in right_weights]

    # Determine the global min and max for the color scale, excluding None values
    valid_weights = [weights for weights in left_weights + right_weights if weights is not None]
    if not valid_weights:
        raise ValueError("No valid weights found to visualize.")

    vmin = 0  # Normalized min
    vmax = 1  # Normalized max

    def visualize_layers(layer_names, all_weights, titles, vmin, vmax, fig, axes, title_prefix, row_index):
        for i, (ax, weights, layer_name, title) in enumerate(zip(axes, all_weights, layer_names, titles)):
            if weights is not None and weights.size > 0:  # Check if weights are not None and not empty
                sns.heatmap(weights, cmap='viridis', vmin=vmin, vmax=vmax, ax=ax, annot=False, cbar=False, square=True)
                ax.set_title(f"{title}", fontsize=10)
                if row_index == 0:
                    # For the top row
                    ax.axhline(77, color='red', linewidth=1.5)
                    ax.axvline(77, color='red', linewidth=1.5)
                    ax.set_xticks([38, (weights.shape[1] + 77) // 2])
                    ax.set_yticks([38, (weights.shape[0] + 77) // 2])
                    ax.set_xticklabels(['Text', 'Image'], rotation=0)
                    if i == 0:
                        ax.set_yticklabels(['Text', 'Image'], rotation=0)
                    else:
                        ax.set_yticklabels([])
                else:
                    # For the bottom row
                    ax.set_xlabel('Features')
                    if i == 0:
                        ax.set_ylabel('Features')
                    else:
                        ax.set_yticklabels([])
                    ax.set_xticks([])
                    ax.set_yticks([])

    # Prepare to plot both sets of layers in one figure
    fig, axes = plt.subplots(2, len(left_layer_names), figsize=(len(left_layer_names) * 4, 8))

    # Visualize left weights in the first row
    visualize_layers(left_layer_names, left_weights, left_titles, vmin, vmax, fig, axes[0], "Left", 0)

    # Visualize right weights in the second row
    visualize_layers(right_layer_names, right_weights, right_titles, vmin, vmax, fig, axes[1], "Right", 1)

    # Add a single colorbar in the center right
    cbar_ax = fig.add_axes([0.92, 0.3, 0.02, 0.4])
    sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    fig.colorbar(sm, cbar_ax)

    plt.tight_layout(rect=[0, 0, 0.9, 1])
    plt.savefig('all_weights_heatmap_grid_with_single_legend.png')
    plt.close()

from absl import flags
from absl import app
from ml_collections import config_flags
import os


FLAGS = flags.FLAGS
config_flags.DEFINE_config_file(
    "config", None, "Training configuration.", lock_config=False)
flags.mark_flags_as_required(["config"])
flags.DEFINE_string("nnet_path", None, "The nnet to evaluate.")
flags.DEFINE_string("output_path", None, "The path to output images.")


def main(argv):
    config = FLAGS.config
    config.nnet_path = FLAGS.nnet_path
    config.output_path = FLAGS.output_path
    vis(config)


if __name__ == "__main__":
    app.run(main)
