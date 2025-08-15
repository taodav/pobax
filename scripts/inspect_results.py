
import orbax.checkpoint as ocp
import jax
import jax.numpy as jnp
import numpy as np  # Import numpy
from definitions import ROOT_DIR
from pathlib import Path

# Replace with the actual path to your checkpoint directory
checkpoint_path = Path(ROOT_DIR, "results", "navix_01_ppo_gd_sf_hs_diff_discrep", "Navix-DMLab-Maze-01-v0_seed(2025)_time(20250730-110442)_a86e9476a6707d1301226b4792964cc2")
# Create a checkpointer instance
checkpointer = ocp.PyTreeCheckpointer()

# Load the checkpoint
try:
    all_results = checkpointer.restore(checkpoint_path)
    print("Checkpoint loaded successfully.")
except Exception as e:
    print(f"Error loading checkpoint: {e}")
    all_results = None

# Inspect the loaded data
if all_results:
    # Print the keys in the loaded dictionary
    print("Keys in all_results:", all_results.keys())

    # Access and print some specific data
    print("Swept Hyperparameters:", all_results['swept_hparams'])
    print("Arguments:", all_results['args'])
    print("Total Runtime:", all_results['total_runtime'])

    # Example of accessing a nested value (modify as needed)
    try:
        final_eval_metric = all_results['out']['final_eval_metric']
        print("Final Eval Metric:", final_eval_metric)

        # Convert JAX arrays to NumPy arrays for easier inspection
        if isinstance(final_eval_metric, jax.Array):
            final_eval_metric = np.array(final_eval_metric)

        # Print the shape and data type of the final eval metric
        print("Shape of Final Eval Metric:", final_eval_metric.shape)
        print("Data type of Final Eval Metric:", final_eval_metric.dtype)

        # Print some values from the final eval metric (if it's not too large)
        if final_eval_metric.size < 100:
            print("Values from Final Eval Metric:", final_eval_metric)

    except KeyError as e:
        print(f"KeyError: {e}.  The 'out' or 'final_eval_metric' key might be missing in the checkpoint.")
    except Exception as e:
        print(f"Error accessing final_eval_metric: {e}")