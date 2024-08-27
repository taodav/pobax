import jax
import jax.numpy as jnp
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn

from pobax.models.network import shift_aug



class TorchShiftAug(nn.Module):
	"""
	Random shift image augmentation.
	Adapted from https://github.com/facebookresearch/drqv2
	"""
	def __init__(self, pad=3):
		super().__init__()
		self.pad = pad

	def forward(self, x, rand_shift=None):
		x = x.float()
		n, _, h, w = x.size()
		assert h == w
		padding = tuple([self.pad] * 4)
		x = F.pad(x, padding, mode='replicate')
		eps = 1.0 / (h + 2 * self.pad)
		arange = torch.linspace(-1.0 + eps, 1.0 - eps, h + 2 * self.pad, device=x.device, dtype=x.dtype)[:h]
		arange = arange.unsqueeze(0).repeat(h, 1).unsqueeze(2)
		base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2)
		base_grid = base_grid.unsqueeze(0).repeat(n, 1, 1, 1)

		shift = rand_shift
		if rand_shift is None:
			shift = torch.randint(0, 2 * self.pad + 1, size=(n, 1, 1, 2), device=x.device, dtype=x.dtype)

		shift *= 2.0 / (h + 2 * self.pad)
		grid = base_grid + shift
		return F.grid_sample(x, grid, padding_mode='zeros', align_corners=False)


if __name__ == "__main__":
	jax.disable_jit(True)
	seed = 2024
	pad = 3
	n = 4

	np.random.seed(seed)
	rng_key = jax.random.PRNGKey(seed)

	x = np.random.randint(low=0, high=255, size=(n, 3, 64, 64))
	rand_shift = np.random.randint(low=0, high=2 * pad + 1, size=(n, 1, 1, 2))

	torch_shift = TorchShiftAug(pad=pad)

	shifted_torch = torch_shift(torch.tensor(x, dtype=torch.float), rand_shift=torch.tensor(rand_shift, dtype=torch.float))
	shifted_jax = shift_aug(jnp.array(x, dtype=float), rng_key, rand_shift=jnp.array(rand_shift, dtype=float), pad=pad)

	np_shifted_torch = shifted_torch.numpy()
	np_shifted_jax = np.array(shifted_jax)

	assert np.all(np_shifted_jax == np_shifted_torch)

	print("test passed")



