"""
To test LD:
check T-Maze optimal policy converges and that values are the same.

To test GVFs:
Have observation signals down a single corridor at different time steps.
The initial GVF should correspond to the discounted time-to-reach each signal

To test hidden-state dependent gammas:
TODO
"""

from pobax.algos.gd_ppo import GDPPOHyperparams

def test_ld():
    args = GDPPOHyperparams().parse_args()
    pass