"""The simulation: a Gymnasium environment in which a PPO agent advises one aircraft at a
time and a modelled controller executes each advisory after a response delay.

ENVIRONMENT.md walks through the data flow, the clocks and the spaces.
"""

from .config import CONFIG, OBS_DIM, OBS_OWNSHIP_LABELS, OBS_INTRUDER_LABELS
from .atco import DELAY_MODES
from .env import AirspaceEnv
from .geometry import latlon_to_nm, nm_to_latlon

__all__ = ['AirspaceEnv', 'CONFIG', 'DELAY_MODES', 'OBS_DIM', 'OBS_OWNSHIP_LABELS',
           'OBS_INTRUDER_LABELS', 'latlon_to_nm', 'nm_to_latlon']
