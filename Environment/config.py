"""Every constant of the simulation: traffic, sector, timing, delay distribution, reward weights, and
the layout of the action and observation spaces.

Validation scenario seeds are excluded from the training seed pool.
"""

import math as _math
from random import Random as _Random

ft   = 0.3048
kts  = 0.514444


def _speed_of_sound(alt_m):
    """The speed of sound in m/s at this altitude, for the Mach-to-knots conversion."""
    temp_k = 288.15 - 0.0065 * min(alt_m, 11000.0)
    return _math.sqrt(1.4 * 287.05287 * temp_k)


CONFIG = {
    # Aircraft settings
    'ac_type':               'A320',
    'ac_speed':              450.0,  # Nominal speed in knots, used for crossing-time estimates
    'ac_mach':               0.78,
    'ac_mach_min':           0.76,
    'ac_mach_max':           0.80,
    'mach_step':             0.02,
    'altitude':              350,  # Flight level (FL350)

    # Traffic and sector generation
    'center_ll':             (0.0, 0.0),  # Sector centre as latitude and longitude in degrees
    'n_aircraft':            lambda rng: rng.randint(15, 30),
    'rho':                   lambda rng: rng.uniform(1/20000, 1/10000),  # Aircraft per km²; sector area is calculated as count / density
    'sep_nm':                5.0,  # Minimum separation in nautical miles
    'on_route_hdg_tol_deg':  5.0,  # Heading tolerance for an on-route exit
    'buffer_nm':             10.0,  # Additional separation required when aircraft are placed
    'n_vertices':            lambda rng: rng.randint(6, 12),
    'min_circularity':       0.7,  # Minimum 4π × area / perimeter² for the sector
    'max_placement_tries':   50,
    'min_chord_nm':          15.0,  # Minimum route length through the sector

    # Simulation timing and episode limits
    'sim_dt':                1.0,  # Seconds per BlueSky step
    'action_freq':           5,  # BlueSky steps per advisory decision
    't_warn':                360.0,  # Conflict look-ahead in seconds
    'exits_per_episode':     4.0,  # Required exits as a multiple of the aircraft count
    'max_crossings':         8.0,  # Episode time limit as a multiple of sector crossing time

    # Action-response delays
    'delay_mode':            'none',
    'delay_mean_s':          30.0,  # Mean response delay in seconds
    'delay_sigma':           0.4,  # Standard deviation in log space for lognormal delays
    'revision_kappa':        0.7,  # Delay multiplier for a revised advisory

    # Observation and focus selection
    'n_neighbours':          4,  # Other aircraft included in the observation
    'focus_clear_steps':     5,  # Focus is retained briefly after a conflict clears
    'focus_emergency_u':     0.67,  # Urgency threshold above which focus can be switched

    # Reward penalty weights
    'w_los':                 10.00,  # Loss of separation
    'w_drift':               0.10,  # Deviation from the initial heading
    'w_work':                1.00,  # Advisory effort
    'seed':                  0,
}


TRAINING_SEEDS  = (1, 2, 3, 4, 5, 6, 7)
VALIDATION_SEED = 8

VALIDATION_EPISODES = 500
TRAINING_SCENARIOS  = 1_000_000_000

VALIDATION_SEEDS = tuple(_Random(VALIDATION_SEED).sample(range(TRAINING_SCENARIOS),
                                                         VALIDATION_EPISODES))
HELD_OUT = frozenset(VALIDATION_SEEDS)


NM_TO_KM = 1.852
KM_TO_NM = 1.0 / NM_TO_KM

STEP_DURATION_S = CONFIG['action_freq'] * CONFIG['sim_dt']

N_NEIGHBOURS = CONFIG['n_neighbours']
OBS_DIM      = 9 + N_NEIGHBOURS * 5

CRUISE_SPD_NMS = CONFIG['ac_speed'] / 3600.0
NMS_TO_KT      = 3600.0
CRUISE_ALT_M   = CONFIG['altitude'] * 100 * ft
KT_PER_MACH    = _speed_of_sound(CRUISE_ALT_M) / kts

EMPTY_RANGE_NM = 1000.0
NO_CONFLICT_S  = CONFIG['t_warn']

TURN_DELTAS   = {0: -60, 1: -45, 2: -30, 4: 30, 5: 45, 6: 60}
SPEED_ACTIONS = {8: +1, 9: -1}
HOLD_ACTION   = 3
RETURN_TO_INITIAL_HDG_ACTION = 7
N_ACTIONS     = 10


_TURN_30 = 0.5

ACT_COST = [
    0.75,
    0.625,
    0.5,
    0.0,
    0.5,
    0.625,
    0.75,
    0.25 * _TURN_30,
    0.5  * _TURN_30,
    0.5  * _TURN_30,
]

OBS_OWNSHIP_LABELS  = ['dpsi', 'v_own', 'h_cmd', 'v_cmd', 'retn_conf', 'pending', 'wait_s',
                       'pend_hdg', 'pend_spd']
OBS_INTRUDER_LABELS = ['dist', 'theta', 'psi', 'vint', 'tlos']
