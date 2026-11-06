"""The delay mechanism is implemented by the ATCO class.

The combined ATCO and pilot response is represented by a single action-response
delay between issuing an advisory and its execution. At most one pending advisory
is stored, together with its aircraft, issue time and scheduled execution time.
For a revision, a newly sampled delay is scaled by kappa. The revised advisory
cannot be executed earlier than the advisory it replaces (Section 4.3).
"""

import math

from .config import CONFIG

DELAY_MODES = ('none', 'deterministic', 'lognormal')


class ATCO:

    def __init__(self, delay_type, rng, mean_s=None):
        """Initialise the delay mechanism with the specified delay type and mean action-response delay in seconds."""
        if delay_type not in DELAY_MODES:
            raise ValueError(f'unknown delay type {delay_type!r}; expected one of {DELAY_MODES}')
        self.delay_type = delay_type
        self.rng        = rng
        self.mean_s     = float(mean_s if mean_s is not None else CONFIG['delay_mean_s'])

        self._mu = (math.log(self.mean_s) - CONFIG['delay_sigma'] ** 2 / 2.0
                    if delay_type == 'lognormal' else 0.0)

        self.cs       = None
        self.advisory = None

    def pending_for(self, cs):
        """Return the pending advisory for the specified aircraft, or None."""
        return self.advisory if self.cs == cs else None

    def release(self, cs):
        """Discard the pending advisory for the specified aircraft."""
        if self.cs == cs:
            self.cs = self.advisory = None

    def receive(self, cs, advisory, now_s):
        """Schedule an advisory for execution. Return False if its target matches the pending instruction.
        """
        held = self.pending_for(cs)
        if held is not None and self._same_instruction(held, advisory):
            return False

        revision = held is not None
        tau      = self._draw_delay_s()
        if revision:
            tau = CONFIG['revision_kappa'] * tau

        execute_at = now_s + round(tau)
        if revision:
            # Preserve the previous execution time as a lower bound for the revised advisory.
            execute_at = max(held['execute_at_s'], execute_at)

        advisory['issued_at_s']      = now_s
        advisory['response_start_s'] = self.advisory['response_start_s'] if revision else now_s
        advisory['execute_at_s']     = execute_at

        self.cs, self.advisory = cs, advisory
        return True

    def _same_instruction(self, held, advisory):
        """Compare the target heading or Mach number of two advisories."""
        for kind in ('target_hdg', 'target_mach'):
            if kind in held and kind in advisory:
                return abs(held[kind] - advisory[kind]) < 1e-9
        return False

    def act_if_ready(self, now_s, still_flying):
        """Return and clear an advisory eligible for execution, or None."""
        if self.advisory is None:
            return None

        if not still_flying(self.cs):
            self.cs = self.advisory = None
            return None

        if now_s < self.advisory['execute_at_s']:
            return None

        ready = (self.cs, self.advisory)
        self.cs, self.advisory = None, None
        return ready

    def _draw_delay_s(self):
        """Sample a response delay in seconds from the configured distribution."""
        if self.delay_type == 'none':
            return 0.0
        if self.delay_type == 'deterministic':
            return self.mean_s
        return float(self.rng.lognormal(self._mu, CONFIG['delay_sigma']))
