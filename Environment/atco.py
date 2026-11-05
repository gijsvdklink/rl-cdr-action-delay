"""The delay mechanism: one outstanding advisory at a time, for one aircraft.

An advisory is executed once the response delay has passed, drawn as no delay, a fixed mean, or
a lognormal with that mean. A revision of a pending advisory draws a shortened delay but can
never execute sooner than the instruction it replaces.
"""

import math

from .config import CONFIG

DELAY_MODES = ('none', 'deterministic', 'lognormal')


class ATCO:

    def __init__(self, delay_type, rng, mean_s=None):
        """Build the controller for one delay law, with the mean response time in seconds."""
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
        """The advisory awaiting execution for this aircraft, or None."""
        return self.advisory if self.cs == cs else None

    def release(self, cs):
        """Drop whatever is pending for this aircraft, as when the focus moves on."""
        if self.cs == cs:
            self.cs = self.advisory = None

    def receive(self, cs, advisory, now_s):
        """Take one advisory and set the second at which it will be executed; False if it
        repeats what is already pending.
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
            # a revision never executes sooner than the instruction it replaces
            execute_at = max(held['execute_at_s'], execute_at)

        advisory['issued_at_s']      = now_s
        advisory['response_start_s'] = self.advisory['response_start_s'] if revision else now_s
        advisory['execute_at_s']     = execute_at

        self.cs, self.advisory = cs, advisory
        return True

    def _same_instruction(self, held, advisory):
        """Whether two advisories command the same heading or the same Mach."""
        for kind in ('target_hdg', 'target_mach'):
            if kind in held and kind in advisory:
                return abs(held[kind] - advisory[kind]) < 1e-9
        return False

    def act_if_ready(self, now_s, still_flying):
        """The advisory whose delay has passed, or None; it is handed over once and then forgotten."""
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
        """One response delay in seconds, from the configured law."""
        if self.delay_type == 'none':
            return 0.0
        if self.delay_type == 'deterministic':
            return self.mean_s
        return float(self.rng.lognormal(self._mu, CONFIG['delay_sigma']))
