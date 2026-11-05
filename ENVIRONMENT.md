# Gymnasium environment

Aircraft conflict detection and resolution is simulated in `AirspaceEnv`. Every five simulated seconds, one aircraft is selected as the *focus ship*. A heading or speed advisory can then be issued. The advisory is executed by a controller model after a response delay. Aircraft motion is simulated in BlueSky at one-second intervals.

## Scenarios

A random sector is generated with 15–30 aircraft. A320 aircraft are simulated at FL350, and departing aircraft are replaced. An episode is ended after four times the initial aircraft count has exited, or when the step limit is reached. Training scenario seeds are drawn separately from the 500 held-out seeds.

## Observations and actions

The focus ship is selected using conflict urgency and heading deviation. Its observation contains 29 values, covering its own state, any pending advisory, and four nearby or conflicting aircraft.

Ten actions are available.

- Six heading changes of ±30°, ±45° and ±60°.
- One return to the initial heading.
- Two speed changes of ±0.02 Mach, limited to Mach 0.76–0.80.
- One hold action.

Heading changes are applied relative to the last commanded heading. A pending advisory is discarded when the focus ship is changed.

## Delays and reward

Advisories are executed with no delay, a fixed delay, or a lognormal delay. The configured delay is the mean response time. Only one advisory can be pending. Repeated instructions are ignored. Revised instructions are assigned a new delay, with execution scheduled no earlier than before.

Penalties are applied for separation below 5 NM, deviation from the initial heading, and advisory effort. Separation and heading deviation are measured for the aircraft being advised.

## Usage

Commands are run from the repository root after installation.

```python
from Environment import AirspaceEnv

env = AirspaceEnv(delay_mode="none", seed=1)
obs, info = env.reset()
obs, reward, terminated, truncated, info = env.step(3)  # hold
env.close()
```

Episodes are ended through `truncated`; `terminated` is always false. Only one environment should be created per process because BlueSky state is shared within a process.

## Files

| File | Purpose |
|---|---|
| `env.py` | Gymnasium interface, observations, actions and reward |
| `config.py` | Simulation settings |
| `traffic.py` | BlueSky interface |
| `atco.py` | Response delays |
| `geometry.py` | Conflict calculations |
| `sector.py` | Sector generation |
| `stats.py` | Episode statistics |
