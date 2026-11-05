"""Display the sector and aircraft in a simple pygame window.

A trained PPO policy can be loaded, or the hold action is used by default.
Playback can be paused with Space, a new episode started with R, and the
window closed with Escape.
"""

import argparse
from pathlib import Path
import sys

import numpy as np
import pygame
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from Environment import AirspaceEnv


def main():
    """Run an interactive view with a trained policy or the hold action."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, help='saved PPO model (.zip)')
    parser.add_argument('--delay', choices=['none', 'deterministic', 'lognormal'], default='none')
    parser.add_argument('--delay-mean', type=float, default=30)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--fps', type=int, default=10)
    args = parser.parse_args()
    if args.fps < 1:
        parser.error('--fps must be positive')
    if args.delay != 'none' and args.delay_mean <= 0:
        parser.error('--delay-mean must be positive for delayed conditions')

    env = AirspaceEnv(delay_mode=args.delay, delay_mean_s=args.delay_mean, seed=args.seed)
    vector_env = DummyVecEnv([lambda: env])
    try:
        model = None
        if args.model:
            stats_path = args.model.with_name(args.model.stem + '_vecnorm.pkl')
            vector_env = VecNormalize.load(str(stats_path), vector_env)
            vector_env.training = False
            vector_env.norm_reward = False
            model = PPO.load(str(args.model), device='cpu')

        pygame.init()
        screen = pygame.display.set_mode((900, 900))
        pygame.display.set_caption('Airspace environment')
        font = pygame.font.SysFont('monospace', 16)
        clock = pygame.time.Clock()
        obs = vector_env.reset()
        paused = False
        running = True

        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key in (pygame.K_ESCAPE, pygame.K_q):
                        running = False
                    elif event.key == pygame.K_SPACE:
                        paused = not paused
                    elif event.key == pygame.K_r:
                        obs = vector_env.reset()
            if not running:
                break
            if not paused:
                action = np.array([3]) if model is None else model.predict(obs, deterministic=True)[0]
                obs, _, _, _ = vector_env.step(action)
            draw_frame(screen, font, env, paused, model is not None)
            pygame.display.flip()
            clock.tick(args.fps)
    finally:
        vector_env.close()
        pygame.quit()


def draw_frame(screen, font, env, paused, has_model):
    """Draw the sector, aircraft and current focus ship."""
    screen.fill((12, 20, 28))
    polygon = np.asarray(env.polygon)
    extent = max(float(np.abs(polygon).max()), 1)
    scale = 360 / extent

    def point(position):
        return (int(450 + position[0] * scale), int(470 - position[1] * scale))

    pygame.draw.polygon(screen, (85, 110, 125), [point(p) for p in polygon], 2)
    for callsign, row in env._row_of.items():
        location = point(env._pos[row])
        color = (100, 200, 255) if callsign == env.focus_cs else (100, 220, 150)
        pygame.draw.circle(screen, color, location, 4)
        pygame.draw.circle(screen, color, location, max(1, round(2.5 * scale)), 1)
        velocity = env._vel[row]
        pygame.draw.line(screen, color, location, point(env._pos[row] + velocity * 60), 2)
        screen.blit(font.render(callsign, True, color), (location[0] + 7, location[1] - 15))
    mode = 'Trained policy' if has_model else 'Hold action'
    status = 'Paused' if paused else 'Running'
    lines = [f'{mode} | {status} | Simulation time {env._sim_time_s:.0f} s',
             'SPACE pause | R new episode | ESC quit',
             'Blue indicates the focus ship. Rings have a radius of 2.5 NM.']
    for index, line in enumerate(lines):
        screen.blit(font.render(line, True, (220, 230, 235)), (20, 20 + index * 24))


if __name__ == '__main__':
    main()
