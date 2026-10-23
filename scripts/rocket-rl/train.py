"""Small, reproducible PPO implementation for the browser rocket lab.

No demonstration data, control-law supervision, or heuristic action blending.
The separate heuristic below is used only as an evaluation baseline.
"""
from __future__ import annotations

import os

# Small matrix products run faster without a thread pool per operation.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse
import copy
import json
from pathlib import Path
import time

import numpy as np

CONFIG = {
    "dt": 1 / 30,
    "acceleration": 4,
    "torqueAcceleration": 6,
    "velocityDamping": .99,
    "angularDamping": .94,
    "positionScale": 3,
    "velocityScale": 3,
    "angularScale": 6,
    "escapeRadius": 5,
    "reward": {
        "progress": 4, "distance": .10, "arrival": .15, "braking": .08,
        "fuel": .008, "torque": .004, "spin": .002,
    },
}


def wrap(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi


def observation(state, target):
    return np.column_stack((
        np.clip((target[:, 0] - state[:, 0]) / 3, -2, 2),
        np.clip((target[:, 1] - state[:, 1]) / 3, -2, 2),
        np.clip(state[:, 2] / 3, -2, 2),
        np.clip(state[:, 3] / 3, -2, 2),
        np.sin(state[:, 4]), np.cos(state[:, 4]),
        np.clip(state[:, 5] / 6, -2, 2),
    )).astype(np.float32)


def step(state, target, action):
    """In-place physics; distance progress uses the same target before/after."""
    before = np.linalg.norm(state[:, :2] - target, axis=1)
    thrust = (action[:, 0] + 1) / 2
    torque = action[:, 1]
    state[:, 5] = .94 * state[:, 5] + 6 * torque / 30
    state[:, 4] = wrap(state[:, 4] + state[:, 5] / 30)
    state[:, 2] = .99 * state[:, 2] + 4 * thrust * np.cos(state[:, 4]) / 30
    state[:, 3] = .99 * state[:, 3] + 4 * thrust * np.sin(state[:, 4]) / 30
    state[:, :2] += state[:, 2:4] / 30
    distance = np.linalg.norm(state[:, :2] - target, axis=1)
    speed2 = np.sum(state[:, 2:4] ** 2, axis=1)
    near = np.exp(-2 * distance ** 2)
    reward = (4 * (before - distance) - .10 * distance + .15 * near
              - .08 * near * speed2 - .008 * thrust - .004 * torque ** 2
              - .002 * state[:, 5] ** 2)
    escaped = np.any(np.abs(state[:, :2]) > 5, axis=1)
    reward -= escaped.astype(np.float32)
    return reward.astype(np.float32), distance, escaped


class Environments:
    def __init__(self, count, rng, episode_steps=240):
        self.count, self.rng, self.episode_steps = count, rng, episode_steps
        self.state = np.zeros((count, 6), np.float32)
        self.age = np.zeros(count, np.int32)
        self.center = np.zeros((count, 2), np.float32)
        self.amplitude = np.zeros((count, 2), np.float32)
        self.frequency = np.zeros((count, 2), np.float32)
        self.phase = np.zeros((count, 2), np.float32)
        self.target = np.zeros((count, 2), np.float32)
        self.reset(np.ones(count, bool), 1)

    def reset(self, mask, difficulty):
        n, rng = int(mask.sum()), self.rng
        if not n:
            return
        self.age[mask] = 0
        self.center[mask] = rng.uniform(-.7 - .9 * difficulty, .7 + .9 * difficulty, (n, 2))
        self.amplitude[mask] = rng.uniform(.15, .55, (n, 2)) * difficulty
        moving = rng.random(n) < .5 * difficulty
        self.amplitude[mask] *= moving[:, None]
        self.frequency[mask] = rng.uniform(.35, .7, (n, 2))
        self.phase[mask] = rng.uniform(-np.pi, np.pi, (n, 2))
        self.target[mask] = self.center[mask] + self.amplitude[mask] * np.sin(self.phase[mask])
        self.state[mask, :2] = rng.uniform(-.4 - difficulty, .4 + difficulty, (n, 2))
        self.state[mask, 2:4] = rng.uniform(-.45, .45, (n, 2)) * difficulty
        delta = self.target[mask] - self.state[mask, :2]
        self.state[mask, 4] = wrap(np.arctan2(delta[:, 1], delta[:, 0])
                                 + rng.uniform(-np.pi, np.pi, n) * (.25 + .75 * difficulty))
        self.state[mask, 5] = rng.uniform(-.5, .5, n) * difficulty

    def advance(self, action, difficulty):
        reward, distance, escaped = step(self.state, self.target, action)
        self.age += 1
        done = escaped | (self.age >= self.episode_steps)
        # Return the terminal observation separately: time limits bootstrap the
        # critic while escaped episodes are genuinely terminal.
        self.target[:] = self.center + self.amplitude * np.sin(
            self.phase + self.frequency * (self.age[:, None] / 30))
        final_obs = observation(self.state, self.target)
        self.reset(done, difficulty)
        return reward, done, escaped, final_obs, distance


class MLP:
    def __init__(self, rng, output, final_scale=1):
        self.weights, self.biases = [], []
        for i, (inp, out) in enumerate(zip([7, 32, 32], [32, 32, output])):
            # Orthogonal initialization; output matrix may be rectangular.
            q, _ = np.linalg.qr(rng.normal(size=(max(inp, out), min(inp, out))))
            w = q[:out, :inp] if out >= inp else q[:inp, :out].T
            scale = np.sqrt(2) if i < 2 else final_scale
            self.weights.append((w * scale).astype(np.float32))
            self.biases.append(np.zeros(out, np.float32))

    @property
    def parameters(self):
        return [p for pair in zip(self.weights, self.biases) for p in pair]

    def forward(self, x, cache=False):
        activations = [x]
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            x = x @ w.T + b
            if i < 2:
                x = np.tanh(x)
            activations.append(x)
        return (x, activations) if cache else x

    def backward(self, activations, grad):
        grads = []
        for i in reversed(range(3)):
            if i < 2:
                grad = grad * (1 - activations[i + 1] ** 2)
            grads[:0] = [grad.T @ activations[i], grad.sum(axis=0)]
            grad = grad @ self.weights[i]
        return grads

    def export(self):
        return [{"weights": w.astype(float).round(8).tolist(), "biases": b.astype(float).round(8).tolist()}
                for w, b in zip(self.weights, self.biases)]


class Adam:
    def __init__(self, parameters):
        self.parameters = parameters
        self.m = [np.zeros_like(p) for p in parameters]
        self.v = [np.zeros_like(p) for p in parameters]
        self.t = 0

    def update(self, gradients, rate, max_norm=.5):
        self.t += 1
        norm = np.sqrt(sum(float(np.sum(g ** 2)) for g in gradients))
        scale = min(1, max_norm / (norm + 1e-8))
        for p, g, m, v in zip(self.parameters, gradients, self.m, self.v):
            g = g * scale
            m[:] = .9 * m + .1 * g
            v[:] = .999 * v + .001 * g ** 2
            p -= rate * (m / (1 - .9 ** self.t)) / (np.sqrt(v / (1 - .999 ** self.t)) + 1e-8)


def log_probability(latent, mean, logstd):
    return (-.5 * ((latent - mean) / np.exp(logstd)) ** 2
            - logstd - .5 * np.log(2 * np.pi)).sum(axis=-1)


def heuristic(state, target):
    # Position / velocity feedback chooses an acceleration; a heading controller
    # then rotates the body toward it. This is NOT used by the learned policy.
    desired = 1.8 * (target - state[:, :2]) - 2.4 * state[:, 2:4]
    heading = np.arctan2(desired[:, 1], desired[:, 0])
    error = wrap(heading - state[:, 4])
    torque = np.clip(3 * error - 1.2 * state[:, 5], -1, 1)
    thrust = np.clip(np.linalg.norm(desired, axis=1) / 4, 0, 1)
    thrust *= np.maximum(0, np.cos(error))
    return np.column_stack((2 * thrust - 1, torque)).astype(np.float32)


def evaluate(actor, seed=90210, episodes=100, episode_steps=240, mode="policy"):
    rng = np.random.default_rng(seed)
    env = Environments(episodes, rng, episode_steps)
    # Environments are never reset during evaluation: after an escape, the
    # metrics use a 7.071m failure distance and zero controls for remaining time.
    alive = np.ones(episodes, bool)
    distances, speeds = [], []
    returns = np.zeros(episodes)
    total_thrust = total_spin = 0.
    for t in range(episode_steps):
        if mode == "random":
            action = rng.uniform(-1, 1, (episodes, 2)).astype(np.float32)
        elif mode == "heuristic":
            action = heuristic(env.state, env.target)
        else:
            action = np.tanh(actor.forward(observation(env.state, env.target)))
        action[~alive] = (-1, 0)
        reward, distance, escaped = step(env.state, env.target, action)
        returns += np.where(alive, reward, -.1 * np.sqrt(50))
        alive &= ~escaped
        distances.append(np.where(alive, distance, np.sqrt(50)))
        speeds.append(np.where(alive, np.linalg.norm(env.state[:, 2:4], axis=1), 10))
        total_thrust += float(np.where(alive, (action[:, 0] + 1) / 2, 0).sum())
        total_spin += float(np.where(alive, np.abs(env.state[:, 5]), 0).sum())
        env.age += 1
        env.target[:] = env.center + env.amplitude * np.sin(
            env.phase + env.frequency * (env.age[:, None] / 30))
    distance = np.asarray(distances)
    speed = np.asarray(speeds)
    success = (distance[-60:].mean(axis=0) < .45) & (speed[-60:].mean(axis=0) < .8)
    return {"meanDistance": round(float(distance.mean()), 5),
            "meanThrust": round(total_thrust / (episodes * episode_steps), 5),
            "meanAngularSpeed": round(total_spin / (episodes * episode_steps), 5),
            "meanReturn": round(float(returns.mean()), 5),
            "successRate": round(float(success.mean()), 5)}


def gradient_check():
    """Check the manual MLP derivative against centered finite differences."""
    rng = np.random.default_rng(42)
    net = MLP(rng, 2, .1)
    x = rng.normal(size=(5, 7)).astype(np.float32)
    upstream = rng.normal(size=(5, 2)).astype(np.float32) / 5
    _, cache = net.forward(x, True)
    gradients = net.backward(cache, upstream)
    worst = 0.
    for p, g in zip(net.parameters, gradients):
        for _ in range(4):
            idx = tuple(rng.integers(s) for s in p.shape)
            orig, eps = p[idx].copy(), .002
            p[idx] = orig + eps
            plus = float(np.sum(net.forward(x) * upstream))
            p[idx] = orig - eps
            minus = float(np.sum(net.forward(x) * upstream))
            p[idx] = orig
            worst = max(worst, abs((plus - minus) / (2 * eps) - float(g[idx])))
    assert worst < .0002, f"MLP gradient error {worst}"
    print(f"Manual-backprop gradient check passed; max absolute error {worst:.8f}", flush=True)


def train(args):
    gradient_check()
    rng = np.random.default_rng(args.seed)
    actor = MLP(rng, 2, .01)
    critic = MLP(rng, 1)
    logstd = np.array([-.5, -.5], np.float32)
    actor_optimizer = Adam(actor.parameters + [logstd])
    critic_optimizer = Adam(critic.parameters)
    env = Environments(args.envs, rng)
    checkpoints, curve = [], []
    batch_size = args.envs * args.rollout
    updates = max(1, int(np.ceil(args.steps / batch_size)))
    thresholds = {"early": max(batch_size, round(args.steps * .1)),
                  "middle": round(args.steps * .4), "final": args.steps}
    start = time.monotonic()

    def checkpoint(name, steps):
        metrics = evaluate(actor)
        label = {"untrained": "Untrained", "early": "Early training", "middle": "Middle training", "final": "Final policy"}[name]
        checkpoints.append({"id": name, "label": label, "steps": steps,
                            "layers": actor.export(), "evaluation": metrics})
        print(f"Checkpoint {name} at {steps:,}: {metrics}", flush=True)

    checkpoint("untrained", 0)
    initial_curve_metrics = evaluate(actor, episodes=32)
    curve.append({"steps": 0, **{k: initial_curve_metrics[k] for k in ("meanDistance", "meanReturn")}})
    obs = observation(env.state, env.target)
    for update in range(updates):
        steps = (update + 1) * batch_size
        difficulty = min(1., steps / (args.steps * .2))
        # Rollout arrays preserve old policy probabilities for on-policy PPO.
        obs_buf, latent_buf, logp_buf, value_buf = [], [], [], []
        rewards, dones, escapes, terminal_values = [], [], [], []
        for _ in range(args.rollout):
            mean = actor.forward(obs)
            latent = mean + rng.normal(size=mean.shape).astype(np.float32) * np.exp(logstd)
            value = critic.forward(obs)[:, 0]
            reward, done, escaped, final_obs, _ = env.advance(np.tanh(latent), difficulty)
            obs_buf.append(obs)
            latent_buf.append(latent)
            logp_buf.append(log_probability(latent, mean, logstd))
            value_buf.append(value)
            rewards.append(reward)
            dones.append(done)
            escapes.append(escaped)
            terminal_values.append(critic.forward(final_obs)[:, 0])
            obs = observation(env.state, env.target)
        values = np.asarray(value_buf)
        advantage = np.zeros_like(values)
        gae = np.zeros(args.envs, np.float32)
        for t in reversed(range(args.rollout)):
            # Bootstrap truncations using the terminal observation, but don't
            # propagate GAE across either truncations or terminal resets.
            delta = rewards[t] + .99 * terminal_values[t] * ~escapes[t] - values[t]
            gae = delta + .99 * .95 * ~dones[t] * gae
            advantage[t] = gae
        returns = (advantage + values).reshape(-1)
        advantage = advantage.reshape(-1)
        advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
        inputs = np.asarray(obs_buf).reshape(-1, 7)
        latent = np.asarray(latent_buf).reshape(-1, 2)
        old_logp = np.asarray(logp_buf).reshape(-1)
        # Mild annealing retains a usable final optimization step size.
        rate = args.learning_rate * (1 - .7 * update / updates)
        for _ in range(args.epochs):
            order = rng.permutation(batch_size)
            for offset in range(0, batch_size, args.minibatch):
                ids = order[offset:offset + args.minibatch]
                count = len(ids)
                mean, cache = actor.forward(inputs[ids], True)
                lp = log_probability(latent[ids], mean, logstd)
                ratio = np.exp(np.clip(lp - old_logp[ids], -15, 15))
                adv = advantage[ids]
                active = ~(((adv > 0) & (ratio > 1.2)) | ((adv < 0) & (ratio < .8)))
                logp_gradient = -(adv * ratio * active)[:, None] / count
                residual = (latent[ids] - mean) / np.exp(2 * logstd)
                grad_mean = logp_gradient * residual
                grad_std = (logp_gradient * (((latent[ids] - mean) ** 2)
                            / np.exp(2 * logstd) - 1)).sum(axis=0) - args.entropy
                actor_optimizer.update(actor.backward(cache, grad_mean) + [grad_std], rate)
                logstd[:] = np.clip(logstd, -2, .5)
                predicted, cache = critic.forward(inputs[ids], True)
                gradient = .5 * (predicted[:, 0] - returns[ids])[:, None] / count
                critic_optimizer.update(critic.backward(cache, gradient), rate * 2, 1)
        for name, threshold in thresholds.items():
            if steps >= threshold and not any(c["id"] == name for c in checkpoints):
                checkpoint(name, steps)
        if update % max(1, updates // 30) == 0 or update == updates - 1:
            metrics = evaluate(actor, episodes=32)
            curve.append({"steps": steps, "meanDistance": metrics["meanDistance"], "meanReturn": metrics["meanReturn"]})
            elapsed = time.monotonic() - start
            print(f"{steps:,}/{updates * batch_size:,} steps | distance {metrics['meanDistance']:.3f} | return {metrics['meanReturn']:.2f} | {elapsed:.1f}s", flush=True)

    result = {
        "schemaVersion": 1, "algorithm": "PPO (NumPy, clipped surrogate + GAE)",
        "architecture": [7, 32, 32, 2], "seed": args.seed, "config": CONFIG,
        "training": {
            "steps": updates * batch_size, "parallelEnvironments": args.envs,
            "rolloutSteps": args.rollout, "epochs": args.epochs,
            "minibatchSize": args.minibatch, "learningRate": args.learning_rate,
            "discount": .99, "gaeLambda": .95, "clipRange": .2,
            "entropyCoefficient": args.entropy,
            "curriculum": "Initial positions, velocities, headings and target motion expand linearly to the full randomized distribution during the first 20% of training.",
            "elapsedSeconds": round(time.monotonic() - start, 2),
        },
        "checkpoints": checkpoints, "learningCurve": curve,
        "evaluation": {
            "episodes": 100, "seed": 90210, "episodeSteps": 240,
            "targetPattern": "50% static; 50% smooth sinusoids, randomized center, amplitude, phase and frequency",
            "successDefinition": "Final 60 steps: mean distance < 0.45 m and mean speed < 0.8 m/s",
            "escapeHandling": "Escape is terminal; remaining steps receive sqrt(50) m error and -0.1*sqrt(50) reward, zero thrust/spin. Failed episodes cannot count as successes.",
            "baselines": [
                {"id": "random", "label": "Random controls", **evaluate(actor, mode="random")},
                {"id": "heuristic", "label": "Handwritten feedback controller", **evaluate(actor, mode="heuristic")},
            ],
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, separators=(",", ":")) + "\n", encoding="utf-8")
    print(f"Saved {args.output}; elapsed {time.monotonic() - start:.1f}s", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=3_000_000)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--envs", type=int, default=128)
    parser.add_argument("--rollout", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--minibatch", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=.0004)
    parser.add_argument("--entropy", type=float, default=.003)
    parser.add_argument("--output", type=Path, default=Path("public/models/rocket-rl.json"))
    train(parser.parse_args())
