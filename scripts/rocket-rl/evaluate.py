"""Re-evaluate exported actor weights, or emit browser parity fixtures."""
import argparse
import json
from pathlib import Path

import numpy as np

from train import MLP, evaluate, observation, step


def import_actor(layers):
    actor = MLP(np.random.default_rng(0), 2)
    for i, layer in enumerate(layers):
        actor.weights[i][:] = layer["weights"]
        actor.biases[i][:] = layer["biases"]
    return actor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("public/models/rocket-rl.json"))
    parser.add_argument("--verify", action="store_true", help="Check stored metrics against rounded exported weights.")
    parser.add_argument("--fixtures", type=Path, help="Save observations, actor actions and one physics step for JS parity checks.")
    args = parser.parse_args()
    model = json.loads(args.model.read_text(encoding="utf-8"))
    rows = []
    for checkpoint in model["checkpoints"]:
        actor = import_actor(checkpoint["layers"])
        metrics = evaluate(actor, seed=model["evaluation"]["seed"],
                           episodes=model["evaluation"]["episodes"],
                           episode_steps=model["evaluation"]["episodeSteps"])
        if args.verify:
            for metric, value in metrics.items():
                assert abs(value - checkpoint["evaluation"][metric]) < .0002, (checkpoint["id"], metric, value)
        rows.append({"id": checkpoint["id"], **metrics})
    for baseline in model["evaluation"]["baselines"]:
        metrics = evaluate(actor, seed=model["evaluation"]["seed"],
                           episodes=model["evaluation"]["episodes"],
                           episode_steps=model["evaluation"]["episodeSteps"], mode=baseline["id"])
        if args.verify:
            assert metrics == {k: baseline[k] for k in metrics}
        rows.append({"id": baseline["id"], **metrics})
    if args.verify:
        initial = model["checkpoints"][0]
        metrics = evaluate(import_actor(initial["layers"]), seed=model["evaluation"]["seed"],
                           episodes=32, episode_steps=model["evaluation"]["episodeSteps"])
        zero_point = model["learningCurve"][0]
        assert zero_point["steps"] == 0
        for metric in ("meanDistance", "meanReturn"):
            assert abs(metrics[metric] - zero_point[metric]) < .0002, ("zero-step curve", metric)
    print(json.dumps(rows, indent=2))
    if args.fixtures:
        states = np.array([
            [0, 0, 0, 0, 0, 0], [.8, -.3, .2, -.4, 1.2, -.7],
            [-1.7, 1.1, -1, .6, -2.3, 1.8], [2, -2, .3, -.2, 3.1, -2.4],
            [4.99, 0, 2, 0, 0, 0],
        ], np.float32)
        targets = np.array([[1, 1], [-.2, .5], [.4, -.6], [-1, 1.2], [0, 0]], np.float32)
        obs = observation(states, targets)
        actions = np.tanh(actor.forward(obs))
        next_states = states.copy()
        reward, distance, escaped = step(next_states, targets, actions)
        fixtures = [{"state": states[i].astype(float).tolist(),
                     "target": targets[i].astype(float).tolist(),
                     "observation": obs[i].astype(float).tolist(),
                     "action": actions[i].astype(float).tolist(),
                     "nextState": next_states[i].astype(float).tolist(),
                     "reward": float(reward[i]), "distance": float(distance[i]),
                     "escaped": bool(escaped[i])} for i in range(len(states))]
        args.fixtures.write_text(json.dumps(fixtures, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
