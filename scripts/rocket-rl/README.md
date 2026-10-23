# Rocket reinforcement learning

This directory trains the small neural controller served by the rocket lab. It
uses genuine policy-gradient reinforcement learning from simulated experience.
There are no expert demonstrations, imitation targets, or heuristic controls
mixed into the policy. A separate handwritten controller appears only in the
evaluation comparison.

## Reproduce the shipped model

Use Python 3.11 or newer from the repository root:

```sh
python -m pip install -r scripts/rocket-rl/requirements.txt
python scripts/rocket-rl/train.py --steps 30000000 --seed 17 --output public/models/rocket-rl.json
python scripts/rocket-rl/evaluate.py --verify
```

Training uses only NumPy; a GPU, Gymnasium, PyTorch, server, and ONNX runtime are
not required. The implementation exports tiny actor matrices directly as JSON.
Training took a few minutes on the development machine; hardware and NumPy's
matrix backend affect runtime and tiny floating-point differences. Seeds fix the
experience and minibatch sampling. The number of steps rounds up to a complete
128-environment × 128-step rollout, so this command performs 30,015,488 steps.
Use a different `--output` filename to experiment without replacing the deployed
weights. A much shorter run is useful for checking installation but is not the
same experiment and should not be expected to track well.

## Environment and observations

Coordinates are meters, with positive y upwards. The nominal browser arena is
6.4 m wide and 4.8 m high. There is no gravity. The engine accelerates along the
body heading. Only thrust `[0,1]` and torque `[-1,1]` can be selected. At 30 Hz:

```text
omega = 0.94 * omega + 6 * torque * dt
angle = wrap(angle + omega * dt)
vx = 0.99 * vx + 4 * thrust * cos(angle) * dt
vy = 0.99 * vy + 4 * thrust * sin(angle) * dt
x = x + vx * dt
y = y + vy * dt
```

The seven observations are target offsets divided by 3, x/y velocity divided
by 3, sin/cos heading, and angular velocity divided by 6. All except sin/cos are
clipped to `[-2,2]`; these fixed scales are shared by Python and the browser.
Target velocity is not observed, so fast cursor changes can cause lag. Episodes
last 240 steps (8 seconds); leaving `|x| > 5` or `|y| > 5` terminates an episode.

The reward uses distance before and after the physics step with the same target:

```text
near = exp(-2 * distance_after²)
reward = 4 * (distance_before - distance_after)
         - 0.10 * distance_after + 0.15 * near
         - 0.08 * near * speed² - 0.008 * thrust
         - 0.004 * torque² - 0.002 * omega²
```

An escape adds `-1`. This reward trades off approaching the target, staying near
it, braking, thrust use, and spinning. It does not supply a desired action.

## Learning algorithm

The actor is a `7 → 32 → 32 → 2` multilayer perceptron. Both hidden layers use
tanh. During training, an independent Gaussian samples two latent actions around
the actor's final linear outputs; applying tanh bounds them before physics.
The first output is then mapped from `[-1,1]` to thrust `[0,1]`. Deployment uses
tanh of the deterministic Gaussian mean, so the exported network has a final
tanh activation too. The critic is a separate `7 → 32 → 32 → 1` network and is
used only during training.

PPO uses a clipped probability-ratio objective (`0.2`), generalized advantage
estimation (`gamma=0.99`, `lambda=0.95`), rollout-wide advantage normalization, an
entropy bonus (`0.003`), manual backpropagation, gradient clipping, and Adam.
The log-probability ratio can be computed on latent Gaussian actions: the tanh
change-of-variables terms cancel because each old/new ratio uses the same
sampled action. Entropy regularization uses the latent Gaussian entropy, rather
than claiming the same entropy after squashing.

Each update collects 128 steps in 128 environments and performs 5 epochs of
1024-sample minibatches. Actor learning rate starts at `0.0004`, decreases
linearly to 30% of its initial value, and the critic uses twice that rate.
Orthogonal initialization and a small actor output scale start the policy near
half thrust and zero torque. The backpropagation finite-difference check runs
before every training session.

During the first 20% of training, reset positions, velocities, heading errors,
angular velocities, and target motion expand linearly from a simpler control
problem to the complete randomized distribution. Thereafter, resets draw
positions and targets across the arena, arbitrary headings, small initial
velocities, and a 50/50 mixture of static and smooth sinusoidal targets. The
curriculum changes the training distribution, not the physics or reward.

## Checkpoints and evaluation

Actual actor weights are saved at initialization, approximately 10%, 40%, and
100% of requested training steps. The exact completed-step counts are stored in
the JSON. The learning curve evaluates 32 deterministic episodes at regular
training intervals; its numbers are evaluations, not noisy rollout averages.

All checkpoints and both baselines are evaluated on the same 100 episodes,
seed `90210`, held separate from the training RNG seed `17`. They use the full
randomized initial-state distribution and static/sinusoidal target mixture.
The fixed suite was inspected during development, so these are validation
results. A final test on another unseen seed remains future work.
The comparisons report mean distance, mean thrust, mean absolute angular speed,
mean episode return, and success rate. Success means mean distance below 0.45 m
and mean speed below 0.8 m/s during the final 60 steps. This tests settling and
tracking, not merely crossing the target once.

After an escape, remaining evaluation steps receive `sqrt(50)` meters of error
and `-0.1*sqrt(50)` reward; thrust/spin are recorded as zero. This keeps episode
lengths comparable and prevents early termination from improving the distance
metric. It also means failed policies can show deceptively low mean fuel use:
interpret thrust together with distance and success.

The random baseline samples uniform thrust and torque each step. The handwritten
baseline aims the body at `1.8*(target-position) - 2.4*velocity`, gates thrust by
heading alignment, and damps angular velocity with `3*heading_error - 1.2*omega`.
Its source is visible in `train.py` and it is never called in the training loop.
Stored metrics come from the actual simulator, including any disappointing
checkpoint or a baseline outperforming the neural network.

To inspect browser/Python agreement, generate fixed test states and expected
observations, actions, rewards, and next states:

```sh
python scripts/rocket-rl/evaluate.py --verify --fixtures scripts/rocket-rl/parity-fixtures.json
```

`evaluate.py --verify` reloads the rounded exported weights and checks all stored
checkpoint and baseline metrics, plus the 32-episode zero-step learning-curve
point, allowing a small floating-point tolerance.

The browser implementation is checked against those Python fixtures with the
Node.js runtime used by this website (22.12 or newer):

```sh
node --experimental-strip-types scripts/rocket-rl/verify-browser.mjs
```

This checks observation normalization, actor inference, action mapping, physics,
reward, and escape detection against the exported policy and fixed Python cases.

## References

- Schulman et al. (2017), [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347).
- Schulman et al. (2015), [High-Dimensional Continuous Control Using Generalized Advantage Estimation](https://arxiv.org/abs/1506.02438).
- Kingma and Ba (2014), [Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980).

The implementation was developed with an AI coding assistant. It is a small
educational implementation, not Stable-Baselines3 or a reproduction of a
published rocket-control benchmark. The lab's metrics describe this environment
and evaluation protocol; they do not establish control guarantees for arbitrary
cursor motion.
