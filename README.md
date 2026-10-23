# DAT158: learning to fly a spacecraft

This project trains a small neural network to control a simulated spacecraft
using only engine thrust and rotational torque. Move the target on the website
to see the saved policy's response, switch training checkpoints, or compare it
with manual control and a handwritten feedback controller.

- [Original live demo](https://thobiashoivik.com/lab/rocket-reinforcement-learning/)
- [Report source](doc.tex) and [compiled report](doc.pdf)
- [Training and evaluation details](scripts/rocket-rl/README.md)

This directory is a standalone reproduction of the rocket lab from the personal
website. It includes the original shared styles, physics, neural inference,
training code, and exported experiment data. It does not need the rest of the
personal website. The live link points to that original deployment; building
this package does not publish it automatically.

## Requirements

- Node.js **22.12.0 or newer**, with npm, to run or build the website.
- Python **3.11 or newer** to train or evaluate the model.
- NumPy **2.3.5**, pinned in `scripts/rocket-rl/requirements.txt`.

Training uses the CPU. A GPU, external dataset, Python web server, PyTorch,
Gymnasium, and ONNX are not required. Website dependencies are recorded in
`package-lock.json`; use `npm ci` to install those versions.

## Run the website

Run these commands from this directory, which contains `package.json`:

```sh
npm ci
npm run dev
```

Open the URL printed by Astro, normally `http://localhost:4321/`, and select the
rocket lab. The direct route is
`http://localhost:4321/lab/rocket-reinforcement-learning/`.

The included `public/models/rocket-rl.json` is sufficient to run the demo.
Retraining is optional. The browser evaluates saved actor weights and never
updates them from visitors' input.

## Evaluate the included experiment

Use a separate Python environment. On Windows PowerShell:

```powershell
python -m venv .venv-rocket-rl
.\.venv-rocket-rl\Scripts\python.exe -m pip install -r scripts/rocket-rl/requirements.txt
.\.venv-rocket-rl\Scripts\python.exe scripts/rocket-rl/evaluate.py --verify
```

On macOS or Linux:

```sh
python3 -m venv .venv-rocket-rl
.venv-rocket-rl/bin/python -m pip install -r scripts/rocket-rl/requirements.txt
.venv-rocket-rl/bin/python scripts/rocket-rl/evaluate.py --verify
```

The evaluation reloads exported actor weights and reproduces the saved
checkpoint and baseline measurements. It allows a small numerical tolerance
for the rounded weights. The fixed evaluation suite uses seed `90210`,
100 episodes of 240 steps, and randomized stationary or smooth sinusoidal
targets. It was inspected during development, so these are validation results.

Check that the browser's inference and physics match the Python fixtures:

```sh
node --experimental-strip-types scripts/rocket-rl/verify-browser.mjs
```

## Train the model again

After setting up Python, use the environment's interpreter in place of
`python` below. These commands are run from the project root:

```sh
python scripts/rocket-rl/train.py --steps 30000000 --seed 17 --output public/models/rocket-rl.json
python scripts/rocket-rl/evaluate.py --verify --fixtures scripts/rocket-rl/parity-fixtures.json
node --experimental-strip-types scripts/rocket-rl/verify-browser.mjs
```

The first command replaces the exported model. Keep a copy of the included JSON
or choose another `--output` path if experimenting. A different output can be
evaluated with `evaluate.py --model path/to/model.json --verify`.

The requested budget rounds up to **30,015,488 environment transitions**:
128 parallel simulations × 128 steps per rollout, with five optimization epochs
and minibatches of 1,024 transitions. Seed `17` controls the random experience
and minibatch order. The included run took about 234 seconds on the development
machine. Runtime and fresh training results can vary with hardware and numerical
libraries; the seed does not guarantee byte-identical output on every platform.

The script exports the actor's matrices and biases, checkpoints, simulation
configuration, learning curve, and evaluation measurements together. The
critic and learned exploration variances are used during training; visitors
only need the deterministic actor. A finite-difference check of the manually
implemented network derivatives runs before training.

## Included validation results

| Controller | Distance (m) ↓ | Thrust ↓ | Absolute spin (rad/s) ↓ | Episode return ↑ | Settled |
| --- | ---: | ---: | ---: | ---: | ---: |
| Untrained actor | 5.56569 | 0.16465 | 0.01717 | -151.67442 | 0% |
| Final PPO | 0.39996 | 0.16664 | 2.34851 | 19.71517 | 100% |
| Handwritten feedback | 0.43265 | 0.07454 | 1.21515 | 21.12771 | 99% |

Thrust is on a scale from 0 to 1. Settled means mean distance below 0.45 m and
mean speed below 0.8 m/s over the last 60 steps. After an escape, the remaining
steps receive a large tracking error and zero thrust/spin; interpret those
control costs together with tracking and success. The full table is stored in
the JSON and displayed by the lab.

## Build and deploy

```sh
npm run build
npm run preview
```

The production website is generated in `dist/`. Publish that entire directory
to a static website host. Set its build command to `npm ci && npm run build`
and its output directory to `dist`. No backend service or secret keys are
needed. Keep `models/rocket-rl.json` in the published output.

The default configuration serves the package at the domain root. For a host
that puts it under a path, such as a GitHub Pages project site, set `ASTRO_BASE`
to that path before both build and preview. For example, on PowerShell:

```powershell
$env:ASTRO_BASE = '/rocket-project/'
npm run build
npm run preview
```

On macOS or Linux:

```sh
ASTRO_BASE=/rocket-project/ npm run build
ASTRO_BASE=/rocket-project/ npm run preview
```

Use the actual repository/path name. The standalone copy makes local
navigation, model loading, and download links relative to that base.

## Files

```text
doc.tex, doc.pdf                   Report source and compiled report
scripts/rocket-rl/train.py         Physics, generated experience, PPO, evaluation, export
scripts/rocket-rl/evaluate.py      Reload exported weights and reproduce metrics
scripts/rocket-rl/requirements.txt Pinned Python dependency
scripts/rocket-rl/README.md        Algorithm and environment details
scripts/rocket-rl/parity-fixtures.json
scripts/rocket-rl/verify-browser.mjs  Python/browser agreement check
public/models/rocket-rl.json       Included experiment artifact
src/features/rocket-rl/           Browser physics and interactive demo
src/pages/                       Standalone entry point and lab route
src/layouts/, src/components/     Supporting interface components
src/styles/                      Existing shared and rocket styles
package.json, package-lock.json   Website commands and dependencies
astro.config.mjs, tsconfig.json   Website configuration
```

Commit the report, source, lockfile, and model artifact. Generated
`node_modules/`, `.venv-rocket-rl/`, `.astro/`, `dist/`, and LaTeX auxiliary files
are ignored. Keep a readable compiled report PDF in the submission repository.

## Report, sources, and attribution

`doc.tex` is a standalone LaTeX document. Open it in a LaTeX editor to compile
the PDF; an installed TeX distribution can also use `latexmk -pdf doc.tex`.
No external figures or bibliography files are required.

The project uses PPO with Generalized Advantage Estimation and Adam:

- Schulman et al. (2017), [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347).
- Schulman et al. (2015), [High-Dimensional Continuous Control Using Generalized Advantage Estimation](https://arxiv.org/abs/1506.02438).
- Kingma and Ba (2014), [Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980).
- [OpenAI Spinning Up: PPO](https://spinningup.openai.com/en/latest/algorithms/ppo.html).
- [Pezza's Work video](https://www.youtube.com/watch?v=L8XkPfY-jc8), the inspiration for the project.

