/** Run with Node >=22.12: node --experimental-strip-types scripts/rocket-rl/verify-browser.mjs */
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { advance, neuralAction, observation, validateBundle } from "../../src/features/rocket-rl/simulation.ts";

const bundle = JSON.parse(await readFile(new URL("../../public/models/rocket-rl.json", import.meta.url), "utf8"));
const fixtures = JSON.parse(await readFile(new URL("./parity-fixtures.json", import.meta.url), "utf8"));
validateBundle(bundle);
const final = bundle.checkpoints.find(checkpoint => checkpoint.id === "final");
assert.ok(final, "The exported run includes its final actor");
const stateKeys = ["x", "y", "vx", "vy", "angle", "omega"];
let maxError = 0;
function close(actual, expected, label) {
  const error = Math.abs(actual - expected);
  maxError = Math.max(maxError, error);
  assert.ok(Number.isFinite(actual) && error < 1e-5, `${label}: ${actual} differs from Python's ${expected}`);
}

for (const [i, fixture] of fixtures.entries()) {
  const state = Object.fromEntries(stateKeys.map((key, j) => [key, fixture.state[j]]));
  const target = { x: fixture.target[0], y: fixture.target[1] };
  observation(state, target, bundle.config).forEach((value, j) => close(value, fixture.observation[j], `Fixture ${i} input ${j}`));
  const action = neuralAction(state, target, final, bundle.config);
  // Python fixtures store the bounded neural outputs before the thrust mapping.
  close(action.thrust, (fixture.action[0] + 1) / 2, `Fixture ${i} thrust`);
  close(action.torque, fixture.action[1], `Fixture ${i} torque`);
  const next = advance(state, target, action, bundle.config);
  stateKeys.forEach((key, j) => close(next.state[key], fixture.nextState[j], `Fixture ${i} next ${key}`));
  close(next.reward, fixture.reward, `Fixture ${i} reward`);
  close(next.distance, fixture.distance, `Fixture ${i} distance`);
  assert.equal(next.escaped, fixture.escaped, `Fixture ${i} escape`);
  close(Object.values(next.terms).reduce((total, term) => total + term, 0), next.reward, `Fixture ${i} reward breakdown`);
}

assert.throws(() => validateBundle({ ...bundle, config: { ...bundle.config, dt: 0 } }));
assert.throws(() => validateBundle({ ...bundle, learningCurve: [] }));
console.log(`${fixtures.length} Python/browser fixtures passed (max absolute error ${maxError.toExponential(2)}).`);
