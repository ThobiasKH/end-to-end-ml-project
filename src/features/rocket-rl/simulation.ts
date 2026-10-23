export type FlightState = {
  x: number;
  y: number;
  vx: number;
  vy: number;
  angle: number;
  omega: number;
};

export type Target = { x: number; y: number };
export type Action = { thrust: number; torque: number };
export type Layer = { weights: number[][]; biases: number[] };
export type Metrics = {
  meanDistance: number;
  meanThrust: number;
  meanAngularSpeed: number;
  meanReturn: number;
  successRate: number;
};
export type Checkpoint = {
  id: string;
  label: string;
  steps: number;
  layers: Layer[];
  evaluation: Metrics;
};
export type FlightConfig = {
  dt: number;
  acceleration: number;
  torqueAcceleration: number;
  velocityDamping: number;
  angularDamping: number;
  positionScale: number;
  velocityScale: number;
  angularScale: number;
  escapeRadius: number;
  reward: {
    progress: number;
    distance: number;
    arrival: number;
    braking: number;
    fuel: number;
    torque: number;
    spin: number;
  };
};
export type PolicyBundle = {
  schemaVersion: number;
  algorithm: string;
  architecture: number[];
  seed: number;
  config: FlightConfig;
  checkpoints: Checkpoint[];
  learningCurve: { steps: number; meanDistance: number; meanReturn: number }[];
  evaluation: {
    episodes: number;
    seed: number;
    episodeSteps: number;
    targetPattern: string;
    baselines: (Metrics & { id: string; label: string })[];
  };
};

export const DEFAULT_CONFIG: FlightConfig = {
  dt: 1 / 30,
  acceleration: 4,
  torqueAcceleration: 6,
  velocityDamping: 0.99,
  angularDamping: 0.94,
  positionScale: 3,
  velocityScale: 3,
  angularScale: 6,
  escapeRadius: 5,
  reward: {
    progress: 4,
    distance: 0.1,
    arrival: 0.15,
    braking: 0.08,
    fuel: 0.008,
    torque: 0.004,
    spin: 0.002,
  },
};

export const clamp = (value: number, min: number, max: number) =>
  Math.min(max, Math.max(min, value));

export function initialFlight(): FlightState {
  return { x: -1.4, y: -0.4, vx: 0, vy: 0, angle: Math.PI * 0.65, omega: 0 };
}

export function observation(state: FlightState, target: Target, config: FlightConfig): number[] {
  return [
    clamp((target.x - state.x) / config.positionScale, -2, 2),
    clamp((target.y - state.y) / config.positionScale, -2, 2),
    clamp(state.vx / config.velocityScale, -2, 2),
    clamp(state.vy / config.velocityScale, -2, 2),
    Math.sin(state.angle),
    Math.cos(state.angle),
    clamp(state.omega / config.angularScale, -2, 2),
  ];
}

/** The exported actor only: no critic, exploration noise, or hidden steering rules. */
export function neuralAction(state: FlightState, target: Target, policy: Checkpoint, config: FlightConfig): Action {
  let values = observation(state, target, config);
  for (const layer of policy.layers) {
    values = layer.weights.map((row, i) =>
      Math.tanh(row.reduce((sum, weight, j) => sum + weight * values[j], layer.biases[i])),
    );
  }
  return { thrust: (values[0] + 1) / 2, torque: values[1] };
}

/** Explicitly labeled conventional baseline, separate from the learned policy. */
export function heuristicAction(state: FlightState, target: Target, config: FlightConfig): Action {
  const ax = 1.8 * (target.x - state.x) - 2.4 * state.vx;
  const ay = 1.8 * (target.y - state.y) - 2.4 * state.vy;
  const error = Math.atan2(Math.sin(Math.atan2(ay, ax) - state.angle), Math.cos(Math.atan2(ay, ax) - state.angle));
  return {
    thrust: clamp(Math.hypot(ax, ay) / config.acceleration, 0, 1) * Math.max(0, Math.cos(error)),
    torque: clamp(3 * error - 1.2 * state.omega, -1, 1),
  };
}

export type RewardTerms = {
  progress: number;
  distance: number;
  arrival: number;
  braking: number;
  fuel: number;
  turning: number;
  escape: number;
};

export function advance(state: FlightState, target: Target, action: Action, config: FlightConfig) {
  const previousDistance = Math.hypot(state.x - target.x, state.y - target.y);
  const thrust = clamp(action.thrust, 0, 1);
  const torque = clamp(action.torque, -1, 1);
  const omega = config.angularDamping * state.omega + config.dt * config.torqueAcceleration * torque;
  const nextAngle = state.angle + config.dt * omega;
  const angle = Math.atan2(Math.sin(nextAngle), Math.cos(nextAngle));
  const vx = config.velocityDamping * state.vx + config.dt * config.acceleration * thrust * Math.cos(angle);
  const vy = config.velocityDamping * state.vy + config.dt * config.acceleration * thrust * Math.sin(angle);
  const next = { x: state.x + config.dt * vx, y: state.y + config.dt * vy, vx, vy, angle, omega };
  const distance = Math.hypot(next.x - target.x, next.y - target.y);
  const near = Math.exp(-2 * distance * distance);
  const escaped = Math.abs(next.x) > config.escapeRadius || Math.abs(next.y) > config.escapeRadius;
  const r = config.reward;
  const terms: RewardTerms = {
    progress: r.progress * (previousDistance - distance),
    distance: -r.distance * distance,
    arrival: r.arrival * near,
    braking: -r.braking * near * (vx * vx + vy * vy),
    fuel: -r.fuel * thrust,
    turning: -r.torque * torque * torque - r.spin * omega * omega,
    escape: escaped ? -1 : 0,
  };
  return { state: next, distance, terms, reward: Object.values(terms).reduce((sum, term) => sum + term, 0), escaped };
}

export function validateBundle(value: unknown): asserts value is PolicyBundle {
  const bundle = value as PolicyBundle;
  if (bundle?.schemaVersion !== 1 || !Array.isArray(bundle.checkpoints) || !bundle.checkpoints.length ||
      JSON.stringify(bundle.architecture) !== JSON.stringify([7, 32, 32, 2]) || !bundle.config ||
      !Array.isArray(bundle.learningCurve) || !bundle.learningCurve.length || !Array.isArray(bundle.evaluation?.baselines)) {
    throw new Error("Unsupported policy data");
  }
  const config = bundle.config;
  const positive = [config.dt, config.acceleration, config.torqueAcceleration, config.positionScale,
    config.velocityScale, config.angularScale, config.escapeRadius];
  if (positive.some(value => !Number.isFinite(value) || value <= 0) || config.dt > 1 ||
      [config.velocityDamping, config.angularDamping].some(value => !Number.isFinite(value) || value < 0 || value > 1) ||
      !config.reward || ["progress", "distance", "arrival", "braking", "fuel", "torque", "spin"].some(key =>
        !Number.isFinite(config.reward[key as keyof FlightConfig["reward"]]))) {
    throw new Error("Invalid simulation configuration");
  }
  const validMetrics = (metrics: Metrics) => metrics &&
    [metrics.meanDistance, metrics.meanThrust, metrics.meanAngularSpeed, metrics.meanReturn, metrics.successRate].every(Number.isFinite) &&
    metrics.meanDistance >= 0 && metrics.meanThrust >= 0 && metrics.meanThrust <= 1 && metrics.meanAngularSpeed >= 0 &&
    metrics.successRate >= 0 && metrics.successRate <= 1;
  const sizes = bundle.architecture;
  for (const checkpoint of bundle.checkpoints) {
    if (!checkpoint || typeof checkpoint.id !== "string" || typeof checkpoint.label !== "string" ||
        !Number.isFinite(checkpoint.steps) || checkpoint.steps < 0 || !validMetrics(checkpoint.evaluation) ||
        !Array.isArray(checkpoint.layers) || checkpoint.layers.length !== sizes.length - 1) throw new Error("Invalid policy checkpoint");
    checkpoint.layers.forEach((layer, i) => {
      if (!layer || !Array.isArray(layer.weights) || !Array.isArray(layer.biases) ||
          layer.weights.length !== sizes[i + 1] || layer.biases.length !== sizes[i + 1] ||
          layer.weights.some(row => !Array.isArray(row) || row.length !== sizes[i] || row.some(weight => !Number.isFinite(weight))) ||
          layer.biases.some(bias => !Number.isFinite(bias))) throw new Error("Invalid policy weights");
    });
  }
  if (bundle.learningCurve.some(point => !point || !Number.isFinite(point.steps) || point.steps < 0 ||
      !Number.isFinite(point.meanDistance) || point.meanDistance < 0 || !Number.isFinite(point.meanReturn)) ||
      !Number.isInteger(bundle.evaluation.episodes) || bundle.evaluation.episodes <= 0 ||
      !Number.isInteger(bundle.evaluation.episodeSteps) || bundle.evaluation.episodeSteps <= 0 ||
      typeof bundle.evaluation.targetPattern !== "string" || bundle.evaluation.baselines.some(row =>
        !validMetrics(row) || typeof row.id !== "string" || typeof row.label !== "string")) {
    throw new Error("Invalid experiment results");
  }
}
