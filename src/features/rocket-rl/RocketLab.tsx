import { useEffect, useId, useRef, useState } from "react";
import "../../styles/components/lab-demo.css";
import "../../styles/features/rocket-rl.css";
import {
  advance, clamp, DEFAULT_CONFIG, heuristicAction, initialFlight, neuralAction, observation, validateBundle,
  type Action, type FlightState, type Metrics, type PolicyBundle, type RewardTerms, type Target,
} from "./simulation";

type Mode = "ai" | "manual" | "heuristic";
type TargetMode = "pointer" | "orbit" | "eight";
type Telemetry = {
  state: FlightState;
  action: Action;
  target: Target;
  distance: number;
  reward: number;
  terms: RewardTerms;
  seconds: number;
};
const emptyTerms: RewardTerms = { progress: 0, distance: 0, arrival: 0, braking: 0, fuel: 0, turning: 0, escape: 0 };
const defaultTarget = { x: 1.6, y: 0.6 };
const modelUrl = import.meta.env.BASE_URL + "models/rocket-rl.json";
const signed = (value: number, digits = 2) => `${value >= 0 ? "+" : ""}${value.toFixed(digits)}`;
const stepLabel = (steps: number) => steps >= 1_000_000 ? `${(steps / 1_000_000).toFixed(1)}m` : steps >= 1000 ? `${Math.round(steps / 1000)}k` : String(steps);

function ExperimentResults({ bundle, selected }: { bundle: PolicyBundle; selected: string }) {
  const [metric, setMetric] = useState<"meanDistance" | "meanReturn">("meanDistance");
  const chartRef = useRef<SVGSVGElement>(null);
  const [chartWidth, setChartWidth] = useState(760);
  const titleId = useId();
  const descriptionId = useId();
  useEffect(() => {
    const chart = chartRef.current;
    if (!chart) return;
    const observer = new ResizeObserver(entries => setChartWidth(Math.max(240, entries[0].contentRect.width)));
    observer.observe(chart);
    return () => observer.disconnect();
  }, []);
  const rows: (Metrics & { id: string; label: string })[] = [
    ...bundle.evaluation.baselines,
    ...bundle.checkpoints.map(checkpoint => ({ ...checkpoint.evaluation, id: checkpoint.id, label: checkpoint.label })),
  ];
  const points = bundle.learningCurve;
  const values = points.map(point => point[metric]);
  const min = metric === "meanDistance" ? 0 : Math.min(...values, 0);
  const max = Math.max(...values, metric === "meanDistance" ? 0.5 : min + 1);
  const range = Math.max(max - min, 0.01);
  const lastStep = Math.max(...points.map(point => point.steps), 1);
  const plotEnd = chartWidth - 24;
  const px = (steps: number) => 64 + (steps / lastStep) * (plotEnd - 64);
  const py = (value: number) => 210 - ((value - min) / range) * 166;
  const path = points.map((point, i) => `${i ? "L" : "M"}${px(point.steps).toFixed(2)},${py(point[metric]).toFixed(2)}`).join(" ");
  const active = bundle.checkpoints.find(checkpoint => checkpoint.id === selected);

  return (
    <section className="rocket-results" aria-labelledby={`${titleId}-heading`}>
      <div className="lab-card">
        <div className="rocket-chart-heading">
          <div>
            <p className="lab-eyebrow">Learning, measured</p>
            <h3 id={`${titleId}-heading`}>Did practice help?</h3>
          </div>
          <div className="rocket-choices" aria-label="Learning curve metric">
            <button type="button" className="rocket-choice" aria-pressed={metric === "meanDistance"} onClick={() => setMetric("meanDistance")}>Tracking error</button>
            <button type="button" className="rocket-choice" aria-pressed={metric === "meanReturn"} onClick={() => setMetric("meanReturn")}>Episode return</button>
          </div>
        </div>
        <svg ref={chartRef} className="rocket-chart" viewBox={`0 0 ${chartWidth} 250`} role="img" aria-labelledby={`${titleId} ${descriptionId}`}>
          <title id={titleId}>Measured {metric === "meanDistance" ? "tracking error" : "episode return"} during training</title>
          <desc id={descriptionId}>Horizontal axis: simulator steps. Vertical axis: {metric === "meanDistance" ? "mean distance to the target, lower is better" : "mean total episode reward, higher is better"}. Data comes from saved policy evaluations.</desc>
          {[0, 0.5, 1].map(fraction => {
            const value = min + fraction * range;
            return <g key={fraction}><line x1="64" x2={plotEnd} y1={py(value)} y2={py(value)} className="rocket-chart-grid" /><text x="54" y={py(value) + 4} textAnchor="end">{value.toFixed(1)}</text></g>;
          })}
          <text x="64" y="232">0</text>
          <text x={(64 + plotEnd) / 2} y="232" textAnchor="middle">Simulator steps</text>
          <text x={plotEnd} y="232" textAnchor="end">{stepLabel(lastStep)}</text>
          {active && <line x1={px(active.steps)} x2={px(active.steps)} y1="35" y2="210" className="rocket-chart-selected" />}
          <path d={path} className="rocket-chart-line" />
          {points.map(point => <circle key={point.steps} cx={px(point.steps)} cy={py(point[metric])} r="3" className="rocket-chart-point"><title>{stepLabel(point.steps)} steps: {point[metric].toFixed(3)}</title></circle>)}
        </svg>
        <p className="lab-note">Real saved evaluations, including the randomly initialized network at step zero. A single training seed is an experiment, not a guarantee that every run improves smoothly.</p>
      </div>
      <div className="lab-card">
        <p className="lab-eyebrow">Held-out flight tests</p>
        <h3>Against the same targets</h3>
        <p className="lab-note">{bundle.evaluation.episodes} seeded episodes per controller, {bundle.evaluation.episodeSteps} steps each. {bundle.evaluation.targetPattern}</p>
        <div className="rocket-table-wrap" tabIndex={0} role="region" aria-label="Controller evaluation results">
          <table className="rocket-results-table">
            <caption>Mean results over the held-out test set</caption>
            <thead><tr><th scope="col">Controller</th><th scope="col">Distance ↓</th><th scope="col">Thrust ↓</th><th scope="col">Spin ↓</th><th scope="col">Return ↑</th><th scope="col">Settled ↑</th></tr></thead>
            <tbody>{rows.map(row => <tr key={row.id} className={row.id === selected ? "rocket-current-row" : undefined}><th scope="row">{row.label}{row.id === selected && <span className="rocket-selected-tag">selected</span>}</th><td>{row.meanDistance.toFixed(2)} m</td><td>{(row.meanThrust * 100).toFixed(0)}%</td><td>{row.meanAngularSpeed.toFixed(2)} rad/s</td><td>{signed(row.meanReturn, 1)}</td><td>{(row.successRate * 100).toFixed(0)}%</td></tr>)}</tbody>
          </table>
        </div>
        <p className="lab-note">The hand-written controller is a comparison, not a part of the neural policy. Lower fuel or spin alone does not imply better flight: a motionless rocket can score well on those measures.</p>
        <p className="lab-note">Settled means a final two-second average below 0.45 m distance and 0.8 m/s speed. After an escape, remaining steps count as large tracking error and zero engine use.</p>
        <a className="rocket-data-link" href={modelUrl} download>Download the weights and experiment data ↗</a>
      </div>
    </section>
  );
}

export default function RocketLab() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const stateRef = useRef(initialFlight());
  const targetRef = useRef<Target>({ ...defaultTarget });
  const keysRef = useRef(new Set<string>());
  const modelRef = useRef<PolicyBundle | null>(null);
  const secondsRef = useRef(0);
  const stepRef = useRef({ action: { thrust: 0, torque: 0 }, reward: 0, terms: { ...emptyTerms } });
  const trailRef = useRef<Target[]>([]);
  const activeRef = useRef(true);
  const geometryRef = useRef({ width: 1, height: 1, scale: 1 });
  const [bundle, setBundle] = useState<PolicyBundle | null>(null);
  const [loadError, setLoadError] = useState(false);
  const [loadAttempt, setLoadAttempt] = useState(0);
  const [mode, setMode] = useState<Mode>("ai");
  const [checkpoint, setCheckpoint] = useState("final");
  const [targetMode, setTargetMode] = useState<TargetMode>("pointer");
  const [paused, setPaused] = useState(false);
  const [showTrail, setShowTrail] = useState(true);
  const [notice, setNotice] = useState("Move your pointer across the flight area, or drag the target on a touchscreen.");
  const [telemetry, setTelemetry] = useState<Telemetry>({ state: initialFlight(), action: { thrust: 0, torque: 0 }, target: defaultTarget, distance: Math.hypot(3, 1), reward: 0, terms: emptyTerms, seconds: 0 });
  const settingsRef = useRef({ mode, checkpoint, targetMode, paused, showTrail });
  const controlId = useId();
  const config = bundle?.config ?? DEFAULT_CONFIG;
  const selectedPolicy = bundle?.checkpoints.find(policy => policy.id === checkpoint);
  const obs = observation(telemetry.state, telemetry.target, config);

  useEffect(() => { settingsRef.current = { mode, checkpoint, targetMode, paused, showTrail }; }, [mode, checkpoint, targetMode, paused, showTrail]);

  useEffect(() => {
    const controller = new AbortController();
    setLoadError(false);
    fetch(modelUrl, { signal: controller.signal })
      .then(response => { if (!response.ok) throw new Error("Policy unavailable"); return response.json(); })
      .then(data => {
        validateBundle(data);
        modelRef.current = data;
        setBundle(data);
        setCheckpoint(data.checkpoints.some(policy => policy.id === "final") ? "final" : data.checkpoints.at(-1)!.id);
      })
      .catch(error => { if (error.name !== "AbortError") setLoadError(true); });
    return () => controller.abort();
  }, [loadAttempt]);

  useEffect(() => {
    if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) setPaused(true);
    const clearKeys = () => keysRef.current.clear();
    window.addEventListener("blur", clearKeys);
    return () => window.removeEventListener("blur", clearKeys);
  }, []);

  useEffect(() => {
    const canvas = canvasRef.current;
    const context = canvas?.getContext("2d");
    if (!canvas || !context) return;
    let frame = 0;
    let lastTime = performance.now();
    let accumulator = 0;
    let lastReadout = 0;
    // Canvas colors are read from the same palette as the surrounding controls.
    const styles = getComputedStyle(canvas.closest(".lab-demo")!);
    const cyan = `rgb(${styles.getPropertyValue("--lab-cyan-rgb").trim() || "125, 211, 252"})`;
    const lilac = `rgb(${styles.getPropertyValue("--lab-lilac-rgb").trim() || "196, 181, 253"})`;

    function resize() {
      const bounds = canvas!.getBoundingClientRect();
      const ratio = Math.min(window.devicePixelRatio || 1, 2);
      canvas!.width = Math.max(1, Math.round(bounds.width * ratio));
      canvas!.height = Math.max(1, Math.round(bounds.height * ratio));
      context!.setTransform(ratio, 0, 0, ratio, 0, 0);
      geometryRef.current = { width: bounds.width, height: bounds.height, scale: Math.max(1, Math.min((bounds.width - 40) / 6.4, (bounds.height - 40) / 4.8)) };
    }

    function draw(now: number) {
      const ctx = context!;
      const { width, height, scale } = geometryRef.current;
      const state = stateRef.current;
      const target = targetRef.current;
      const action = stepRef.current.action;
      const sx = (x: number) => width / 2 + x * scale;
      const sy = (y: number) => height / 2 - y * scale;
      ctx.clearRect(0, 0, width, height);
      const backdrop = ctx.createRadialGradient(width * 0.4, height * 0.35, 10, width / 2, height / 2, Math.max(width, height) * 0.7);
      backdrop.addColorStop(0, "rgba(40, 55, 105, 0.35)");
      backdrop.addColorStop(0.6, "rgba(8, 12, 28, 0.75)");
      backdrop.addColorStop(1, "rgba(2, 4, 12, 1)");
      ctx.fillStyle = backdrop;
      ctx.fillRect(0, 0, width, height);
      ctx.lineWidth = 1;
      ctx.strokeStyle = "rgba(170, 190, 255, 0.065)";
      for (let x = -3; x <= 3; x += 0.5) { ctx.beginPath(); ctx.moveTo(sx(x), 0); ctx.lineTo(sx(x), height); ctx.stroke(); }
      for (let y = -2; y <= 2; y += 0.5) { ctx.beginPath(); ctx.moveTo(0, sy(y)); ctx.lineTo(width, sy(y)); ctx.stroke(); }
      ctx.fillStyle = "rgba(220, 230, 255, 0.27)";
      for (let i = 0; i < 42; i++) { const x = ((i * 167 + 83) % 997) / 997 * width; const y = ((i * 281 + 57) % 991) / 991 * height; ctx.beginPath(); ctx.arc(x, y, i % 3 === 0 ? 1 : 0.65, 0, Math.PI * 2); ctx.fill(); }

      if (settingsRef.current.showTrail && trailRef.current.length > 1) {
        ctx.beginPath();
        trailRef.current.forEach((point, i) => i ? ctx.lineTo(sx(point.x), sy(point.y)) : ctx.moveTo(sx(point.x), sy(point.y)));
        ctx.strokeStyle = "rgba(125, 211, 252, 0.25)";
        ctx.lineWidth = 2;
        ctx.stroke();
      }
      ctx.setLineDash([4, 6]);
      ctx.strokeStyle = "rgba(196, 181, 253, 0.28)";
      ctx.beginPath(); ctx.moveTo(sx(state.x), sy(state.y)); ctx.lineTo(sx(target.x), sy(target.y)); ctx.stroke();
      ctx.setLineDash([]);

      const tx = sx(target.x), ty = sy(target.y);
      ctx.strokeStyle = lilac;
      ctx.lineWidth = 1.5;
      ctx.beginPath(); ctx.arc(tx, ty, 13, 0, Math.PI * 2); ctx.stroke();
      ctx.strokeStyle = "rgba(196, 181, 253, 0.15)";
      ctx.beginPath(); ctx.arc(tx, ty, 23, 0, Math.PI * 2); ctx.stroke();
      ctx.fillStyle = lilac;
      ctx.beginPath(); ctx.arc(tx, ty, 3, 0, Math.PI * 2); ctx.fill();
      ctx.font = "11px ui-monospace, monospace";
      ctx.fillStyle = "rgba(220, 230, 255, 0.6)";
      ctx.fillText("TARGET", clamp(tx + 22, 8, width - 65), clamp(ty - 18, 38, height - 8));

      const rx = sx(state.x), ry = sy(state.y);
      // A small off-screen arrow keeps the rocket locatable before an escape reset.
      if (rx < 16 || rx > width - 16 || ry < 16 || ry > height - 16) {
        ctx.save(); ctx.translate(clamp(rx, 18, width - 18), clamp(ry, 48, height - 18)); ctx.rotate(Math.atan2(ry - height / 2, rx - width / 2));
        ctx.fillStyle = cyan; ctx.beginPath(); ctx.moveTo(8, 0); ctx.lineTo(-5, -5); ctx.lineTo(-5, 5); ctx.closePath(); ctx.fill(); ctx.restore();
      }
      ctx.save(); ctx.translate(rx, ry); ctx.rotate(-state.angle);
      if (action.thrust > 0.02 && !settingsRef.current.paused) {
        const flame = 8 + action.thrust * 30 + Math.sin(now * 0.024) * action.thrust * 3;
        const glow = ctx.createLinearGradient(-12, 0, -12 - flame, 0);
        glow.addColorStop(0, "rgba(240, 235, 255, 0.95)"); glow.addColorStop(0.3, "rgba(125, 211, 252, 0.75)"); glow.addColorStop(1, "rgba(196, 181, 253, 0)");
        ctx.fillStyle = glow; ctx.beginPath(); ctx.moveTo(-11, -5); ctx.quadraticCurveTo(-20, -9, -11 - flame, 0); ctx.quadraticCurveTo(-20, 9, -11, 5); ctx.closePath(); ctx.fill();
      }
      ctx.shadowColor = "rgba(125, 211, 252, 0.5)"; ctx.shadowBlur = 14;
      ctx.fillStyle = "#dce6ff"; ctx.strokeStyle = "rgba(255, 255, 255, 0.85)"; ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(18, 0); ctx.quadraticCurveTo(5, -8, -11, -6); ctx.lineTo(-9, 0); ctx.lineTo(-11, 6); ctx.quadraticCurveTo(5, 8, 18, 0); ctx.fill(); ctx.stroke();
      ctx.shadowBlur = 0;
      ctx.fillStyle = lilac; ctx.beginPath(); ctx.moveTo(-3, -5); ctx.lineTo(-12, -13); ctx.lineTo(-10, -3); ctx.closePath(); ctx.fill(); ctx.beginPath(); ctx.moveTo(-3, 5); ctx.lineTo(-12, 13); ctx.lineTo(-10, 3); ctx.closePath(); ctx.fill();
      ctx.fillStyle = "#263d68"; ctx.beginPath(); ctx.ellipse(7, 0, 3.5, 3, 0, 0, Math.PI * 2); ctx.fill();
      ctx.restore();
      ctx.fillStyle = "rgba(220, 230, 255, 0.4)"; ctx.font = "11px ui-monospace, monospace";
      const scaleBottom = settingsRef.current.mode === "manual" ? 76 : 16;
      ctx.fillText("1 m", 18, height - scaleBottom); ctx.strokeStyle = "rgba(220, 230, 255, 0.32)"; ctx.beginPath(); ctx.moveTo(18, height - scaleBottom - 16); ctx.lineTo(18 + scale, height - scaleBottom - 16); ctx.stroke();
    }

    function animate(now: number) {
      const elapsed = Math.min((now - lastTime) / 1000, 0.1);
      lastTime = now;
      const settings = settingsRef.current;
      const model = modelRef.current;
      const flightConfig = model?.config ?? DEFAULT_CONFIG;
      const policy = model?.checkpoints.find(saved => saved.id === settings.checkpoint);
      if (!settings.paused && activeRef.current && !document.hidden && (settings.mode !== "ai" || policy)) {
        accumulator += elapsed;
        while (accumulator >= flightConfig.dt) {
          accumulator -= flightConfig.dt;
          secondsRef.current += flightConfig.dt;
          const t = secondsRef.current;
          if (settings.targetMode === "orbit") targetRef.current = { x: 1.65 * Math.cos(t * 0.4), y: 1.1 * Math.sin(t * 0.4) };
          if (settings.targetMode === "eight") targetRef.current = { x: 1.65 * Math.sin(t * 0.4), y: 1.1 * Math.sin(t * 0.8) };
          const action = settings.mode === "ai" && policy ? neuralAction(stateRef.current, targetRef.current, policy, flightConfig)
            : settings.mode === "heuristic" ? heuristicAction(stateRef.current, targetRef.current, flightConfig)
            : { thrust: keysRef.current.has("w") || keysRef.current.has("arrowup") ? 1 : 0, torque: (keysRef.current.has("a") || keysRef.current.has("arrowleft") ? 1 : 0) - (keysRef.current.has("d") || keysRef.current.has("arrowright") ? 1 : 0) };
          const result = advance(stateRef.current, targetRef.current, action, flightConfig);
          stateRef.current = result.state;
          stepRef.current = { action, reward: result.reward, terms: result.terms };
          trailRef.current.push({ x: result.state.x, y: result.state.y });
          if (trailRef.current.length > 260) trailRef.current.shift();
          if (result.escaped) {
            stateRef.current = initialFlight(); trailRef.current = [];
            setNotice("The rocket left the flight zone and was reset. Try another checkpoint, or compare the hand-written controller.");
          }
        }
      } else accumulator = 0;
      if (activeRef.current) {
        draw(now);
        if (now - lastReadout >= 100) {
          lastReadout = now;
          const step = stepRef.current;
          setTelemetry({ state: { ...stateRef.current }, action: { ...step.action }, target: { ...targetRef.current }, distance: Math.hypot(stateRef.current.x - targetRef.current.x, stateRef.current.y - targetRef.current.y), reward: step.reward, terms: { ...step.terms }, seconds: secondsRef.current });
        }
      }
      frame = requestAnimationFrame(animate);
    }
    const observer = new ResizeObserver(resize);
    observer.observe(canvas);
    const visibility = new IntersectionObserver(entries => { activeRef.current = entries[0].isIntersecting; });
    visibility.observe(canvas);
    resize(); frame = requestAnimationFrame(animate);
    return () => { cancelAnimationFrame(frame); observer.disconnect(); visibility.disconnect(); };
  }, []);

  function reset() {
    stateRef.current = initialFlight(); targetRef.current = { ...defaultTarget }; secondsRef.current = 0; trailRef.current = []; keysRef.current.clear();
    stepRef.current = { action: { thrust: 0, torque: 0 }, reward: 0, terms: { ...emptyTerms } };
    setTelemetry({ state: initialFlight(), action: { thrust: 0, torque: 0 }, target: { ...defaultTarget }, distance: Math.hypot(3, 1), reward: 0, terms: { ...emptyTerms }, seconds: 0 });
    setNotice("Flight reset. Move the target, then watch how the controller responds.");
  }

  function moveTarget(event: React.PointerEvent<HTMLCanvasElement>) {
    if (settingsRef.current.targetMode !== "pointer" || (event.pointerType === "touch" && event.buttons === 0)) return;
    const bounds = event.currentTarget.getBoundingClientRect();
    const { width, height, scale } = geometryRef.current;
    targetRef.current = { x: clamp((event.clientX - bounds.left - width / 2) / scale, -3, 3), y: clamp(-(event.clientY - bounds.top - height / 2) / scale, -2.1, 2.1) };
  }

  function handleKey(event: React.KeyboardEvent<HTMLCanvasElement>, pressed: boolean) {
    const key = event.key.toLowerCase();
    if (!["w", "a", "d", "arrowup", "arrowdown", "arrowleft", "arrowright", " "].includes(key)) return;
    event.preventDefault();
    if (key === " ") { if (pressed && !event.repeat) setPaused(value => !value); return; }
    if (mode === "manual") { if (pressed) keysRef.current.add(key); else keysRef.current.delete(key); }
    else if (pressed && targetMode === "pointer") {
      targetRef.current = { x: clamp(targetRef.current.x + (key === "arrowright" ? 0.15 : key === "arrowleft" ? -0.15 : 0), -3, 3), y: clamp(targetRef.current.y + (key === "arrowup" ? 0.15 : key === "arrowdown" ? -0.15 : 0), -2.1, 2.1) };
    }
  }

  function holdButton(key: string, label: string) {
    return <button type="button" className="rocket-choice rocket-flight-key" aria-label={label}
      onPointerDown={event => { event.preventDefault(); event.currentTarget.setPointerCapture(event.pointerId); keysRef.current.add(key); }}
      onPointerUp={() => keysRef.current.delete(key)} onPointerCancel={() => keysRef.current.delete(key)} onLostPointerCapture={() => keysRef.current.delete(key)}
      onKeyDown={event => { if (event.key === " " || event.key === "Enter") { event.preventDefault(); keysRef.current.add(key); } }}
      onKeyUp={event => { if (event.key === " " || event.key === "Enter") keysRef.current.delete(key); }}
      onBlur={() => keysRef.current.delete(key)}>{label}</button>;
  }

  return (
    <section className="lab-demo rocket-lab" aria-label="Rocket reinforcement learning laboratory">
      <div className="lab-stage">
        <div className="rocket-stage-label"><span className="rocket-live-dot" />{paused ? "Flight paused" : mode === "manual" ? "You have control" : mode === "heuristic" ? "Hand-written controller" : selectedPolicy ? selectedPolicy.label : loadError ? "Controller unavailable" : "Loading neural policy…"}</div>
        <canvas ref={canvasRef} className="lab-canvas" tabIndex={0} aria-label="Interactive rocket flight area. Move the pointer or drag to set a target. Arrow keys move the target. In manual mode W applies thrust and A or D turns. Space pauses."
          onPointerDown={event => { event.currentTarget.focus({ preventScroll: true }); event.currentTarget.setPointerCapture(event.pointerId); moveTarget(event); }}
          onPointerMove={moveTarget} onPointerUp={event => { if (event.currentTarget.hasPointerCapture(event.pointerId)) event.currentTarget.releasePointerCapture(event.pointerId); }}
          onKeyDown={event => handleKey(event, true)} onKeyUp={event => handleKey(event, false)} onBlur={() => keysRef.current.clear()} />
        {mode === "manual" && <div className="rocket-manual-controls" aria-label="Manual flight controls">{holdButton("a", "↶ Turn left")}{holdButton("w", "↑ Thrust")}{holdButton("d", "Turn right ↷")}</div>}
      </div>

      {loadError && <div className="rocket-load-error" role="alert">The neural policy could not be loaded. You can still fly manually or use the hand-written controller. <button type="button" className="rocket-choice" onClick={() => setLoadAttempt(attempt => attempt + 1)}>Retry loading</button></div>}
      <div className="rocket-telemetry" aria-label="Live flight readings">
        <div><span>Distance</span><strong>{telemetry.distance.toFixed(2)} <small>m</small></strong></div>
        <div><span>Speed</span><strong>{Math.hypot(telemetry.state.vx, telemetry.state.vy).toFixed(2)} <small>m/s</small></strong></div>
        <div><span>Reward / step</span><strong className={telemetry.reward >= 0 ? "rocket-positive" : undefined}>{signed(telemetry.reward, 3)}</strong></div>
        <div><span>Flight time</span><strong>{telemetry.seconds.toFixed(1)} <small>s</small></strong></div>
      </div>

      <div className="lab-controls">
        <div className="lab-card">
          <p className="lab-eyebrow">Who is flying?</p>
          <div className="rocket-choices" aria-label="Flight controller">
            <button type="button" className="rocket-choice" aria-pressed={mode === "ai"} onClick={() => { setMode("ai"); keysRef.current.clear(); }}>Neural policy</button>
            <button type="button" className="rocket-choice" aria-pressed={mode === "manual"} onClick={() => { setMode("manual"); keysRef.current.clear(); }}>Manual</button>
            <button type="button" className="rocket-choice" aria-pressed={mode === "heuristic"} onClick={() => { setMode("heuristic"); keysRef.current.clear(); }}>Hand-written</button>
          </div>
          <label className="lab-select-label" htmlFor={`${controlId}-checkpoint`}>Training checkpoint
            <select id={`${controlId}-checkpoint`} value={checkpoint} disabled={!bundle || mode !== "ai"} onChange={event => { setCheckpoint(event.currentTarget.value); reset(); }}>
              {bundle ? bundle.checkpoints.map(policy => <option key={policy.id} value={policy.id}>{policy.label} · {stepLabel(policy.steps)} steps</option>) : <option value="final">Loading checkpoints…</option>}
            </select>
          </label>
          <p className="lab-note">{mode === "ai" ? "Switch checkpoints to see what the same network learned at different points in training." : mode === "manual" ? "Focus the flight area. Hold W to thrust and A/D to turn, or hold the flight buttons in the arena." : "A conventional proportional controller aims its engine along a desired acceleration and damps rotation."}</p>
        </div>
        <div className="lab-card">
          <p className="lab-eyebrow">Two controls. That is all.</p>
          <div className="rocket-action-heading"><span>Thrust</span><strong>{telemetry.action.thrust.toFixed(2)}</strong></div>
          <meter className="rocket-thrust-meter" aria-label="Engine thrust" min="0" max="1" value={telemetry.action.thrust} />
          <div className="rocket-action-heading"><span>Rotational torque</span><strong>{signed(telemetry.action.torque)}</strong></div>
          <div className="rocket-torque-meter" role="meter" aria-label="Rotational torque" aria-valuemin={-1} aria-valuemax={1} aria-valuenow={Number(telemetry.action.torque.toFixed(2))}><span style={{ left: `${50 + Math.min(telemetry.action.torque, 0) * 50}%`, width: `${Math.abs(telemetry.action.torque) * 50}%` }} /><i /></div>
          <div className="rocket-meter-ends"><span>Clockwise −1</span><span>+1 Counterclockwise</span></div>
          <p className="lab-note">The rocket accelerates along its nose. Braking means turning the engine against its motion. This is a spacecraft simulation, with no gravity.</p>
        </div>
        <div className="lab-card">
          <p className="lab-eyebrow">Set a challenge</p>
          <label className="lab-select-label" htmlFor={`${controlId}-target`}>Target movement
            <select id={`${controlId}-target`} value={targetMode} onChange={event => setTargetMode(event.currentTarget.value as TargetMode)}><option value="pointer">Follow pointer / touch</option><option value="orbit">Orbit</option><option value="eight">Figure eight</option></select>
          </label>
          <button type="button" className="lab-primary-button" aria-pressed={paused} onClick={() => setPaused(value => !value)}>{paused ? "Resume flight" : "Pause flight"}</button>
          <button type="button" className="lab-secondary-button" onClick={reset}>Reset flight</button>
          <label className="lab-control-row lab-checkbox lab-checkbox--glow"><input type="checkbox" checked={showTrail} onChange={event => setShowTrail(event.currentTarget.checked)} /><span className="lab-checkmark" aria-hidden="true" /><span>Show flight trail</span></label>
          <p className="lab-note">In neural and hand-written modes, arrow keys place the target when the flight area has focus. Space pauses. Every checkpoint starts from the same flight state when selected.</p>
        </div>
      </div>
      <p className="rocket-interaction-message" role="status">{notice}</p>

      <div className="rocket-inspection-grid">
        <div className="lab-card">
          <p className="lab-eyebrow">What does the network know?</p>
          <h3>Seven numbers in, two numbers out.</h3>
          <dl className="rocket-observations">{["Δx / 3", "Δy / 3", "vₓ / 3", "vᵧ / 3", "sin θ", "cos θ", "ω / 6"].map((name, i) => <div key={name}><dt>{name}</dt><dd>{signed(obs[i])}</dd></div>)}</dl>
          <div className="rocket-network-flow" aria-label="Seven inputs, two hidden layers of 32 neurons, two actions"></div>
          <p className="lab-note">Position and velocity are relative to the flight frame. The angle is encoded as sine and cosine so a full turn does not create a numerical jump. The actor runs in your browser; it is not learning from your pointer.</p>
        </div>
        <div className="lab-card">
          <p className="lab-eyebrow">What counts as a good move?</p>
          <h3>The reward, taken apart.</h3>
          <dl className="rocket-reward-terms">{([ ["Moving closer", "progress"], ["Distance penalty", "distance"], ["Staying close", "arrival"], ["Speed near target", "braking"], ["Fuel penalty", "fuel"], ["Turning & spin", "turning"], ...(telemetry.terms.escape ? [["Flight-zone exit", "escape"]] : []) ] as [string, keyof RewardTerms][]).map(([label, key]) => <div key={key}><dt>{label}</dt><dd className={telemetry.terms[key] > 0 ? "rocket-positive" : undefined}>{signed(telemetry.terms[key], 3)}</dd></div>)}</dl>
          <p className="lab-note">These terms sum to the live reward per simulation step. The learning algorithm optimizes reward accumulated over time, so a brief negative reward can still be part of a useful braking maneuver.</p>
        </div>
      </div>
      {bundle && <ExperimentResults bundle={bundle} selected={mode === "ai" ? checkpoint : mode} />}
    </section>
  );
}
