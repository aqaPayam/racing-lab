"""Local clean-lap personal bests and passive ghosts; never expert demonstrations."""

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import numpy as np
from .vehicle import SIMULATOR_VERSION


class PracticeBook:
    def __init__(self, env, directory="results/practice"):
        self.identity = {
            "track": env.track.digest,
            "simulator": SIMULATOR_VERSION,
            "dt": env.dt,
            "driving": asdict(env.driving),
            "reset": env.reset_options,
            "initial_state": env.state().tolist(),
        }
        key = hashlib.sha256(json.dumps(self.identity, sort_keys=True).encode()).hexdigest()[:24]
        self.path = Path(directory) / f"{key}.npz"
        self.best_time = None
        self.states = self.progress = self.times = None
        if self.path.exists():
            with np.load(self.path, allow_pickle=False) as data:
                if json.loads(str(data["identity"])) != self.identity:
                    return
                states, times, progress = data["states"], data["times"], data["progress"]
                if (
                    states.ndim != 2
                    or states.shape[1] != 7
                    or times.shape != (len(states),)
                    or progress.shape != times.shape
                ):
                    raise ValueError("Invalid practice ghost shapes")
                if (
                    len(times) < 2
                    or not all(np.isfinite(a).all() for a in (states, times, progress))
                    or np.any(np.diff(times) <= 0)
                ):
                    raise ValueError("Invalid practice ghost samples")
                self.states, self.times, self.progress = (
                    states.copy(),
                    times.copy(),
                    progress.copy(),
                )
                self.best_time = float(times[-1])

    def consider(self, recorder):
        env = recorder.env
        if not env.info()["success"] or env.collisions or env.offtrack_steps or env.recoveries:
            return False
        time = env.steps * env.dt
        if self.best_time is not None and time >= self.best_time:
            return False
        self.states = np.asarray(recorder.states)
        self.times = np.arange(len(self.states)) * env.dt
        self.progress = np.r_[0, np.cumsum(np.asarray(recorder.features)[:, 0])]
        self.best_time = time
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Atomic replace prevents a partially written best lap after interruption.
        temporary = self.path.with_suffix(".tmp")
        with temporary.open("wb") as f:
            np.savez_compressed(
                f,
                states=self.states,
                times=self.times,
                progress=self.progress,
                identity=json.dumps(self.identity),
            )
        temporary.replace(self.path)
        return True

    def ghost(self, elapsed):
        if self.states is None or not 0 <= elapsed <= self.best_time:
            return None
        index = min(len(self.times) - 2, max(0, int(np.searchsorted(self.times, elapsed) - 1)))
        t = (elapsed - self.times[index]) / (self.times[index + 1] - self.times[index])
        a, b = self.states[index], self.states[index + 1]
        pose = a + (b - a) * t
        pose[2] = a[2] + ((b[2] - a[2] + np.pi) % (2 * np.pi) - np.pi) * t
        return pose

    def delta(self, progress, elapsed):
        if self.progress is None or progress < 0:
            return None
        # First-passage progress avoids interpolating across backwards manoeuvres.
        maximum = np.maximum.accumulate(self.progress)
        values, indices = np.unique(maximum, return_index=True)
        return float(elapsed - np.interp(progress, values, self.times[indices]))


def compare_runs(first, second):
    """Aligned progress/speed/braking curves for two same-track recordings."""
    from .data import load

    a, b = load(first), load(second)
    if a["metadata"]["track_sha256"] != b["metadata"]["track_sha256"]:
        raise ValueError("Compare runs from the same track layout")
    result = {"first": str(first), "second": str(second), "runs": []}
    for run in (a, b):
        progress = np.r_[0, np.cumsum(run["reward_features"][:, 0])]
        result["runs"].append(
            {
                "progress_m": progress.tolist(),
                "time_s": run["timestamps"].tolist(),
                "speed_kmh": (run["states"][:, 3] * 3.6).tolist(),
                "brake": np.r_[0, run["actions"][:, 2]].tolist(),
                "trajectory_xy": run["states"][:, :2].tolist(),
                "simulator_version": run["metadata"]["simulator_version"],
            }
        )
    return result


def comparison_html(report):
    """Standalone offline canvas chart; no external libraries or scripts."""
    payload = json.dumps(report).replace("<", "\\u003c")
    details = ""
    ghost = report.get("lap_comparison")
    if ghost:
        human = (
            f"{ghost['human_lap_seconds']:.2f}s"
            if ghost["human_valid"]
            else "No valid completed lap"
        )
        delta = f"{ghost['human_minus_sac_seconds']:+.2f}s" if ghost["human_valid"] else "Undefined"
        details = (
            f"<h2>Standing-start lap comparison</h2><p>You: <b>{human}</b> · "
            f"Frozen SAC: <b>{ghost['sac_lap_seconds']:.2f}s</b> · You minus SAC: <b>{delta}</b></p>"
            "<p>A negative finish gap means you were faster. Only laps with every gate and zero off-road steps, collisions and recoveries receive a lap time. "
            "Live gaps use first passage at equal progress. Both use the same timestep and vehicle aids; human keyboard/gamepad input shaping differs from SAC. "
            "This is a local practice comparison, with no training updates or claim of independent trials.</p>"
            "<p>Orange: your line · Teal: SAC's line. Inspect braking and speed below.</p>"
        )
    return (
        """<!doctype html><meta charset="utf-8"><title>Racing Lab comparison</title>
<style>body{background:#101923;color:#e8eef2;font:16px system-ui;margin:32px}canvas{display:block;background:#172632;margin:18px 0;width:min(1100px,100%)}b{color:#64d9c0}small{color:#b1c2cf}</style>
<h1>Racing Lab / Run comparison</h1><p id="legend"></p>
<p id="names"></p><small>Speed and brake are plotted against cumulative progress. Trajectories use world metres. Different simulator versions should not be pooled as one experiment.</small>
"""
        + details
        + """
<canvas id="speed" width="1100" height="280"></canvas><canvas id="brake" width="1100" height="220"></canvas><canvas id="path" width="1100" height="650"></canvas>
<script>const report="""
        + payload
        + """;document.getElementById('names').textContent=report.first+' | '+report.second;
const palette=report.lap_comparison?['#ffb477','#64d9c0']:['#64d9c0','#ffb477'];
document.getElementById('legend').textContent=report.lap_comparison?'Orange: you | Teal: SAC':'Teal: first run | Orange: second run';
function chart(id,label,xkey,ykey){const c=document.getElementById(id),g=c.getContext('2d'),w=c.width,h=c.height;
const series=report.runs.map(r=>id==='path'?r.trajectory_xy:r[xkey].map((x,i)=>[x,r[ykey][i]]));
let xmin=Infinity,xmax=-Infinity,ymin=Infinity,ymax=-Infinity;for(const s of series)for(const [x,y]of s){xmin=Math.min(xmin,x);xmax=Math.max(xmax,x);ymin=Math.min(ymin,y);ymax=Math.max(ymax,y)}
if(id!=='path')ymin=Math.min(0,ymin);let sx=(w-100)/Math.max(1,xmax-xmin),sy=(h-65)/Math.max(.01,ymax-ymin);if(id==='path')sx=sy=Math.min(sx,sy);
g.strokeStyle='#3d5060';g.strokeRect(65,35,w-95,h-65);g.fillStyle='#e8eef2';g.font='16px system-ui';g.fillText(label,20,23);g.fillText(ymax.toFixed(1),8,48);g.fillText(ymin.toFixed(1),8,h-32);g.fillText(xmin.toFixed(0),65,h-8);g.fillText(xmax.toFixed(0)+' m',w-85,h-8);
series.forEach((s,j)=>{g.strokeStyle=palette[j];g.lineWidth=2;g.beginPath();s.forEach(([x,y],i)=>{let px=65+(x-xmin)*sx,py=h-30-(y-ymin)*sy;i?g.lineTo(px,py):g.moveTo(px,py)});g.stroke()})}
chart('speed','Speed (km/h) vs progress','progress_m','speed_kmh');chart('brake','Brake input vs progress','progress_m','brake');chart('path','Trajectory comparison');</script>"""
    )


def ghost_environment(track, pose, elapsed):
    from types import SimpleNamespace
    from .vehicle_model import VehicleState, RADIUS

    vehicle = VehicleState.rolling(pose[3])
    vehicle.angle[:] = pose[3] * elapsed / RADIUS
    return SimpleNamespace(
        track=track, pos=pose[:2], heading=pose[2], steering=pose[6], vehicle=vehicle
    )
