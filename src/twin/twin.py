"""
twin.py
The DIGITAL TWIN of one patient.

Idea: we keep a "cloud" of 300 guesses (called particles) about this patient's body.
Each guess has its own insulin sensitivity, insulin secretion, etc.
  1. START  : the EHR (HbA1c, BMI, years with diabetes) gives a rough first guess.
  2. ADVANCE: every minute every guess is moved forward with the glucose model,
              using the meals the patient logged and the steps from the wearable.
  3. LEARN  : every 5 minutes a CGM reading arrives. Guesses that predicted it well
              are kept, bad guesses are thrown away (this is the "particle filter").
  4. FORECAST: roll all guesses 2 hours into the future. The spread of the guesses
              gives an uncertainty band, and the share of guesses that go above
              180 mg/dL is the spike probability.
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "simulate"))
from glucose_model import step  # noqa: E402  (the same physics as the simulator)

SPIKE_THRESHOLD = 180.0


def prior_centers(ehr):
    """What the EHR tells us BEFORE we see any glucose readings."""
    eag = 28.7 * ehr["hba1c"] - 46.7
    gb = 0.84 * eag * (1 - 0.08 * ehr["metformin"])
    si = 1.2e-3 * np.exp(-0.035 * (ehr["bmi"] - 25)) * np.exp(-0.02 * ehr["years_dx"]) \
         * (1 + 0.10 * ehr["metformin"])
    gamma = 0.045 * np.exp(-0.04 * ehr["years_dx"])
    return gb, si, gamma, 0.012


class PatientTwin:
    def __init__(self, ehr, n_particles=300, seed=0, cgm_sd=8.0):
        self.rng = np.random.default_rng(seed)
        self.n = n_particles
        self.cgm_sd = cgm_sd
        gb0, si0, gm0, p10 = prior_centers(ehr)
        n, r = self.n, self.rng
        # ----- the cloud of guesses about this patient's body -----
        self.params = dict(
            gb=gb0 * np.exp(r.normal(0, 0.08, n)),
            si=si0 * np.exp(r.normal(0, 0.35, n)),
            gamma=gm0 * np.exp(r.normal(0, 0.35, n)),
            p1=np.clip(p10 * np.exp(r.normal(0, 0.2, n)), 0.005, 0.03),
            vg_dl=np.full(n, 1.6 * float(ehr["weight_kg"])),
        )
        z = np.zeros(n)
        self.state = dict(G=self.params["gb"].copy(), X=z.copy(), I=z.copy(), Q1=z.copy(), Q2=z.copy())
        self.ex = 0.0                     # smoothed exercise level (same for all guesses)
        self.weights = np.full(n, 1.0 / n)

    # ---------- 2. ADVANCE one minute ----------
    def advance_minute(self, carbs_mg=0.0, steps_per_min=0.0):
        self.ex += (min(steps_per_min / 100.0, 1.0) - self.ex) / 60.0
        c = np.full(self.n, carbs_mg)
        e = np.full(self.n, self.ex)
        m = np.ones(self.n)
        self.state = step(self.state, self.params, c, e, m)
        self.state["G"] = self.state["G"] + self.rng.normal(0, 0.3, self.n)   # small model uncertainty

    def advance(self, minutes, carbs_mg_first_minute=0.0, steps_per_min=0.0):
        for k in range(minutes):
            self.advance_minute(carbs_mg_first_minute if k == 0 else 0.0, steps_per_min)

    # ---------- 3. LEARN from a CGM reading ----------
    def assimilate(self, cgm):
        if cgm is None or np.isnan(cgm):
            return
        err = (cgm - self.state["G"]) / self.cgm_sd
        logw = np.log(self.weights + 1e-300) - 0.5 * err ** 2
        logw -= logw.max()
        w = np.exp(logw)
        self.weights = w / w.sum()
        ess = 1.0 / np.sum(self.weights ** 2)
        if ess < self.n / 2:
            self._resample()

    def _resample(self):
        # keep good guesses (several copies), drop bad ones; add a tiny jitter so copies can differ
        positions = (self.rng.random() + np.arange(self.n)) / self.n
        idx = np.searchsorted(np.cumsum(self.weights), positions)
        idx = np.clip(idx, 0, self.n - 1)
        self.params = {k: v[idx].copy() for k, v in self.params.items()}
        self.state = {k: v[idx].copy() for k, v in self.state.items()}
        for k, s in (("si", 0.01), ("gamma", 0.01), ("gb", 0.004), ("p1", 0.01)):
            self.params[k] = self.params[k] * np.exp(self.rng.normal(0, s, self.n))
        self.weights = np.full(self.n, 1.0 / self.n)

    # ---------- 4. FORECAST ----------
    def forecast(self, horizon=120, carbs_mg_now=0.0, steps_per_min=6.0, extra_exercise=None):
        """
        Roll every guess forward `horizon` minutes WITHOUT changing the real twin.
        carbs_mg_now : a meal being eaten right now (mg of carbs)
        extra_exercise: optional (start_min, length_min, steps_per_min) walk, for what-if later
        Returns minutes (every 5), mean glucose, 10th/90th percentile, spike probability.
        """
        st = {k: v.copy() for k, v in self.state.items()}
        ex = self.ex
        c = np.full(self.n, carbs_mg_now)
        peak = st["G"].copy()
        traj = []
        m = np.ones(self.n)
        for t in range(1, horizon + 1):
            spm = steps_per_min
            if extra_exercise is not None and extra_exercise[0] <= t < extra_exercise[0] + extra_exercise[1]:
                spm = extra_exercise[2]
            ex += (min(spm / 100.0, 1.0) - ex) / 60.0
            st = step(st, self.params, c if t == 1 else np.zeros(self.n), np.full(self.n, ex), m)
            peak = np.maximum(peak, st["G"])
            if t % 5 == 0:
                traj.append(st["G"].copy())
        traj = np.array(traj)                             # shape (steps, particles)
        w = self.weights
        mean = traj @ w
        order = np.argsort(traj, axis=1)
        def wq(q):
            out = []
            for row, o in zip(traj, order):
                cw = np.cumsum(w[o])
                out.append(row[o][min(np.searchsorted(cw, q), self.n - 1)])
            return np.array(out)
        return dict(minutes=np.arange(5, horizon + 1, 5), mean=mean, p10=wq(0.10), p90=wq(0.90),
                    spike_prob=float(w[peak > SPIKE_THRESHOLD].sum()))

    def estimate(self):
        """The twin's current best guess of this patient's body parameters."""
        w = self.weights
        return {k: float(np.sum(w * v)) for k, v in self.params.items() if k != "vg_dl"}
