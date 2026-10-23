"""
make_dataset.py
Creates N virtual type-2-diabetes patients and simulates DAYS days for each.
Outputs (in the data/ folder):
    patients.csv  - one row per patient (EHR + hidden true physiology)
    timeseries.csv - one row per patient per 5 minutes (CGM, steps, HR, HRV, meals, sleep)
"""
import os
import sys
import numpy as np
import pandas as pd
from glucose_model import make_params, step, initial_state

N_PATIENTS = int(sys.argv[1]) if len(sys.argv) > 1 else 200
DAYS = int(sys.argv[2]) if len(sys.argv) > 2 else 14
SEED = 42
MIN_PER_DAY = 1440


def make_ehr(n, rng):
    age = np.clip(rng.normal(52, 11, n), 28, 80).round()
    sex = rng.choice(["M", "F"], n)
    bmi = np.clip(rng.normal(27, 4, n), 18.5, 42)
    years_dx = np.clip(rng.gamma(2.0, 3.0, n), 0.5, 25)
    hba1c = np.clip(rng.normal(7.4 + 0.04 * years_dx, 1.0, n), 5.8, 12.0)
    metformin = (rng.random(n) < 0.85).astype(int)
    weight = np.clip(bmi * rng.normal(1.65, 0.07, n) ** 2, 40, 140)
    sbp = np.clip(rng.normal(128 + 0.2 * (age - 50), 12, n), 95, 180).round()
    egfr = np.clip(rng.normal(95 - 0.6 * (age - 40) - 0.5 * years_dx, 10, n), 30, 120).round()
    return dict(age=age, sex=sex, bmi=bmi, years_dx=years_dx, hba1c=hba1c,
                metformin=metformin, weight_kg=weight, sbp=sbp, egfr=egfr)


def make_inputs(n, days, rng):
    """Build minute-by-minute meals, activity and sleep for every patient."""
    T = days * MIN_PER_DAY
    carbs = np.zeros((n, T))            # mg per minute
    steps = np.zeros((n, T))            # steps per minute
    sleeping = np.zeros((n, T), dtype=bool)
    sleep_hours = np.zeros((n, days))
    meal_g = np.zeros((n, T))           # carbs in grams, for the CSV

    for p in range(n):
        wake = rng.normal(6.5, 0.5)     # usual wake-up hour
        for d in range(days):
            base = d * MIN_PER_DAY
            sh = np.clip(rng.normal(6.8, 1.1), 3.5, 9.5)
            sleep_hours[p, d] = sh
            bed_prev = wake * 60 - sh * 60
            s0 = int(base + max(0, bed_prev))        # sleep ends at wake time
            sleeping[p, base: base + int(wake * 60)] = True
            # meals: (hour, mean carbs g)
            for hour, mean_g in [(8.0, 50), (13.0, 80), (20.5, 70)]:
                t = int(base + (hour + rng.normal(0, 0.4)) * 60)
                g = max(15, rng.normal(mean_g, 0.25 * mean_g))
                if 0 <= t < T:
                    carbs[p, t] += g * 1000
                    meal_g[p, t] += g
            if rng.random() < 0.5:                    # afternoon snack
                t = int(base + (16.5 + rng.normal(0, 0.5)) * 60)
                g = max(10, rng.normal(25, 8))
                if 0 <= t < T:
                    carbs[p, t] += g * 1000
                    meal_g[p, t] += g
            # daytime light activity + 1-3 walking bouts
            day_start = base + int(wake * 60)
            day_end = base + 22 * 60
            steps[p, day_start:day_end] = rng.poisson(6, day_end - day_start)
            for _ in range(rng.integers(1, 4)):
                bout = int(day_start + rng.random() * (day_end - day_start - 40))
                length = int(rng.integers(10, 35))
                steps[p, bout: bout + length] += rng.normal(95, 12)
    steps = np.clip(steps, 0, 160)
    return carbs, steps, sleeping, sleep_hours, meal_g


def main():
    rng = np.random.default_rng(SEED)
    n, days = N_PATIENTS, DAYS
    ehr = make_ehr(n, rng)
    ehr["_rng"] = rng
    params = make_params(ehr)
    carbs, steps, sleeping, sleep_hours, meal_g = make_inputs(n, days, rng)

    T = days * MIN_PER_DAY
    state = initial_state(params)
    # smoothed exercise level (exponential average of recent steps), 0..1
    ex = np.zeros(n)
    # daily random factor on insulin sensitivity + effect of last night's sleep
    daily = np.exp(rng.normal(0, 0.10, (n, days)))
    sleep_eff = 1.0 - 0.06 * np.clip(7.0 - sleep_hours, 0, None)   # poor sleep -> -6% per hour missed

    G_true = np.zeros((n, T), dtype=np.float32)
    for t in range(T):
        d = t // MIN_PER_DAY
        hour = (t % MIN_PER_DAY) / 60.0
        ex = ex + (np.clip(steps[:, t] / 100.0, 0, 1) - ex) / 60.0
        # dawn phenomenon: a small morning rise in glucose production
        dawn = 1.0 + 0.04 * np.exp(-((hour - 6.0) ** 2) / 2.0)
        p_now = dict(params)
        p_now["gb"] = params["gb"] * dawn
        state = step(state, p_now, carbs[:, t], ex, daily[:, d] * sleep_eff[:, d])
        G_true[:, t] = state["G"]

    # ----- sample every 5 minutes and add CGM noise, HR, HRV -----
    idx = np.arange(0, T, 5)
    rows = []
    hr_rest = rng.normal(72, 6, n)
    hrv_base = np.clip(rng.normal(38, 8, n), 15, 80)
    for p in range(n):
        g = G_true[p, idx].astype(float)
        # CGM noise is correlated in time (AR(1)), like a real sensor
        noise = np.zeros(len(idx))
        for k in range(1, len(idx)):
            noise[k] = 0.7 * noise[k - 1] + rng.normal(0, 3.0)
        cgm = np.clip(g + noise, 40, 400)
        cgm[rng.random(len(idx)) < 0.01] = np.nan                   # sensor dropouts
        st = steps[p].reshape(-1, 5).sum(axis=1)
        hr = hr_rest[p] + 0.35 * (st / 5.0) + rng.normal(0, 2.5, len(idx))
        dayidx = (idx // MIN_PER_DAY).astype(int)
        hrv = hrv_base[p] * (1 - 0.04 * np.clip(7 - sleep_hours[p, dayidx], 0, None)) \
              + rng.normal(0, 3, len(idx)) - 0.05 * (st / 5.0)
        meal5 = meal_g[p].reshape(-1, 5).sum(axis=1)
        rows.append(pd.DataFrame(dict(
            patient_id=p, minute=idx, day=dayidx, hour=(idx % MIN_PER_DAY) / 60.0,
            glucose_true=g.round(1), cgm=np.round(cgm, 1), steps=st.round(0),
            hr=hr.round(1), hrv=hrv.round(1), carbs_g=meal5.round(0),
            asleep=sleeping[p].reshape(-1, 5).any(axis=1).astype(int))))
    ts = pd.concat(rows, ignore_index=True)

    pat = pd.DataFrame({k: v for k, v in ehr.items() if k != "_rng"})
    pat.insert(0, "patient_id", np.arange(n))
    for k in ("gb", "si", "gamma", "p1"):
        pat["true_" + k] = params[k]
    pat = pat.round(5)

    os.makedirs("data", exist_ok=True)
    pat.to_csv("data/patients.csv", index=False)
    ts.to_csv("data/timeseries.csv", index=False)
    print(f"Done: {n} patients x {days} days.")
    print("patients.csv rows:", len(pat), "| timeseries.csv rows:", len(ts))
    g = ts["glucose_true"]
    print(f"Glucose: mean {g.mean():.0f}, 5th pct {g.quantile(.05):.0f}, 95th pct {g.quantile(.95):.0f}, max {g.max():.0f}")
    print(f"Time above 180 mg/dL: {(g > 180).mean() * 100:.1f}% | below 70: {(g < 70).mean() * 100:.2f}%")


if __name__ == "__main__":
    main()
