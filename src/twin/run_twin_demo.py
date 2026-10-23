"""
run_twin_demo.py
Runs the twin on virtual patients and checks: does it actually LEARN each patient and
predict glucose + spikes better than a simple baseline?

  - Days 0-6  : twin only watches and learns (warm-up)
  - Days 7-13 : every 15 minutes we ask "what happens in the next 2 hours?" and compare
                with what REALLY happened in the simulator.
Baseline = "persistence": guess that glucose stays where it is now.
"""
import sys
import time
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from twin import PatientTwin, SPIKE_THRESHOLD

N_PAT = int(sys.argv[1]) if len(sys.argv) > 1 else 12
WARMUP_DAYS = 7
rng = np.random.default_rng(7)

pat = pd.read_csv("data/patients.csv")
ts = pd.read_csv("data/timeseries.csv")
test_ids = list(range(150, 150 + N_PAT))        # patients 150+ (the ML step later will train on the others)

res = []          # one row per forecast
est_rows = []
t0 = time.time()
for pid in test_ids:
    row = pat[pat.patient_id == pid].iloc[0]
    d = ts[ts.patient_id == pid].reset_index(drop=True)
    cgm = d.cgm.values
    gtrue = d.glucose_true.values
    steps_pm = d.steps.values / 5.0
    # the patient LOGS meals, but never exactly right: +/-25% error in carb amount
    logged_mg = d.carbs_g.values * 1000 * np.exp(rng.normal(0, 0.25, len(d)))
    twin = PatientTwin(row, n_particles=300, seed=pid)
    twin.assimilate(cgm[0])
    start_k = WARMUP_DAYS * 288
    for k in range(1, len(d)):
        twin.advance(5, carbs_mg_first_minute=logged_mg[k - 1], steps_per_min=steps_pm[k - 1])
        twin.assimilate(cgm[k])
        if k >= start_k and k % 3 == 0 and k + 24 < len(d):          # every 15 min
            f = twin.forecast(120, carbs_mg_now=logged_mg[k], steps_per_min=steps_pm[k])
            future = gtrue[k + 1: k + 25]
            res.append(dict(pid=pid, cgm_now=cgm[k],
                            y30=future[5], y60=future[11], y120=future[23],
                            p30=f["mean"][5], p60=f["mean"][11], p120=f["mean"][23],
                            spike=int(future.max() > SPIKE_THRESHOLD),
                            twin_prob=f["spike_prob"]))
    e = twin.estimate()
    est_rows.append(dict(pid=pid, est_si=e["si"], true_si=row.true_si, est_gb=e["gb"], true_gb=row.true_gb))
    print(f"patient {pid} done ({time.time() - t0:.0f}s)")

r = pd.DataFrame(res).dropna()
print("\n=========== RESULTS (forecasts on days 8-14, unseen future) ===========")
print(f"forecasts evaluated: {len(r)} from {N_PAT} patients")
print("\nError in mg/dL (RMSE) - lower is better")
print(f"{'horizon':>10} {'persistence':>13} {'digital twin':>14}")
for h in (30, 60, 120):
    rm_b = np.sqrt(np.mean((r.cgm_now - r[f"y{h}"]) ** 2))
    rm_t = np.sqrt(np.mean((r[f"p{h}"] - r[f"y{h}"]) ** 2))
    print(f"{h:>7} min {rm_b:>13.1f} {rm_t:>14.1f}")
print("\nSpike detection (will glucose exceed 180 in next 2h?)")
print(f"  spike rate in data      : {r.spike.mean() * 100:.1f}%")
print(f"  AUROC persistence score : {roc_auc_score(r.spike, r.cgm_now):.3f}")
print(f"  AUROC digital twin prob : {roc_auc_score(r.spike, r.twin_prob):.3f}")
e = pd.DataFrame(est_rows)
print("\nDid the twin LEARN each patient's body? (correlation across patients, 1.0 = perfect)")
print(f"  insulin sensitivity : {np.corrcoef(np.log(e.est_si), np.log(e.true_si))[0, 1]:.2f}")
print(f"  fasting glucose     : {np.corrcoef(e.est_gb, e.true_gb)[0, 1]:.2f}")
low = r[r.cgm_now <= 180]
print("\nHarder, more honest test: only moments when glucose is NOT yet high (<=180 now)")
print(f"  spike rate              : {low.spike.mean() * 100:.1f}%")
print(f"  AUROC persistence score : {roc_auc_score(low.spike, low.cgm_now):.3f}")
print(f"  AUROC digital twin prob : {roc_auc_score(low.spike, low.twin_prob):.3f}")
sure = low[low.twin_prob > 0.95]
print(f"  when twin was >95% sure : spike really happened {sure.spike.mean() * 100:.0f}% of the time ({len(sure)} cases)")
unseen = low[low.twin_prob <= 0.05]
print(f"  when twin said <=5%     : spike still happened {unseen.spike.mean() * 100:.0f}% of the time (meals nobody logged yet)")
r.to_csv("data/twin_eval.csv", index=False)
