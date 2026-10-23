"""
plot_check.py
Draws 3 virtual patients for 2 days so we can SEE if the curves look realistic.
Saves the picture to docs/virtual_patients.png and also shows it on screen.
"""
import os
import pandas as pd
import matplotlib
if os.environ.get("NO_SHOW"):
    matplotlib.use("Agg")
import matplotlib.pyplot as plt

ts = pd.read_csv("data/timeseries.csv")
pat = pd.read_csv("data/patients.csv")

fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
for ax, pid in zip(axes, [0, 1, 2]):
    d = ts[(ts.patient_id == pid) & (ts.day.isin([3, 4]))]
    x = (d.day - 3) * 24 + d.hour
    ax.plot(x, d.glucose_true, label="true glucose", lw=2)
    ax.plot(x, d.cgm, label="CGM sensor reading", lw=1, alpha=0.7)
    ax.axhline(180, color="red", ls="--", lw=1)
    meals = d[d.carbs_g > 0]
    ax.scatter((meals.day - 3) * 24 + meals.hour, [60] * len(meals), marker="^", color="green", label="meal")
    p = pat[pat.patient_id == pid].iloc[0]
    ax.set_title(f"Patient {pid}: age {p.age:.0f}, BMI {p.bmi:.1f}, HbA1c {p.hba1c:.1f}, "
                 f"{p.years_dx:.0f} yrs diabetes, metformin={int(p.metformin)}")
    ax.set_ylabel("mg/dL")
    ax.legend(loc="upper right", fontsize=8)
axes[-1].set_xlabel("hours (2 days)")
plt.tight_layout()
os.makedirs("docs", exist_ok=True)
plt.savefig("docs/virtual_patients.png", dpi=110)
print("Saved docs/virtual_patients.png")
if not os.environ.get("NO_SHOW"):
    plt.show()
