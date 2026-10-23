"""
glucose_model.py
The "physics" of our virtual patients: a simple glucose-insulin model.
It is written with numpy arrays so that ONE call advances MANY patients
(or many "particles" later in the twin) by one minute.

State variables (per patient):
    G  = blood glucose (mg/dL)
    X  = how strongly insulin is currently pushing glucose down (1/min)
    I  = insulin level above baseline (uU/mL)
    Q1, Q2 = carbohydrates (mg) travelling through the gut
"""
import numpy as np

# Fixed constants (same for everyone)
P2 = 0.025        # how fast insulin action fades (1/min)
N_CLR = 0.10      # how fast insulin is cleared (1/min)
GUT_TAU = 40.0    # gut absorption time constant (min)
BIOAVAIL = 0.9    # fraction of eaten carbs that reaches blood
H_STEPS = 100.0   # steps/min that count as "full exercise"


def make_params(ehr):
    """
    Turn a patient's EHR numbers (HbA1c, BMI, ...) into physiology parameters.
    ehr: dict of numpy arrays, one entry per patient.
    """
    n = len(ehr["age"])
    rng = ehr["_rng"]
    # Average glucose from HbA1c (standard clinical formula), fasting is lower
    eag = 28.7 * ehr["hba1c"] - 46.7
    gb = 0.84 * eag + rng.normal(0, 4, n)
    gb = np.clip(gb, 85, 220)
    # Insulin sensitivity: worse with higher BMI and longer disease
    si = 1.2e-3 * np.exp(-0.035 * (ehr["bmi"] - 25)) * np.exp(-0.02 * ehr["years_dx"])
    si = si * np.exp(rng.normal(0, 0.15, n))
    # Beta-cell insulin secretion: declines with years since diagnosis
    gamma = 0.045 * np.exp(-0.04 * ehr["years_dx"]) * np.exp(rng.normal(0, 0.15, n))
    # Metformin: lowers liver glucose output (lower Gb) and helps sensitivity a bit
    met = ehr["metformin"].astype(float)
    gb = gb * (1 - 0.08 * met)
    si = si * (1 + 0.10 * met)
    # Glucose effectiveness (how well glucose lowers itself) and distribution volume
    p1 = np.clip(0.012 + rng.normal(0, 0.002, n), 0.006, 0.02)
    vg_dl = 1.6 * ehr["weight_kg"]          # glucose volume in dL
    return dict(gb=gb, si=si, gamma=gamma, p1=p1, vg_dl=vg_dl)


def step(state, params, carbs_mg, exercise, si_mult, dt=1.0):
    """
    Advance every patient by dt minutes.
      carbs_mg : carbs eaten in this minute (mg), array
      exercise : smoothed activity level 0..1, array
      si_mult  : extra multiplier on insulin sensitivity (sleep, daily mood), array
    """
    G, X, I, Q1, Q2 = state["G"], state["X"], state["I"], state["Q1"], state["Q2"]
    gb, si, gamma, p1, vg = (params[k] for k in ("gb", "si", "gamma", "p1", "vg_dl"))

    # Gut: food enters Q1, moves to Q2, then into blood
    Q1 = Q1 + carbs_mg
    dQ1 = -Q1 / GUT_TAU
    dQ2 = Q1 / GUT_TAU - Q2 / GUT_TAU
    ra = BIOAVAIL * Q2 / GUT_TAU / vg                 # mg/dL per min entering blood

    si_eff = si * si_mult * (1.0 + 0.6 * exercise)    # exercise improves sensitivity
    p3 = si_eff * P2

    dG = -p1 * (G - gb) - X * G + ra
    dX = -P2 * X + p3 * I
    dI = -N_CLR * I + gamma * np.maximum(G - gb, 0.0)

    return dict(
        G=np.clip(G + dt * dG, 30.0, 600.0),
        X=np.maximum(X + dt * dX, 0.0),
        I=np.maximum(I + dt * dI, 0.0),
        Q1=np.maximum(Q1 + dt * dQ1, 0.0),
        Q2=np.maximum(Q2 + dt * dQ2, 0.0),
    )


def initial_state(params):
    n = len(params["gb"])
    z = np.zeros(n)
    return dict(G=params["gb"].copy(), X=z.copy(), I=z.copy(), Q1=z.copy(), Q2=z.copy())
