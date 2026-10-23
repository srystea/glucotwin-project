# GlucoTwin - Digital Twin Challenge 2026 (work in progress)

A personalised digital twin for type 2 diabetes: it fuses EHR data with wearable/CGM
streams, learns each patient's physiology online, and forecasts glucose spikes
2 hours ahead with uncertainty. All data is **synthetic**.

## Status
- [x] Step 2: synthetic patient simulator (glucose-insulin model, meals, steps, sleep, CGM noise)
- [x] Step 3: particle-filter digital twin (learns insulin sensitivity per patient, forecasts with uncertainty)
- [ ] Step 4: ML residual layer, baselines, held-out-patient validation
- [ ] Step 5: clinician dashboard with what-if simulation
- [ ] Step 6: full documentation, limitations, demo video

## Run
```
pip install -r requirements.txt
python src/simulate/make_dataset.py
python src/simulate/plot_check.py
python src/twin/run_twin_demo.py
```

## Limitations
Synthetic data only; not validated on real patients; research prototype, not a medical device.
