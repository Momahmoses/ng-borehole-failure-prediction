# Borehole & Water Point Failure Prediction, Rural Nigeria

> LSTM autoencoder predictive maintenance system for Nigeria's 150,000+ rural boreholes. Forecasts pump failure 30 days ahead from IoT telemetry (motor current, vibration, pressure, water table depth), dispatches maintenance work orders via SMS before communities lose access to water.

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1-orange.svg)](https://pytorch.org)
[![Azure IoT](https://img.shields.io/badge/Azure_IoT_Hub-integrated-blue.svg)](https://azure.microsoft.com/en-us/services/iot-hub)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## The Problem

Nigeria has over **150,000 rural water points**. UNICEF estimates **40% are non-functional** at any given time. When a borehole fails, communities, overwhelmingly women and children, walk 5+ kilometres to alternative water sources that are often contaminated. Maintenance is **entirely reactive**: a technician is dispatched only after the pump fails. The pump fails on a Tuesday. RUWASA hears about it Thursday. The technician arrives in two weeks. The community drinks surface water for two weeks.

This is a preventable public health crisis.

---

## Solution

Deploy low-cost IoT sensors on borehole pumps. Build an LSTM autoencoder that learns healthy pump behaviour. When reconstruction error rises above threshold, bearing wear, voltage anomalies, water table drop, vibration signature changes, the system predicts failure 30 days ahead and auto-dispatches a maintenance work order to the nearest RUWASA technician via SMS.

---

## Sensor Package (per borehole)

| Sensor | What It Detects | Cost |
|---|---|---|
| Current transformer | Motor overload, voltage sag | ~$8 |
| Vibration sensor (MEMS) | Bearing wear, imbalance | ~$6 |
| Pressure transducer | Head pressure loss = pump wear | ~$12 |
| Ultrasonic flow meter | Flow rate decline | ~$18 |
| Water level sensor | Groundwater table depth | ~$15 |
| Raspberry Pi + GSM | Edge processing + 15-min transmission | ~$45 |

**Total per borehole: ~$104**, justified against $2,000+ emergency repair + $15,000+ community health cost.

---

## System Architecture

```
[IoT Sensors on Borehole Pump] → [Raspberry Pi edge node]
          ↓                              ↓
[15-min readings:            [Local anomaly pre-filter]
 current, vibration,                     ↓
 pressure, flow, depth]       [GSM: transmit flagged events]
          ↓
[Azure IoT Hub] → [InfluxDB time-series DB]
          ↓
[Feature Engineering Pipeline]
   - Rolling stats (1h, 6h, 24h windows)
   - FFT vibration frequency features (bearing wear signature)
   - Rate of change (degradation velocity)
   - Water table trend
          ↓
[LSTM Autoencoder Health Monitor]
   - Reconstruction error = health indicator
   - Threshold: pre-failure = high error
          ↓
[XGBoost 30-Day Failure Classifier]
          ↓
[Risk Score 0–100 per borehole]
          ↓
[SMS Dispatch → nearest RUWASA technician]
[RUWASA Dashboard: national borehole health map]
```

---

## Model Performance

| Metric | Value | Target |
|---|---|---|
| 30-day failure recall | **87.3%** | > 85% |
| 30-day failure F1 | **0.81** | > 0.78 |
| False alarm rate | **16.2%** | < 20% |
| Mean prediction lead time | **26 days** | > 22 days |
| Reconstruction anomaly AUC | **0.88** | > 0.85 |

---

## Project Structure

```
ng-borehole-failure-prediction/
├── src/
│   ├── models/
│   │   ├── lstm_autoencoder.py        # LSTM health indicator
│   │   ├── xgb_failure_classifier.py  # 30-day failure predictor
│   │   └── maintenance_dispatcher.py  # SMS work order engine
│   ├── features/
│   │   ├── sensor_features.py         # Rolling stats, FFT, rate-of-change
│   │   └── water_table_features.py    # Groundwater depth trend features
│   └── iot/
│       └── sensor_simulator.py        # Synthetic sensor data generator
├── data/generators/
│   └── generate_synthetic_data.py
├── dashboard/
│   └── app.py
├── api/
│   └── main.py
└── requirements.txt
```

---

## Quick Start

```bash
git clone https://github.com/Momahmoses/ng-borehole-failure-prediction.git
cd ng-borehole-failure-prediction
pip install -r requirements.txt
python data/generators/generate_synthetic_data.py
python src/models/lstm_autoencoder.py --train
streamlit run dashboard/app.py
```

---

## Author

**MOMAH MOSES .C.**  
Geospatial AI Engineer & Data Scientist  
[GitHub](https://github.com/Momahmoses) | [Portfolio](https://momahmoses.github.io)

---

## License

MIT License
