# 🛡️ SAVE: Session-Aware Velocity Engine

> An adaptive, real-time fraud risk scoring engine for digital money transactions — built with Streamlit, scikit-learn, and Plotly.

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-FF4B4B?logo=streamlit&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-F7931E?logo=scikit-learn&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 📖 Overview

**SAVE Engine** is an interactive fraud-detection simulator that demonstrates how adaptive risk scoring works in real-time payment systems. It combines a **machine-learning model** (Random Forest trained on the [PaySim](https://www.kaggle.com/datasets/ealaxi/paysim1) dataset) with **rule-based heuristics** to produce a single _friction score_ for every transaction — determining whether it is **auto-approved**, requires **step-up authentication**, or is **hard-blocked**.

The system is persona-driven: each simulated user has unique behavioral baselines, and the engine adapts its risk thresholds accordingly.

---

## ✨ Key Features

| Feature | Description |
| :--- | :--- |
| **Live Risk Monitor** | Real-time friction gauge, signal breakdown bars, and waterfall chart for every transaction |
| **5-Signal Scoring** | ML Model probability, Purpose Drift, Preamble Anomaly, Velocity Penalty, and Amount Risk |
| **Persona System** | 6 configurable user personas with distinct spending profiles, platforms, and wallet balances |
| **Attack Simulation** | Toggle Session Hijack, CNP Fraud, Credential Stuffing, or Account Drain to see scores spike |
| **Population Benchmark** | Run 1,000+ synthetic transactions and view Precision, Recall, Confusion Matrix, and Economic Impact |
| **Explainability** | Plain-English reasoning for every decision, plus a dedicated "How Attacks Work" educational tab |
| **Expert Mode** | Live weight-tuning sliders for all five risk signals |

---

## 🏗️ Architecture

```
Adaptive-Risk-Engine-for-Money-Transactions/
├── app.py                      # Streamlit UI — tabs, sidebar, visualizations
├── requirements.txt            # Python dependencies
│
├── logic/                      # Core risk-scoring engine
│   ├── context.py              # ContextRetriever — purpose drift, preamble gate, amount risk
│   ├── velocity.py             # VelocityEngine — token-bucket rate limiter + daily spend cap
│   ├── slider.py               # FrictionSlider — weighted sigmoid decision matrix
│   └── ml_model.py             # SAVEModel — RandomForest trained on PaySim / synthetic data
│
├── simulation/                 # Benchmarking & evaluation
│   ├── agents.py               # PersonaAgent — stochastic transaction generator
│   ├── benchmarker.py          # SAVESimulator — population-scale trial runner
│   ├── metrics.py              # QualityMetrics — confusion matrix & economic impact (NER)
│   └── feature_importance.py   # Feature importance utilities
│
└── data/                       # Static config & datasets
    ├── personas.json           # 6 user persona definitions (traits, thresholds, wallets)
    ├── weights_config.json     # Signal weights, thresholds, velocity parameters
    └── PS_*.csv                # PaySim Kaggle dataset (not tracked in Git)
```

---

## 🔬 How the Scoring Works

Each transaction passes through a **five-signal pipeline**, then a **sigmoid squash function** produces a final friction score in **[0.2 – 0.9]**:

```
                    ┌─────────────────────┐
                    │    ML Model Score    │ ── RandomForest P(attack)
                    ├─────────────────────┤
                    │    Purpose Drift     │ ── 1 − historical affinity for this tx type
                    ├─────────────────────┤
   Transaction ──►  │  Preamble Anomaly    │ ── Did the user navigate the app first?
                    ├─────────────────────┤
                    │   Velocity Penalty   │ ── Token bucket depletion + daily spend cap
                    ├─────────────────────┤
                    │     Amount Risk      │ ── Tx amount vs. persona's avg daily spend
                    └────────┬────────────┘
                             │
                     Weighted sum → Sigmoid
                             │
                     ┌───────▼───────┐
                     │ Friction Score │
                     └───────┬───────┘
                             │
              ┌──────────────┼──────────────┐
              ▼              ▼              ▼
         < step_up     < hard_block     ≥ hard_block
        AUTO-APPROVE    STEP-UP AUTH      HARD BLOCK
```

Default thresholds (configurable in `weights_config.json`):

| Decision | Threshold |
| :--- | :---: |
| 🟢 Auto-Approve | `< 0.48` |
| 🟡 Step-Up Auth | `0.48 – 0.65` |
| 🔴 Hard Block | `≥ 0.65` |

---

## 👤 Personas

| Persona | Platform | Age Group | Avg Daily Spend | Starting Balance |
| :--- | :---: | :---: | ---: | ---: |
| The Everyday Student | Touch N Go | 18–24 | RM 80 | RM 350 |
| The Functional Professional | MAE | 28–40 | RM 350 | RM 2,500 |
| The Cautious Adopter | Touch N Go | 50+ | RM 60 | RM 800 |
| The Digital Freelancer | MAE | 25–35 | RM 500 | RM 4,000 |
| The Small Biz Owner | MAE | 35–50 | RM 2,000 | RM 12,000 |
| The Visiting Tourist | Touch N Go | 25–45 | RM 800 | RM 3,500 |

---

## ⚠️ Attack Vectors

| Attack | How it Works | Primary Signals Triggered |
| :--- | :--- | :--- |
| 🚨 **Session Hijack** | Attacker takes over an authenticated session (stolen token/XSS) | ML score forced to 0.90 · Preamble = 0.9 |
| 💳 **Card-Not-Present (CNP)** | Stolen card details used from an unknown device | ML score +0.35 boost · Preamble = 0.9 |
| 🤖 **Credential Stuffing** | Bot cycles leaked credentials; jumps straight to payment | ML score +0.25 boost · Preamble = 0.9 |
| 💸 **Account Drain** | Rapid-fire large transfers to empty the wallet | Velocity penalty ×2–3 · Amount Risk spikes |

---

## 🚀 Getting Started

### Prerequisites

- **Python 3.10+**
- *(Optional)* [PaySim dataset](https://www.kaggle.com/datasets/ealaxi/paysim1) — place the CSV in `data/`. Without it, the engine falls back to synthetic training data automatically.

### Installation

```bash
# Clone the repository
git clone https://github.com/JannsenRamos/Adaptive-Risk-Engine-for-Money-Transactions.git
cd Adaptive-Risk-Engine-for-Money-Transactions

# Create and activate a virtual environment
python -m venv venv

# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Run

```bash
streamlit run app.py
```

The app opens at **http://localhost:8501**.

---

## ⚙️ Configuration

All scoring parameters live in [`data/weights_config.json`](data/weights_config.json):

```json
{
  "friction_weights": {
    "w_model":           0.25,
    "w_drift":           0.30,
    "w_preamble":        0.20,
    "w_velocity":        0.15,
    "w_amount":          0.10,
    "sigmoid_sharpness": 3.0
  },
  "thresholds": {
    "hard_block": 0.65,
    "step_up":    0.48
  },
  "velocity": {
    "burst_capacity":        3.0,
    "off_hours_start":       23,
    "off_hours_end":         6,
    "off_hours_penalty":     1.5,
    "daily_spend_multiplier": 3.0
  }
}
```

> Weights can also be tuned **live** via the Expert Mode panel in the sidebar without restarting the app.

---

## 📊 Benchmark & Metrics

The **Benchmark Analysis** tab runs a population-scale simulation and reports:

| Metric | Description |
| :--- | :--- |
| **Precision / Recall** | Classification quality of the friction threshold |
| **Confusion Matrix** | Interactive heatmap (TP, FP, TN, FN) |
| **Net Error Revenue (NER)** | `Fraud Prevented − Churn Cost` economic metric |
| **ROI Ratio** | Return on investment of the fraud engine |
| **Friction Distribution** | Histogram overlay of normal vs. attack scores |
| **ML Feature Importances** | Bar chart of RandomForest feature weights |

---

## 🛠️ Tech Stack

| Layer | Technology |
| :--- | :--- |
| Frontend | Streamlit, Plotly |
| ML Model | scikit-learn (RandomForestClassifier) |
| Data Processing | Pandas, NumPy |
| Training Dataset | [PaySim](https://www.kaggle.com/datasets/ealaxi/paysim1) (Kaggle) |
| Configuration | JSON-driven weights & personas |

---

## 📂 Data Sources

| Source | Details |
| :--- | :--- |
| **PaySim** (Kaggle) | ~6.3M synthetic mobile money transactions with fraud labels; primary training data |
| **Synthetic fallback** | Auto-generated when PaySim CSV is absent — 2,000 normal + 400 attack samples |
| **Personas** | 6 hand-crafted user archetypes reflecting Malaysian e-wallet demographics |

---

## 📄 License

This project is available under the [MIT License](LICENSE).

---

<p align="center">
  Built with ❤️ using Streamlit & scikit-learn
</p>