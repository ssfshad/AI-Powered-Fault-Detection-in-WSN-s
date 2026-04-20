# 📡 AI-Powered Fault Detection in Wireless Sensor Networks

A machine learning web app that simulates a 20-node wireless sensor network (WSN), detects faults in real time using a Random Forest classifier, and automatically re-routes data around failed nodes — all in your browser.

**Live features:**
- Real-time network monitoring with animated fault detection
- Self-healing network demo (watches AI detect a fault and re-route data)
- Manual sensor reading predictor
- Dataset explorer & model performance charts

---

## 🖥️ What It Looks Like

The app has 6 tabs:

| Tab | What it does |
|-----|-------------|
| 🔴 Live Monitor | Simulates all 20 nodes updating every second with fault detection |
| 🔧 Self-Healing | Step-by-step + animated demo of fault injection → AI detection → re-routing |
| 📊 Performance | Confusion matrix, battery life comparison, detection latency charts |
| 🔍 Predict | Enter sensor values manually or upload a CSV to classify faults |
| 📋 Dataset | Explore the training data, feature distributions, correlations |
| 🤖 Model Info | Algorithm details, feature reference, per-class accuracy |

---

## ⚙️ Requirements

- **Python 3.9 or higher** — [Download Python](https://www.python.org/downloads/)
- That's it. All other dependencies are installed automatically.

> **Not sure if you have Python?**  
> Open a terminal (Command Prompt on Windows, Terminal on Mac/Linux) and type:
> ```
> python --version
> ```
> If you see something like `Python 3.11.2` you're good. If you get an error, download Python from the link above — during installation on Windows, check **"Add Python to PATH"**.

---

## 🚀 How to Run (Step by Step)

### Step 1 — Download the project

If you have Git installed:
```bash
git clone https://github.com/ssfshad/AI-Powered-Fault-Detection-in-WSN-s.git
cd AI-Powered-Fault-Detection-in-WSN-s
```

Or click the green **"Code"** button on GitHub → **"Download ZIP"**, unzip it, and open a terminal inside that folder.

---

### Step 2 — (Recommended) Create a virtual environment

A virtual environment keeps this project's packages separate from the rest of your computer. You only need to do this once.

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**Mac / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

You'll see `(venv)` appear in your terminal — that means it's active.

> **Skip this step?** You can, but it's better practice to use one.

---

### Step 3 — Install dependencies

```bash
pip install -r requirements.txt
```

This installs: `streamlit`, `pandas`, `numpy`, `scikit-learn`, `plotly`.  
It takes about 1–2 minutes on first run.

---

### Step 4 — Run the app

```bash
streamlit run app.py
```

Your browser will open automatically at `http://localhost:8501`.

> **First launch only:** The app will generate the dataset (~5,000 rows) and train the Random Forest model. This takes about 10–20 seconds and only happens once. After that, the trained model is saved to the `model/` folder and reused on every run.

---

## 📁 Project Structure

```
├── app.py              ← Main Streamlit web app
├── generate_data.py    ← Generates the synthetic WSN dataset
├── train_model.py      ← Trains the Random Forest classifier
├── requirements.txt    ← Python dependencies
├── data/
│   └── wsn_dataset.csv ← Auto-generated on first run (5,000 samples)
└── model/
    ├── rf_model.pkl        ← Trained model (auto-saved)
    ├── scaler.pkl          ← Feature scaler
    ├── label_encoder.pkl   ← Class label encoder
    ├── confusion_matrix.npy
    └── test_accuracy.npy
```

> The `data/` and `model/` folders are created automatically on first run — you don't need to create them.

---

## 🔍 What the Model Detects

The Random Forest classifier identifies 5 fault types from 6 sensor readings:

| Fault Type | Description |
|---|---|
| ✅ Normal | All readings within expected operating ranges |
| 💀 Node Failure | Node stops responding — extreme packet loss, erratic readings |
| ⚠️ Data Anomaly | Sensor readings spike beyond physical possibility (e.g., 90°C temp) |
| 🔋 Battery Drain | Battery critically low — node entering degraded mode |
| 📶 Communication Loss | Weak signal causing high packet loss and latency |

**Input features:**

| Feature | Unit | Normal Range |
|---|---|---|
| Temperature | °C | 20 – 35 |
| Humidity | % | 30 – 70 |
| Signal Strength | dBm | -60 to -40 |
| Battery Level | % | 50 – 100 |
| Packet Loss Rate | % | 0 – 5 |
| Response Time | ms | 10 – 50 |

---

## 🧠 Model Details

- **Algorithm:** Random Forest Classifier (scikit-learn)
- **Trees:** 100
- **Max depth:** 15
- **Training data:** 5,000 synthetic samples
- **Test split:** 20% (held out)
- **Accuracy:** ~98–99% on test set

---

## 🛠️ Troubleshooting

**`streamlit: command not found`**  
Make sure your virtual environment is activated and you ran `pip install -r requirements.txt`.

**`ModuleNotFoundError`**  
Run `pip install -r requirements.txt` again inside the activated virtual environment.

**App opens but shows an error about missing files**  
Delete the `data/` and `model/` folders if they exist and restart — the app will regenerate them cleanly.

**Port 8501 already in use**  
Run on a different port:
```bash
streamlit run app.py --server.port 8502
```

---

## 📦 Dependencies

| Package | Version |
|---|---|
| streamlit | ≥ 1.32.0 |
| pandas | ≥ 1.5.0 |
| numpy | ≥ 1.24.0 |
| scikit-learn | ≥ 1.3.0 |
| plotly | ≥ 5.18.0 |

---

Built with Streamlit + Plotly + scikit-learn.
