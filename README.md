<div align="center">

<img src="https://img.shields.io/badge/model-MLP%20%2B%20PCA-DC2626?style=for-the-badge"/>
<img src="https://img.shields.io/badge/app-Streamlit-DC2626?style=for-the-badge&logo=streamlit&logoColor=white"/>
<img src="https://img.shields.io/badge/domain-Network%20Security-DC2626?style=for-the-badge"/>
<img src="https://img.shields.io/badge/Python-3.10+-DC2626?style=for-the-badge&logo=python&logoColor=white"/>

# 🛡️ Network Traffic Classifier
### Heavy vs. Normal Traffic Detection — MLP + PCA + Streamlit

> *28 packet-level features. One question: is this traffic heavy?*

</div>

---

## 📖 Overview

A machine learning pipeline for classifying network traffic flows as **Heavy** or **Not Heavy** based on packet burst statistics. Built for cybersecurity analytics, the app accepts either a CSV batch upload or manual feature entry — and flags suspicious high-volume flows instantly.

The model pipeline chains a **StandardScaler → PCA → MLP classifier**, all serialized and loaded at runtime for zero-latency inference.

---

## 🧠 Model Pipeline

```
Raw features (28)
      │
      ▼
┌─────────────────────┐
│   StandardScaler    │  → scaler.pkl
└─────────────────────┘
      │
      ▼
┌─────────────────────┐
│        PCA          │  → pca.pkl
└─────────────────────┘
      │
      ▼
┌─────────────────────┐
│   MLP Classifier    │  → mlp_model.pkl
└─────────────────────┘
      │
      ▼
  Heavy Traffic / Not Heavy
```

All three artifacts are serialized with `pickle` and loaded at app startup via `@st.cache_resource`.

---

## 📊 Features — 28 Packet Burst Statistics

| Category | Features |
|---|---|
| **Burst stats** | Burst count, burst byte variation, burst time variation, burst duration variation |
| **Packet timing** | Max / 75th percentile / mean time between packets, avg packet timing |
| **PCA-derived** | Burst start time (×2), packet length analysis, burst interval analysis (×2), burst duration analysis (×2) |
| **Burst bytes** | 10 decile buckets — 0–10% through 80–90% |
| **Packet direction** | Mean direction, direction variation |
| **Traffic direction** | Reverse traffic ratio |

---

## 🚀 App Features

### 📂 CSV Upload mode
Upload a CSV with the 28 required feature columns → the app runs batch inference, appends a `Prediction` column, highlights **Heavy Traffic** rows in red, and lets you download the results.

### ✍️ Manual Entry mode
Enter all 28 feature values individually via number inputs split across two columns → hit **Predict** → instant result with color-coded feedback (🛑 red for heavy, ✅ green for normal).

---

## 🛠️ Stack

| Layer | Tool |
|---|---|
| Model | MLP Classifier (scikit-learn) |
| Preprocessing | StandardScaler + PCA |
| App | Streamlit |
| Data | Pandas, NumPy |
| Serialization | Pickle |

---

## 🚀 Getting Started

```bash
git clone https://github.com/MedAmirSoltani/network-traffic-classifier.git
cd network-traffic-classifier

pip install -r requirements.txt

streamlit run app.py
```

> Make sure `scaler.pkl`, `pca.pkl`, and `mlp_model.pkl` are present in the root directory before running.

---

## 📁 Structure

```
├── app.py                  # Streamlit application
├── scaler.pkl              # Fitted StandardScaler
├── pca.pkl                 # Fitted PCA transformer
├── mlp_model.pkl           # Trained MLP classifier
├── hacker.png              # Sidebar image
├── requirements.txt
└── README.md
```

---

## 👤 Author

**Mohamed Amir Soltani** — AI Engineer & Data Scientist
UTT · PST&B · ESPRIT

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0A66C2?style=flat&logo=linkedin)](https://linkedin.com/in/mohamedamirsoltani)

---

<div align="center">
<sub>Network security · Traffic classification · Machine learning</sub>
</div>
