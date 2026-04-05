# SENTINEL IDS — Web-Based Intrusion Detection System

A full-stack Flask web application for network intrusion detection using LightGBM.
Built for CICIDS 2017 dataset format (PortScan, DDoS, Botnet, Benign).

---

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the app
```bash
python app.py
```

### 3. Open in browser
```
http://localhost:5000
```

---

## Usage

### Train Tab
1. Upload your CICIDS 2017 CSV file (e.g. `Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv`)
2. Click **Start Training**
3. View accuracy, confusion matrix, and feature importance

### Detect Tab
1. Upload any compatible CSV for detection
2. Click **Run Threat Detection**
3. Results are shown in the Analysis tab

### Analysis Tab
- View detected threat distribution
- See top attacker IPs
- Browse sample attack flows

---

## Project Structure
```
ids_system/
├── app.py              # Flask backend (pipeline + API routes)
├── requirements.txt    # Python dependencies
├── templates/
│   └── index.html      # Full dashboard UI (single-page)
├── uploads/            # Uploaded CSV files (auto-created)
└── README.md
```

---

## Dataset Compatibility 
sumber dataset : https://drive.google.com/file/d/1BG70vkXPe70mhn0Z9dE0ibxk4Qa2Kbwk/view?usp=sharing
- CICIDS 2017 (all variants)
- Supports column names: `Tot Fwd Pkts` / `Total Fwd Packets`
- Supports IP columns: `Src IP` / `Source IP`
- Supports flow columns: `Flow Byts/s` / `Flow Bytes/s`

## Model
- **Algorithm**: LightGBM (multiclass)
- **Features**: 80+ network flow features + engineered aggregations
- **Engineered**: `TotalPackets`, `TotalConnPerIP`, `AvgPktPerFlowIP`
