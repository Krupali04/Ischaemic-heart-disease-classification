# Ischemic Heart Disease (IHD) Diagnosis

This project implements a machine learning system for detecting Ischemic Heart Disease using ECG data.

## Project Structure
```
IHD_Diagnosis/
├── data/
│   ├── RECORDS-ISCHEMIA.csv  # Labels file
│   ├── *.mat                 # ECG data files
├── src/
│   ├── data_preprocessing.py # Data preprocessing utilities
│   ├── models.py            # ML model implementations
│   ├── evaluation.py        # Model evaluation metrics
│   └── main.py             # Main execution script
```

## Setup
1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Place your ECG .mat files in the `data/` directory
3. Place the RECORDS-ISCHEMIA.csv file in the `data/` directory

## Usage
Run the main script:
```bash
python src/main.py
```

## Data Format
- ECG data should be in .mat format
- Labels should be provided in RECORDS-ISCHEMIA.csv
