# Deep Learning AAT (22IS741)

- `AAT_Report_Tensor_Ops_and_Drowsiness_Detection.docx` / `.pdf`: final report (19 pages)
- `aat1/`: tensor operations notebook (original and executed) and figure scripts
- `aat2/src/`: drowsiness detection code
  - `prepare_eyes.py`: build the eye datasets (needs pyarrow, run with `.venv_pq`)
  - `train_eye_cnn.py`: train the eye-state CNN (final model)
  - `train.py`: whole-face CNN used for the comparison
  - `app.py`: Gradio web app, `realtime.py`: webcam monitor with alarm
- `aat2/outputs/eye_state_cnn.keras`: trained model, `metrics.json`: results
- `report/build_report.py`: rebuilds the report from the figures and results

Run the apps from this folder:

    PYTHONPATH=aat2/src .venv/bin/python aat2/src/app.py
    PYTHONPATH=aat2/src .venv/bin/python aat2/src/realtime.py

## Setup

    python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
    python3 -m venv .venv_pq && .venv_pq/bin/pip install pyarrow pillow numpy

pyarrow is kept in its own environment because it crashes TensorFlow 2.20 on macOS when both are imported.

## Data (not stored in this repository)

- Driver Drowsiness Dataset: https://huggingface.co/datasets/akahana/Driver-Drowsiness-Dataset (parquet files into `aat2/data/`, then `aat2/src/prepare_data.py`)
- Open and closed eyes: https://huggingface.co/datasets/MichalMlodawski/closed-open-eyes (every 6th parquet file into `aat2/eyes/synth/`)
- Real eye photos: https://huggingface.co/datasets/Ezra23/eye_closed_data (`Open/` and `Closed/` into `aat2/eyes/real/`)

Then run `.venv_pq/bin/python aat2/src/prepare_eyes.py` and `.venv/bin/python aat2/src/train_eye_cnn.py`.
