Audio Artifact Detection Project

Description:
This project uses a CNN model to detect compression artifacts in audio signals.

Pipeline:
1. Extract MFCC features from audio
2. Build dataset (.npy)
3. Train CNN model
4. Evaluate performance
5. Predict audio type (compressed or original)

How to run:
1. python3 extract_mfcc.py
2. python3 build_dataset.py
3. python3 train.py
4. python3 predict.py

Note:
Dataset is small, so accuracy may not reflect real-world performance.
