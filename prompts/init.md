Act as researcher, biomedical engineer, sw architect and python developer.
Project focuses of classification of hand movement (opening, opened, closing, closed) using bioelectric signals, comparing linear/statisic models and modesl of NN, and comparing 3 approaches:
- only EEG
- only EMG
- fusion of signals.

1. Process dataset:
- annotations are not needed.
- it is multimodal measurement, so if subj, session, task, run, date is the same - its the same activity measured.
- every file is named subj, session, task, run, date and device. I need to anonymize. Keep Subj number, session, task and run, device if MAXI its EEG, if MINI its EMG.
- remake it to separate folders by sessions.

2. Create jupyter notebook. I need to classify KMI from EEG and detect activity from EMG.
- its multimodal measurement, so try to sync measurements, processing.
- use only biosignals - not IMU or other.

3. Compare accuracy of extraction activity:
- for each measurement evaluate EMG, EEG and fusion of data.
- use only biosignals - not IMU or other.
- I need to compare simple linear/statistic models, and models of NN.

Suggest other things to implement. I need to compare these metrics.

Do not implement, consult all things to do before.