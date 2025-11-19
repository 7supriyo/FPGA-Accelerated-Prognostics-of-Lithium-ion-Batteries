# FPGA-Accelerated Prognostics of Lithium-ion Batteries
⚠️ LEGAL NOTICE: PROPRIETARY CODE This repository contains proprietary research and source code. All rights are reserved. No permission is granted to use, copy, modify, distribute, or sublicense this software for any commercial or academic purpose without explicit written consent from the author.

# 📖 Overview
This project implements a high-precision prognostic framework for estimating the Remaining Useful Life (RUL) and State of Health (SOH) of Lithium-ion batteries.

Currently, the system exists as a high-fidelity software prototype utilizing hybrid deep learning architectures (PSO-XGBoost and Transformer-LSTM). The primary roadmap objective for the upcoming year is the transition of these models onto Field-Programmable Gate Arrays (FPGAs) to achieve microsecond-level latency for embedded Battery Management Systems (BMS).

# 🚀 Project Roadmap
Phase 1: Software Simulation & Validation (Current Status)
Focus: Model architecture design, physics-informed feature engineering, and hyperparameter optimization.

Tech Stack: Python, PyTorch (CUDA), XGBoost, Scikit-Learn.

 # Outcome:

 The models demonstrate high accuracy in predicting battery health metrics:
* **PSO-XGBoost:** Achieved **R² > 0.999** for both RUL and SOH prediction on the augmented test set.
* **Transformer-LSTM:** Showed strong performance in capturing temporal degradation trends with **R² > 0.99**.
* **Baseline Comparisons:** Outperformed standard baselines like Random Forest, KNN, and simple LSTMs in terms of error metrics and inference speed potential.
 <img width="2332" height="2633" alt="rul output" src="https://github.com/user-attachments/assets/6a06b533-186a-48de-aa0a-4e9a06433df4" />
 <img width="2375" height="2976" alt="soh output" src="https://github.com/user-attachments/assets/07686829-a672-4650-b085-a737ad6dd150" />



Phase 2: Hardware Acceleration (Next Year Goal)
Focus: Porting the validated inference engines to FPGA hardware.

Goal: To overcome the power and latency bottlenecks of standard software, enabling real-time prognostics at the network edge.

# Impact:

Eliminating Premature Retirement: Utilizing batteries to their true physical limits.

Circular Economy: Enabling instant, low-energy screening of retired EV batteries for second-life grid storage.

# 📊 Datasets
This research utilizes premier open-source datasets for model training and validation:

NASA PCoE Data Set Repository

Usage: Randomized usage profiles (Random Walk) to simulate unpredictable load conditions.

Link:https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/

Oxford Battery Degradation Dataset

Usage: Long-term degradation data for characterizing aging curves across different thermal conditions.

Link: https://ora.ox.ac.uk/objects/uuid:03ba4b01-cfed-46d3-9b1a-7d4a7bdf6fac

# 🧠 Methodological Approach
The repository (XGBoost+LSTM(CUDA).ipynb) contains the complete end-to-end pipeline:

1. Physics-Informed Data Augmentation
To address data scarcity, a custom augmentation pipeline generates synthetic cycling data based on electrochemical degradation laws (Square Root, Logarithmic, and Polynomial aging factors), significantly expanding the training distribution.

2. Hybrid Modeling Architecture
PSO-XGBoost: Utilizes Particle Swarm Optimization to auto-tune XGBoost regressors, capturing non-linear relationships between voltage/current features and SOH.

Transformer-LSTM: Leveraging the attention mechanism for global context and LSTMs for sequential temporal dependencies, providing robust RUL trajectory prediction.

# 📈 Key Results
The proposed hybrid approach outperforms traditional baselines (Random Forest, SVR, Vanilla LSTM) in the following metrics:

Accuracy: Superior R² and lower RMSE scores on unseen validation data.

Robustness: Stable predictions even under noise-injected scenarios simulating real-world sensor error.

🔒 License & Copyright
Copyright © 2025 Supriyo Roy. All Rights Reserved.

No License Granted: This repository is visible for portfolio and demonstration purposes only.

Restrictions: You may not clone, fork, download, use, modify, or distribute this code.

Academic Integrity: Usage of this code or methodology in academic papers or theses without permission is prohibited.

For access inquiries or collaboration requests, please contact the author directly.
