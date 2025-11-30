# FPGA-Accelerated Prognostics of Lithium-ion Batteries
⚠️ LEGAL NOTICE: PROPRIETARY CODE This repository contains proprietary research and source code. All rights are reserved. No permission is granted to use, copy, modify, distribute, or sublicense this software for any commercial or academic purpose without explicit written consent from the author.

Here's the comprehensive README.md content tailored for your GitHub project. You can copy this directly into your README.md file in the repository:

# FPGA-Accelerated Prognostics of Lithium-ion Batteries

## Project Overview

This project focuses on developing advanced machine learning models for accurate State of Health (SOH) and Remaining Useful Life (RUL) prediction of Electric Vehicle (EV) Lithium-ion batteries. The goal is to create robust, data-driven solutions capable of real-time deployment on embedded systems, specifically FPGAs, for enhanced battery management systems (BMS).

## Research Motivation

Accurate SOH and RUL predictions are crucial for proactive battery maintenance, optimizing charging strategies, reducing battery waste, and enhancing safety in EVs. This research addresses the limitations of traditional physics-based models by proposing novel data-driven approaches that are robust to real-world variability and suitable for hardware acceleration.

## Proposed Contributions

1.  **Novel GCDA-LSTM Architecture**: Combining Graph Convolutional Networks with Dual Attention mechanisms for battery degradation modeling.
2.  **Comprehensive Comparison**: Benchmarking PSO-optimized XGBoost against Deep Learning models (GCDA-LSTM, Transformer-LSTM).
3.  **Data Augmentation**: Implementing 3-parameter synthetic data augmentation for robust training.
4.  **Battery-Level Stratification**: Ensuring zero data leakage during train/validation/test splits.
5.  **FPGA-Ready Quantization**: Preparing models for real-time BMS deployment using Q16.16 fixed-point quantization.

## Datasets

The study utilizes two prominent battery aging datasets:

*   **NASA Li-ion Battery Aging Dataset**: Comprising 34 18650 Li-ion cells with a nominal capacity of 2.0 Ah, tested under various temperature and discharge conditions.
*   **Oxford Battery Degradation Dataset 1**: Featuring 8 Kokam SLPB533459H4 pouch cells (LCO-NCO chemistry) with a nominal capacity of 0.74 Ah, tested at 40°C under urban Artemis drive cycles.

## Methodology & Workflow

The project follows a structured methodology:

1.  **Environment Setup**: Configuration of libraries, reproducibility settings, device detection (GPU/CPU/TPU), IEEE plotting standards, and project constants.
2.  **Data Loading & Preprocessing**: Loading raw `.mat` files, extracting relevant features (capacity, cycle number, temperature), calculating SOH and RUL based on physics principles, and performing rigorous battery-level train/validation/test splitting to prevent data leakage.
3.  **Feature Engineering**: Creation of derived features like `capacity_fade`, `capacity_retention`, and `degradation_rate` to enhance model performance.
4.  **ML Baseline (PSO-XGBoost)**:
    *   **RUL Prediction**: Particle Swarm Optimization (PSO) is used to tune XGBoost hyperparameters for Remaining Useful Life prediction. Features include `cycle_num`, `ambient_temp`, and `capacity_Ahr`.
    *   **SOH Prediction**: Similarly, PSO-optimized XGBoost is applied for State of Health prediction, leveraging enhanced features.
5.  **Deep Learning Models**: Implementation and training of two advanced sequence models:
    *   **GCDA-LSTM**: A novel architecture incorporating Graph Convolutional layers and Dual Attention with LSTM for capturing temporal and relational dependencies.
    *   **Transformer-LSTM**: Combining Transformer Encoder layers with LSTM for robust sequence modeling.
6.  **Comparative Analysis**: Evaluation of all models (baselines, PSO-optimized, and Deep Learning) using standard metrics (R², RMSE, MAE) and visualization of performance, convergence, residual distributions, and training times.
7.  **FPGA Deployment**: Exporting trained model weights (e.g., a simple LSTM) to Q16.16 fixed-point C++ header files, generating HLS-compatible inference code, and conducting quantization error analysis for hardware implementation.

## Key Results Summary

### RUL Prediction Performance (on Test Set)

| Model          | R²      | RMSE (Cycles) | MAE (Cycles) |
| :------------- | :------ | :------------ | :----------- |
| PSO-XGBoost    | 0.8907  | 443.34        | 234.13       |
| GCDA-LSTM      | 0.7856  | 520.2         | 245.8        |
| Transformer-LSTM | 0.8033  | 498.2         | 236.4        |

*(Note: PSO-XGBoost demonstrated superior RUL prediction among the listed models.)*

### SOH Prediction Performance (on Test Set)

| Model          | R²      | RMSE (%) | MAE (%) |
| :------------- | :------ | :------- | :------ |
| PSO-XGBoost    | 0.9925  | 1.34     | 0.33    |
| GCDA-LSTM      | 0.9284  | 4.02     | 1.56    |
| Transformer-LSTM | 0.9959  | 0.97     | 0.69    |

*(Note: Both PSO-XGBoost and Transformer-LSTM achieved excellent SOH prediction with R² > 0.99, with Transformer-LSTM showing slightly better overall accuracy.)*

### Hardware Implementation

An example LSTM model was prepared for FPGA deployment, including:

*   **Q16.16 Fixed-Point Quantization**: Conversion of floating-point weights to integer representation for hardware efficiency. Average quantization error was approximately `0.000004`.
*   **C++ HLS Code Generation**: Creation of `quantized_weights.h`, `inference.cpp` (with fixed-point arithmetic functions and LSTM cell logic), and `testbench.cpp` for 
3.  **Review Outputs**: Check the `outputs/` directory for generated plots, saved models (`.json`, `.pth`), data splits (`.pkl`), and FPGA deployment files (`.h`, `.cpp`, `.zip`).

### Dependencies

Key Python libraries required include:

*   `numpy`
*   `pandas`
*   `scipy`
*   `matplotlib`
*   `seaborn`
*   `scikit-learn`
*   `xgboost`
*   `lightgbm`
*   `torch`
*   `pyswarms`

*(`pyswarms` might require `pip install pyswarms` if not already present.)*

## Generated Outputs

The `outputs/` directory will contain:

*   `train.pkl`, `val.pkl`, `test.pkl`: Processed data splits.
*   `dataset_statistics.csv`: Statistical summary of the dataset.
*   `data_integrity_report.json`: Details on data splitting and leakage verification.
*   `fig*.png`: Various IEEE-style plots for data analysis, model performance, and comparisons.
*   `xgb_rul_model.json`, `xgb_soh_model.json`: Saved PSO-XGBoost models.
*   `best_GCDA_LSTM.pth`, `best_Transformer_LSTM.pth`, `best_soh_GCDA_LSTM.pth`, `best_soh_Transformer_LSTM.pth`: Saved Deep Learning model weights.
*   `fpga_deployment/`: Directory containing C++ headers and source files for FPGA implementation.
*   `fpga_deployment_package.zip`: Compressed archive of FPGA deployment artifacts.

## References

This research is intended to contribute to the field of battery prognostics, building upon and extending works such as:

*   NASA Prognostics Center of Excellence Battery Data Set. (Available at: https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/)
*   Birkl, C.R., "Diagnosis and Prognosis of Degradation in Lithium-Ion Batteries", PhD thesis, University of Oxford, 2017.


🔒 License & Copyright
Copyright © 2025 Supriyo Roy. All Rights Reserved.

No License Granted: This repository is visible for portfolio and demonstration purposes only.

Restrictions: You may not clone, fork, download, use, modify, or distribute this code.

Academic Integrity: Usage of this code or methodology in academic papers or theses without permission is prohibited.

For access inquiries or collaboration requests, please contact the author directly.
