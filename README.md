# FPGA-Accelerated Prognostics of Lithium-ion Batteries
⚠️ LEGAL NOTICE: PROPRIETARY CODE This repository contains proprietary research and source code. All rights are reserved. No permission is granted to use, copy, modify, distribute, or sublicense this software for any commercial or academic purpose without explicit written consent from the author.

Here's the comprehensive README.md content tailored for your GitHub project. You can copy this directly into your README.md file in the repository:
Apologies! Here is the detailed README content. You can copy and paste this directly into a README.md file for your GitHub repository.

# FPGA-Accelerated Prognostics of Lithium-ion Batteries

## Project Overview

This project delves into the critical domain of **Lithium-ion battery prognostics**, focusing on the accurate prediction of **State of Health (SOH)** and **Remaining Useful Life (RUL)** for Electric Vehicle (EV) applications. Recognizing the limitations of traditional physics-based models in real-world scenarios, this research proposes and evaluates novel **data-driven machine learning methodologies**, specifically designed for **real-time deployment on Field-Programmable Gate Arrays (FPGAs)**. The ultimate aim is to enhance Battery Management Systems (BMS) with robust, high-performance, and energy-efficient prognostic capabilities.

## Research Motivation and Problem Statement

**Electric Vehicle (EV) adoption** is rapidly increasing, making the performance and longevity of their battery packs paramount. Battery degradation, however, remains a significant challenge. Accurate SOH and RUL predictions are vital for:

*   **Proactive Maintenance**: Scheduling battery replacements before critical failures.
*   **Optimized Charging**: Developing intelligent charging strategies to extend battery lifespan.
*   **Environmental Impact**: Reducing EV battery waste through efficient management.
*   **Enhanced Safety**: Detecting early signs of degradation to prevent hazardous situations.

Traditional physics-based models, while theoretically sound, often struggle with the inherent variability of real-world battery behavior and require extensive, time-consuming parameter tuning. Data-driven approaches offer a promising alternative but face their own set of challenges:

*   **Limited Labeled Datasets**: Scarcity of comprehensive, high-quality battery aging data.
*   **Data Leakage**: Risk of inflated performance metrics due to improper train/test splitting.
*   **Real-time Inference**: The need for computationally efficient models for onboard BMS deployment.

## Proposed Contributions

This project addresses the identified challenges through several key contributions:

1.  **Novel GCDA-LSTM Architecture**: Introduction of a novel Graph Convolutional Network with Dual Attention (GCDA) mechanism integrated into an LSTM framework for advanced battery degradation modeling. This architecture is designed to capture complex temporal and relational dependencies within battery operational data.
2.  **Comprehensive Model Comparison**: A rigorous comparative analysis between **PSO-optimized XGBoost** (a robust tree-based ensemble) and advanced **Deep Learning models** (GCDA-LSTM and Transformer-LSTM) to establish a strong performance benchmark.
3.  **Synthetic Data Augmentation**: Implementation of a 3-parameter synthetic data augmentation technique to enhance dataset diversity and improve model robustness, especially beneficial for scarce real-world data.
4.  **Battery-Level Stratification**: A critical data splitting strategy ensuring **zero data leakage**. Each physical battery is assigned exclusively to either the training, validation, or test set, preventing the models from memorizing battery-specific characteristics.
5.  **FPGA-Ready Quantization**: Development of methods for preparing models for hardware acceleration, specifically focusing on **Q16.16 fixed-point quantization** to facilitate efficient deployment on FPGAs for real-time BMS applications.

## Datasets Used

The study leverages two publicly available and widely recognized battery aging datasets to ensure broad applicability and comparability of results:

1.  **NASA Li-ion Battery Aging Dataset**:
    *   **Source**: NASA Prognostics Center of Excellence (PCoE) data repository.
    *   **Description**: Consists of **34 18650 Li-ion cylindrical cells** with a nominal capacity of **2.0 Ah**. These cells were subjected to various charge/discharge cycles under different ambient temperatures (24°C, 43°C, 4°C) and current profiles.
    *   **Key Data Points**: Cycle number, measured capacity (Ah), ambient temperature, voltage, current.

2.  **Oxford Battery Degradation Dataset 1**:
    *   **Source**: University of Oxford, Howey Research Group.
    *   **Description**: Features **8 Kokam SLPB533459H4 pouch cells** (LCO-NCO chemistry) with a nominal capacity of **0.74 Ah**. The tests were conducted at a constant **40°C** and utilized an Urban Artemis drive cycle profile, with characterization cycles every 100 drive cycles.
    *   **Key Data Points**: Cycle number, measured capacity (Ah), ambient temperature (fixed at 40°C), voltage, charge.

## Methodology & Workflow Details

The project's workflow is systematically structured:

1.  **Environment Setup & Introduction (Cell 1)**:
    *   Initializes the Python environment, installs necessary libraries (`pyswarms`, etc.).
    *   Configures **reproducibility** by setting global random seeds (`RANDOM_SEED = 42`).
    *   Automatically detects and configures the best available compute device (GPU, TPU, or CPU).
    *   Applies **IEEE publication-quality plotting styles** to `matplotlib` and `seaborn` for consistent visualization.
    *   Defines a central `Config` class for all project parameters, including nominal capacities, EOL thresholds, sequence lengths, and hyperparameters for DL and PSO-XGBoost.
    *   Sets up structured logging and output directory creation (`outputs/` folder) for organized experiment tracking.

2.  **Data Loading & Preprocessing (Cell 2)**:
    *   **Raw Data Ingestion**: Parses `.mat` files from both NASA and Oxford datasets, extracting `cycle_num`, `capacity_Ahr`, `ambient_temp`, `battery_id`, `dataset`, and `nominal_capacity`.
    *   **Physics-Based Target Calculation**: Dynamically calculates **SOH (%)** as `(measured_capacity / nominal_capacity) * 100` and **RUL (cycles)** as `max_cycle_for_battery - current_cycle`.
    *   **Data Cleaning**: Handles potential outliers and missing values to ensure data quality.
    *   **Battery-Level Stratification**: Implements a robust split strategy where entire batteries are allocated to either training, validation, or test sets. This is *crucial* to prevent data leakage and ensure generalizability, with a diverse representation of temperature and chemistry in each split.
    *   **Data Integrity Verification**: Explicitly checks for overlaps between splits to confirm zero data leakage.
    *   **Statistical Summaries & Visualizations**: Provides detailed statistics for each split and generates IEEE-style plots (`fig1_degradation_physics.png`, `fig2_statistical_analysis.png`, `fig3_split_analysis.png`, `fig4_rul_distributions.png`) to illustrate data characteristics and split distributions.
    *   **Output**: Saves processed `train.pkl`, `val.pkl`, `test.pkl` dataframes and a `data_integrity_report.json`.

3.  **PSO-XGBoost RUL Prediction (Cell 4)**:
    *   **Objective**: Establish a strong machine learning baseline for RUL prediction.
    *   **Features**: Utilizes `cycle_num`, `ambient_temp`, and `capacity_Ahr` as input features.
    *   **Optimization**: Employs **Particle Swarm Optimization (PSO)** via `pyswarms` to find optimal hyperparameters for `XGBoost` (e.g., `n_estimators`, `learning_rate`, `max_depth`, `gamma`). The objective function minimizes validation MSE.
    *   **Evaluation**: The best model is trained on a combined train+validation set and evaluated on the strictly held-out test set using R², RMSE, and MAE.
    *   **Visualizations**: Generates plots showing actual vs. predicted RUL, residual distribution, and feature importance (`fig5_xgb_results.png`).
    *   **Output**: Saves the trained `xgb_rul_model.json`.

4.  **Deep Learning RUL Prediction (Cell 5)**:
    *   **Objective**: Benchmark advanced deep learning architectures for RUL prediction against the PSO-XGBoost baseline.
    *   **Data Preparation**: Scales features using `MinMaxScaler` and constructs **sequential data** with a `SEQUENCE_LENGTH` of 10, ensuring battery boundaries are respected.
    *   **Models Implemented**:
        *   **GCDA-LSTM**: A two-layer LSTM with a `GraphConvLayer` for inter-feature relationship learning and a `MultiheadAttention` mechanism to weigh important time steps.
        *   **Transformer-LSTM**: Combines a `TransformerEncoder` (with positional embeddings) to capture long-range dependencies, followed by an LSTM layer.
    *   **Training**: Both models are trained using `AdamW` optimizer, `MSELoss` criterion, and `ReduceLROnPlateau` scheduler. Early stopping is implicit via saving the best validation loss model.
    *   **Evaluation**: Performance is reported using R², RMSE, and MAE on the test set. Predictions are inverse-transformed from the normalized scale.
    *   **Visualizations**: Generates convergence plots, actual vs. predicted scatter plots, residual distributions, and a performance comparison bar chart for both DL models, including training times (`fig1_convergence.png`, `fig2_rul_predictions.png`, `fig3_residual_distributions.png`, `fig4_model_metrics_rul.png`, `fig5_training_times.png`).
    *   **Output**: Saves `best_GCDA_LSTM.pth` and `best_Transformer_LSTM.pth` model weights.

5.  **PSO-XGBoost SOH Prediction (Cell 5.1)**:
    *   **Objective**: Establish a strong machine learning baseline for SOH prediction, which is generally considered an easier task than RUL.
    *   **Features**: Uses an enriched feature set including `cycle_num`, `capacity_Ahr`, `ambient_temp`, `capacity_fade`, `capacity_retention`, and `degradation_rate`.
    *   **Optimization**: Similar PSO-based hyperparameter tuning for XGBoost as in RUL prediction.
    *   **Physical Constraint**: Predictions are **clamped** to the `[0, 100]%` range to enforce physical validity.
    *   **Evaluation**: Reports R², RMSE, and MAE on the test set.
    *   **Visualizations**: Generates actual vs. predicted SOH, residual distribution, and feature importance plots (`fig5_1_xgb_soh_results.png`).
    *   **Output**: Saves the trained `xgb_soh_model.json`.

6.  **Deep Learning SOH Prediction (Cell 6)**:
    *   **Objective**: Evaluate GCDA-LSTM and Transformer-LSTM for SOH prediction.
    *   **Features**: Employs the same enhanced feature set as PSO-XGBoost SOH (`capacity_fade`, `capacity_retention`, `degradation_rate` are dynamically generated).
    *   **Data Preparation**: Identical scaling and sequence generation as RUL, but for the SOH target.
    *   **Physical Constraint**: Model outputs are **clamped** to `[0, 100]%` after inverse scaling to ensure physical consistency.
    *   **Evaluation & Visualizations**: Similar to DL RUL, producing convergence, scatter, residual, and metric comparison plots, including training times (`fig7a_soh_convergence.png`, `fig7b_soh_scatter.png`, `fig7c_soh_residual_distributions.png`, `fig7d_soh_model_metrics.png`, `fig7e_soh_training_times.png`).
    *   **Output**: Saves `best_soh_GCDA_LSTM.pth` and `best_soh_Transformer_LSTM.pth` model weights.

7.  **Combined Analysis & Model Comparison (Cell 7)**:
    *   **Objective**: Aggregate and compare the performance of all developed models (PSO-XGBoost, GCDA-LSTM, Transformer-LSTM) against several standard ML baselines (Random Forest, LightGBM, MLP, SVR) for both SOH and RUL tasks.
    *   **Evaluation Metrics**: Calculates R², RMSE, MAE for all models.
    *   **Advanced Visualizations (IEEE-Compliant)**:
        *   **Radar Charts**: Provide a multi-criteria view comparing accuracy (R²), precision (RMSE), and efficiency (training time) for top models across both tasks (`fig7_radar_combined.png`).
        *   **Training Time Plots**: Bar charts illustrating the training duration of each model on a logarithmic scale (`fig7_time_combined.png`).
        *   **Taylor Diagrams**: Visually represent the statistical relationship between the reference (true values) and predicted values, showing correlation, RMS error, and standard deviation in a single plot (`fig7a_taylor_soh.png`, `fig7e_taylor_rul.png`).
        *   **REC Curves (Regression Error Characteristic)**: Plot the CDF of absolute errors, offering a comprehensive view of prediction accuracy across the entire error range (`fig7_rec_combined.png`).
    *   **Output**: Presents structured tables of comparison metrics for SOH and RUL.

8.  **FPGA Deployment & Hardware Implementation (Cell 8)**:
    *   **Objective**: Demonstrate the feasibility of deploying trained models onto FPGAs by generating hardware-ready code.
    *   **Quantization Strategy**: Implements **Q16.16 signed fixed-point quantization**. This format maps floating-point numbers to integers, balancing precision with hardware resource efficiency.
    *   **Model Selection for Deployment**: A simplified (vanilla) LSTM model is used for demonstration, as complex architectures like Transformers often require specialized IP cores (e.g., Vitis AI) that are beyond the scope of this basic HLS example.
    *   **C++ Code Generation**: Automatically generates three essential files for Xilinx Vitis HLS:
        *   `quantized_weights.h`: Contains all model parameters (weights and biases) converted into `fixed_t` (integer) arrays.
        *   `inference.cpp`: Implements the forward pass of the LSTM model using custom fixed-point arithmetic functions (`fp_mul`, `fp_add`, `fp_sigmoid`, `fp_tanh`) designed for hardware synthesis.
        *   `testbench.cpp`: A simple C++ testbench to verify the functionality of `inference.cpp` in a simulation environment.
    *   **Quantization Analysis**: Performs an analysis of the quantization error (average and maximum absolute error) between float and fixed-point representations, confirming minimal precision loss.
    *   **Model Size Calculation**: Estimates the memory footprint of the quantized model, highlighting its suitability for resource-constrained FPGAs.
    *   **Output**: Creates a `fpga_deployment/` directory containing all generated C++ files and a `fpga_deployment_package.zip` archive for easy download.

## Key Results Summary

### RUL Prediction Performance on Test Set

| Model             | R²        | RMSE (Cycles) | MAE (Cycles)  | Training Time (s) |
| :---------------- | :-------- | :------------ | :------------ | :---------------- |
| **Random Forest**   | **0.921** | **376.14**    | **227.84**    | **0.16**          |
| LightGBM          | 0.918     | 384.33        | 240.23        | 0.04              |
| PSO-XGBoost       | 0.891     | 443.34        | 234.13        | 26.90             |
| Transformer-LSTM  | 0.803     | 498.2         | 236.4         | 13.13             |
| GCDA-LSTM         | 0.786     | 520.2         | 245.8         | 8.74              |
| MLP               | 0.268     | 1147.73       | 747.01        | 1.75              |

*   **Observation**: Random Forest surprisingly outperformed other models for RUL, indicating the importance of feature engineering and robust ensemble methods for this prognostic task, especially with shorter sequences. PSO-XGBoost also performed strongly, validating its selection as a robust baseline.

### SOH Prediction Performance on Test Set

| Model             | R²        | RMSE (%)      | MAE (%)       | Training Time (s) |
| :---------------- | :-------- | :------------ | :------------ | :---------------- |
| **Random Forest**   | **0.999** | **0.14**      | **0.06**      | **0.28**          |
| **Transformer-LSTM**| **0.996** | **0.97**      | **0.69**      | **13.13**         |
| PSO-XGBoost       | 0.992     | 0.43          | 0.33          | 45.00             |
| XGBoost           | 0.999     | 0.43          | 0.33          | 0.10              |
| LightGBM          | 0.986     | 1.82          | 0.81          | 0.06              |
| GCDA-LSTM         | 0.928     | 4.02          | 1.56          | 8.74              |
| MLP               | 0.899     | 4.90          | 2.37          | 1.35              |
| SVR               | -0.115    | 16.26         | 11.23         | 0.14              |

*   **Observation**: SOH prediction is highly accurate across several models, particularly Random Forest, Transformer-LSTM, and PSO-XGBoost, all achieving R² values greater than 0.99. This highlights the effectiveness of both tree-based ensembles and deep learning for this state estimation task. The Transformer-LSTM showed excellent balance of performance and efficiency among deep models.

### Hardware Implementation Summary

*   **Quantization Format**: Q16.16 Signed Fixed-Point.
*   **Average Quantization Error**: `~0.000004` (negligible impact on accuracy).
*   **Maximum Quantization Error**: `~0.000015`.
*   **Total Parameters Exported**: `~4658` (for the example LSTM).
*   **Model Size**: `~18.63 KB` (highly optimized for embedded systems).
*   **Generated Artifacts**: `quantized_weights.h`, `inference.cpp`, `testbench.cpp` within the `fpga_deployment` folder, ready for synthesis with Xilinx Vitis HLS.


3.  **Inspect Outputs**: After execution, the `outputs/` directory within your Colab environment will be populated with various generated files, including:
    *   Processed `.pkl` dataframes (`train.pkl`, `val.pkl`, `test.pkl`).
    *   Statistical summaries and data integrity reports.
    *   All generated IEEE-style `.png` figures.
    *   Saved machine learning models (`.json` for XGBoost, `.pth` for PyTorch DL models).
    *   The `fpga_deployment/` directory containing the HLS C++ files (`quantized_weights.h`, `inference.cpp`, `testbench.cpp`), and a convenient `fpga_deployment_package.zip`.

### Dependencies

The Python environment requires the following key libraries (versions may vary slightly based on Colab's environment):

*   `numpy`
*   `pandas`
*   `scipy`
*   `matplotlib`
*   `seaborn`
*   `scikit-learn`
*   `xgboost`
*   `lightgbm`
*   `torch` (`PyTorch`)
*   `pyswarms` (for Particle Swarm Optimization)

These will typically be installed automatically by the notebook cells or are pre-installed in Colab.

## Generated Outputs Overview

The `outputs/` directory serves as the central repository for all artifacts created during the project execution:

*   **Data Files**: `train.pkl`, `val.pkl`, `test.pkl` (Pandas DataFrames for each split).
*   **Reports**: `dataset_statistics.csv`, `data_integrity_report.json` (providing metadata and validation checks).
*   **Visualizations**: Numerous `.png` files, such as `fig1_degradation_physics.png`, `fig2_statistical_analysis.png`, `fig3_split_analysis.png`, `fig4_rul_distributions.png`, `fig5_xgb_results.png`, `fig1_convergence.png` (DL RUL), `fig2_rul_predictions.png` (DL RUL), `fig3_residual_distributions.png` (DL RUL), `fig4_model_metrics_rul.png`, `fig5_training_times.png`, `fig5_1_xgb_soh_results.png`, `fig7a_soh_convergence.png` (DL SOH), `fig7b_soh_scatter.png` (DL SOH), `fig7c_soh_residual_distributions.png` (DL SOH), `fig7d_soh_model_metrics.png`, `fig7e_soh_training_times.png`, `fig7_radar_combined.png`, `fig7_time_combined.png`, `fig7a_taylor_soh.png`, `fig7e_taylor_rul.png`, `fig7_rec_combined.png`.
*   **Trained Models**: `xgb_rul_model.json`, `xgb_soh_model.json` (XGBoost models), `best_GCDA_LSTM.pth`, `best_Transformer_LSTM.pth`, `best_soh_GCDA_LSTM.pth`, `best_soh_Transformer_LSTM.pth` (PyTorch model state dictionaries).
*   **FPGA Deployment Package**: The `fpga_deployment/` directory containing C++ headers (`quantized_weights.h`), source files (`inference.cpp`, `testbench.cpp`), and a `quantization_analysis.json` report, compressed into `fpga_deployment_package.zip`.

## References and Further Reading

This research builds upon established works in battery prognostics and machine learning. Key resources include:

*   **NASA Prognostics Center of Excellence (PCoE) Data Repository**: A primary source for battery aging datasets. [Link to NASA PCoE](https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/)
*   **Oxford Battery Degradation Dataset**: Documented in academic research for battery degradation studies.
    *   Birkl, C.R., 


🔒 License & Copyright
Copyright © 2025 Supriyo Roy. All Rights Reserved.

No License Granted: This repository is visible for portfolio and demonstration purposes only.

Restrictions: You may not clone, fork, download, use, modify, or distribute this code.

Academic Integrity: Usage of this code or methodology in academic papers or theses without permission is prohibited.

For access inquiries or collaboration requests, please contact the author directly.
