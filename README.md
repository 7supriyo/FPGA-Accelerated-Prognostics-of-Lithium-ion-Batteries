# FPGA-Accelerated EV Battery Prognostics using PSO-XGBoost and Deep Learning on Zynq SoC

![FPGA](https://img.shields.io/badge/FPGA-Zynq7020-blue)
![Python](https://img.shields.io/badge/Python-3.10-yellow)
![Vivado](https://img.shields.io/badge/Xilinx-Vivado%202025.2-red)
![PyTorch](https://img.shields.io/badge/DeepLearning-PyTorch-orange)
![License](https://img.shields.io/badge/License-Research%20Only-green)

---

# Table of Contents

- [Overview](#overview)
- [Research Motivation](#research-motivation)
- [Problem Statement](#problem-statement)
- [Project Objectives](#project-objectives)
- [Key Contributions](#key-contributions)
- [Complete System Workflow](#complete-system-workflow)
- [Battery Prognostics](#battery-prognostics)
- [Datasets Used](#datasets-used)
- [Data Preprocessing Pipeline](#data-preprocessing-pipeline)
- [Machine Learning Models](#machine-learning-models)
- [Deep Learning Architectures](#deep-learning-architectures)
- [PSO-XGBoost Optimization](#pso-xgboost-optimization)
- [Evaluation Metrics](#evaluation-metrics)
- [Results and Benchmarking](#results-and-benchmarking)
- [FPGA Hardware Acceleration](#fpga-hardware-acceleration)
- [Vivado Hardware Design](#vivado-hardware-design)
- [AXI-Based Hardware-Software Co-Design](#axi-based-hardware-software-co-design)
- [Fixed-Point Quantization](#fixed-point-quantization)
- [FPGA Deployment Workflow](#fpga-deployment-workflow)
- [Hardware Resource Optimization](#hardware-resource-optimization)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Applications](#applications)
- [Future Work](#future-work)
- [Author](#author)
- [License](#license)

---

# Overview

This repository presents a complete end-to-end implementation of an FPGA-accelerated intelligent Battery Management System (BMS) for Electric Vehicles (EVs).

The project focuses on accurate prediction of:

- State of Health (SOH)
- Remaining Useful Life (RUL)

using advanced Machine Learning and Deep Learning algorithms deployed on FPGA hardware.

The system integrates:

- PSO-optimized XGBoost
- GCDA-LSTM
- Transformer-LSTM
- Fixed-point quantization
- Vivado FPGA design
- AXI-based hardware-software co-design
- Zynq-7020 FPGA deployment

to achieve:

- real-time inference
- low-power edge AI
- intelligent battery monitoring
- FPGA-accelerated prognostics

---

# Research Motivation

Electric Vehicle adoption is increasing rapidly worldwide.

As EV deployment scales, Lithium-ion battery degradation becomes a major challenge due to:

- reduced driving range
- battery aging
- thermal instability
- expensive replacement costs
- battery waste generation
- safety hazards

Accurate battery prognostics enables:

- predictive maintenance
- intelligent charging strategies
- early failure detection
- second-life battery applications
- sustainable battery recycling

Traditional physics-based battery models suffer from:

- high computational complexity
- parameter sensitivity
- poor generalization
- limited adaptability

Therefore, this project explores data-driven AI approaches accelerated on FPGA hardware for real-time deployment.

---

# Problem Statement

The main problem addressed in this project is:

> How can advanced machine learning algorithms for battery prognostics be efficiently deployed on resource-constrained FPGA hardware while maintaining high prediction accuracy and real-time performance?

Major challenges include:

- limited FPGA BRAM
- hardware resource constraints
- timing closure issues
- quantization accuracy loss
- data leakage in datasets
- real-time inference requirements
- balancing accuracy and hardware efficiency

---

# Project Objectives

The primary goals of this research are:

- Develop accurate SOH prediction models
- Develop accurate RUL prediction models
- Prevent catastrophic battery failures
- Reduce computational complexity
- Enable FPGA-based edge inference
- Minimize power consumption
- Design scalable hardware architectures
- Bridge AI and digital hardware acceleration

---

# Key Contributions

## 1. PSO-XGBoost Optimization

Particle Swarm Optimization (PSO) is used to optimize XGBoost hyperparameters such as:

- learning rate
- max depth
- number of trees
- gamma
- subsample ratio
- column sample ratio

Benefits:

- improved accuracy
- reduced overfitting
- optimized inference complexity
- hardware-friendly tree structures

---

## 2. Advanced Deep Learning Architectures

Implemented advanced sequence learning architectures:

### GCDA-LSTM

Graph Convolution + Dual Attention + LSTM architecture.

Capabilities:

- captures temporal degradation
- learns feature dependencies
- improves degradation modeling

---

### Transformer-LSTM

Transformer encoder integrated with LSTM.

Capabilities:

- long-range dependency learning
- attention-based feature extraction
- improved sequence representation

---

## 3. FPGA Hardware Deployment

Machine learning models are transformed into FPGA-compatible hardware architectures using:

- fixed-point arithmetic
- quantized weights
- BRAM-based storage
- FSM-based traversal logic
- AXI-Lite communication
- hardware pipelining

---

## 4. Hardware-Software Co-Design

Implemented complete Zynq SoC architecture:

| Component | Function |
|---|---|
| ARM Cortex-A9 | Software execution |
| FPGA PL | Hardware acceleration |
| AXI4-Lite | Communication |
| BRAM | Tree storage |
| FSM | Tree traversal |
| DSP blocks | Arithmetic acceleration |

---

# Complete System Workflow

```text
Battery Dataset
        ↓
Data Cleaning
        ↓
Feature Engineering
        ↓
SOH/RUL Calculation
        ↓
Train/Validation/Test Split
        ↓
Machine Learning Training
        ↓
PSO Hyperparameter Optimization
        ↓
Deep Learning Training
        ↓
Model Benchmarking
        ↓
Quantization
        ↓
Vitis HLS / Verilog
        ↓
Vivado Integration
        ↓
Custom FPGA Accelerator
        ↓
AXI-Lite Communication
        ↓
Real-Time Hardware Inference
```

---

# Battery Prognostics

Battery prognostics estimates battery degradation using two major indicators:

---

## State of Health (SOH)

SOH represents the remaining usable capacity of the battery.

### Mathematical Definition

\[
SOH(\%)=\frac{C_{measured}}{C_{nominal}}\times100
\]

Where:

| Symbol | Meaning |
|---|---|
| \(C_{measured}\) | Current battery capacity |
| \(C_{nominal}\) | Rated battery capacity |

---

## Remaining Useful Life (RUL)

RUL estimates the remaining operational lifetime.

### Mathematical Definition

\[
RUL=N_{max}-N_{current}
\]

Where:

| Symbol | Meaning |
|---|---|
| \(N_{max}\) | Maximum cycle life |
| \(N_{current}\) | Current cycle number |

---

# Datasets Used

## NASA Battery Dataset

Source:

- NASA Prognostics Center of Excellence (PCoE)

Contains:

- 34 Li-ion batteries
- multiple thermal conditions
- variable discharge profiles

Features:

- voltage
- current
- capacity
- temperature
- cycle number

---

## Oxford Battery Dataset

Source:

- University of Oxford Battery Degradation Dataset

Contains:

- Kokam pouch cells
- Urban Artemis drive cycle
- real automotive degradation patterns

---

# Data Preprocessing Pipeline

The preprocessing pipeline includes:

- missing value removal
- outlier filtering
- normalization
- feature engineering
- sequential data generation
- train/test stratification

---

## Battery-Level Stratification

Each battery belongs exclusively to:

- training set
- validation set
- test set

This prevents:

- data leakage
- memorization
- inflated performance

---

# Feature Engineering

Additional degradation-aware features:

| Feature | Description |
|---|---|
| capacity fade | degradation magnitude |
| degradation rate | capacity loss rate |
| capacity retention | remaining capacity percentage |

---

# Machine Learning Models

Implemented models:

| Model | Category |
|---|---|
| Random Forest | Ensemble Learning |
| XGBoost | Gradient Boosting |
| PSO-XGBoost | Optimized Gradient Boosting |
| LightGBM | Gradient Boosting |
| SVR | Kernel Learning |
| MLP | Neural Network |

---

# Deep Learning Architectures

## GCDA-LSTM Architecture

Components:

- Graph convolution layer
- Multi-head attention
- LSTM sequence modeling
- Dense regression layers

Advantages:

- temporal dependency learning
- nonlinear degradation modeling
- enhanced feature correlation learning

---

## Transformer-LSTM Architecture

Components:

- positional embeddings
- transformer encoder
- self-attention
- LSTM decoder

Advantages:

- long-range sequence modeling
- attention-based learning
- improved sequence representation

---

# PSO-XGBoost Optimization

PSO minimizes validation MSE:

\[
MSE=\frac{1}{N}\sum_{i=1}^{N}(y_i-\hat{y}_i)^2
\]

Each particle represents:

- learning rate
- tree depth
- estimator count
- regularization parameters

The global best particle defines the optimal XGBoost model.

---

# Evaluation Metrics

## R² Score

\[
R^2=1-\frac{\sum(y_i-\hat{y}_i)^2}{\sum(y_i-\bar{y})^2}
\]

Measures prediction goodness.

---

## RMSE

\[
RMSE=\sqrt{\frac{1}{N}\sum(y_i-\hat{y}_i)^2}
\]

Measures prediction error magnitude.

---

## MAE

\[
MAE=\frac{1}{N}\sum|y_i-\hat{y}_i|
\]

Measures average absolute prediction error.

---

# Results and Benchmarking

## RUL Prediction Results

| Model | R² | RMSE |
|---|---|---|
| Random Forest | 0.921 | 376.14 |
| LightGBM | 0.918 | 384.33 |
| PSO-XGBoost | 0.891 | 443.34 |
| Transformer-LSTM | 0.803 | 498.2 |
| GCDA-LSTM | 0.786 | 520.2 |

---

## SOH Prediction Results

| Model | R² | RMSE |
|---|---|---|
| Random Forest | 0.999 | 0.14 |
| Transformer-LSTM | 0.996 | 0.97 |
| PSO-XGBoost | 0.992 | 0.43 |
| LightGBM | 0.986 | 1.82 |

---

# FPGA Hardware Acceleration

The project deploys optimized ML models onto:

## Xilinx Zynq-7020 FPGA

The FPGA accelerator performs:

- low-latency inference
- real-time prediction
- energy-efficient computation

---

# Vivado Hardware Design

The Vivado `design_1` architecture contains:

| Module | Purpose |
|---|---|
| Zynq Processing System | ARM processor |
| AXI Interconnect | Communication |
| Custom XGBoost IP | ML accelerator |
| BRAM | Tree storage |
| Clock Wizard | Clock generation |
| Reset Controller | Synchronization |

---

# AXI-Based Hardware-Software Co-Design

The ARM processor communicates with FPGA accelerator through:

## AXI4-Lite Interface

Communication flow:

```text
ARM Processor
      ↓
AXI Write Transaction
      ↓
FPGA Accelerator
      ↓
Tree Traversal Engine
      ↓
Prediction Output
      ↓
AXI Read Transaction
      ↓
ARM Processor
```

---

# XGBoost Hardware Inference

The FPGA accelerator computes:

\[
\hat{y}=\sum_{k=1}^{K}f_k(x)
\]

Each tree node performs:

\[
x_j \le \theta
\]

using comparator logic.

---

# Tree Traversal Hardware Flow

```text
Feature Input
      ↓
Threshold Comparator
      ↓
Branch Decision
      ↓
Next Node Address
      ↓
BRAM Access
      ↓
Leaf Node
      ↓
Prediction Accumulation
      ↓
Final Output
```

---

# Fixed-Point Quantization

The project uses:

```text
Q16.16 Fixed-Point Format
```

Benefits:

- reduced DSP usage
- lower power consumption
- faster arithmetic
- FPGA efficiency

---

# FPGA Deployment Workflow

## Step 1: Train Models

Train ML/DL models using Python.

---

## Step 2: Quantization

Convert floating-point weights into fixed-point integers.

---

## Step 3: Generate HLS Code

Generated files:

- `quantized_weights.h`
- `inference.cpp`
- `testbench.cpp`

---

## Step 4: Vitis HLS Synthesis

Convert C++ into RTL hardware.

---

## Step 5: Vivado Integration

Integrate accelerator IP into Zynq design.

---

## Step 6: Bitstream Generation

Generate FPGA configuration bitstream.

---

## Step 7: Hardware Validation

Validate prediction accuracy on FPGA.

---

# Hardware Resource Optimization

Optimization techniques:

- fixed-point arithmetic
- BRAM mapping
- hardware pipelining
- FSM traversal
- AXI-Lite communication
- memory-efficient tree storage

---

# Vivado Timing Analysis

Timing analysis includes:

- synthesis reports
- implementation reports
- WNS/TNS analysis
- clock utilization
- DSP usage
- LUT usage
- BRAM usage

---

# Repository Structure

```text
├── datasets/
├── outputs/
├── figures/
├── notebooks/
├── fpga_deployment/
├── vivado_project/
├── models/
├── inference.cpp
├── quantized_weights.h
├── testbench.cpp
├── requirements.txt
└── README.md
```

---

# Installation

## Clone Repository

```bash
git clone https://github.com/your_username/fpga-battery-prognostics.git
cd fpga-battery-prognostics
```

---

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Usage

## Train Models

```bash
python train.py
```

---

## Run FPGA Deployment

```bash
python generate_hls.py
```

---

## Launch Vivado Project

```bash
vivado design_1.xpr
```

---

# Applications

- Electric Vehicle Battery Management Systems
- Smart Energy Storage
- Predictive Maintenance
- Battery Recycling
- Intelligent Charging Stations
- Edge AI Systems
- Embedded Prognostics

---

# Future Work

Future extensions include:

- CNN-LSTM FPGA deployment
- Transformer hardware acceleration
- TinyML deployment
- ASIC implementation
- multi-battery pack prediction
- edge-cloud BMS systems
- dynamic partial reconfiguration

---

# Research Impact

This project demonstrates that:

- advanced AI models
- can be accelerated using FPGA hardware
- for real-time EV battery prognostics
- with low latency and low power
- suitable for intelligent edge deployment

---

# Author

## Supriyo Roy

Electrical Engineering  
Indian Institute of Engineering Science and Technology (IIEST) Shibpur

Research Areas:

- FPGA acceleration
- AI hardware
- Embedded ML
- Battery prognostics
- Digital design
- Edge AI systems

---

# License

Copyright © Supriyo Roy

All rights reserved.

This repository is intended strictly for:

- academic research
- portfolio demonstration
- educational presentation

Unauthorized use, modification, or redistribution is prohibited.
