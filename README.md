# Mql_Neural: A set of MQL5 EAs and Python scripts that use ONNX models to predict the market

## Features

- MQL5 EAs for MetaTrader 5
- Python scripts for model training and evaluation
- ONNX models for neural network inference

## Screenshots

- SimpleONNX_3_Feat_Test results in NASDAQ100 M5

![SimpleONNX_3_Feat_Test.mq5](./img/Screenshot_2026-03-28-09-58-14_3840x1080.png)

- SimpleONNX_3_Feat_Test parameters used for backtest

![SimpleONNX_3_Feat_Test.mq5](./img/Screenshot_2026-03-28-09-58-29_3840x1080.png)

## How to backtest the EA using an inference model

- Compile ONNX as resource inside the executable

![SimpleONNX_3_Feat_Test.mq5](./img/Screenshot_2026-03-28-10-05-12_3840x1080.png)

- Load the ONNX from resource buffer

![SimpleONNX_3_Feat_Test.mq5](./img/Screenshot_2026-03-28-10-05-25_3840x1080.png)