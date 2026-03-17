### S&P 500 Stock Price Prediction XAI Analysis

This project implements and compares two Explainable AI (XAI) attribution methods, Integrated Gradients (IG) and Layer-wise Relevance Propagation (LRP), on an LSTM architecture. The model is trained to predict S&P 500 Daily OHLCV (Open, High, Low, Close, Volume) data, utilising historical data spanning from December 1984 to May 2021.

Model architecture: A multi-layer LSTM designed for sequential time-series forecasting. \
Dataset: S&P 500 Daily OHLCV data sourced from https://github.com/Warren-Freeborough/Explainable-RNN/tree/main  
To ensure the validity of the XAI interpretations, the model architecture was kept identical to the original study (Freeborough & van
Zyl, 2022) to allow for the direct application of the pre-trained weights sourced from the official repository.


#### Requirements:
- Python 3.12+
- PyTorch
- Captum (for Integrated Gradients)
- Zennit (for LRP)
- Pandas, NumPy, Matplotlib, Seaborn

#### Implementation Details:
1. Integrated Gradients \
    IG is implemented using a Mean Baseline. Instead of comparing against a sequence of zeros, the model's prediction is explained relative to the average historical behavior of the features.
2. Layer-wise Relevance Propagation (LRP) \
    Due to the recursive nature of LSTMs, LRP is implemented using the Zennit library with an EpsilonGammaBox composite. This provides stability for noisy financial time-series data.

#### How to run:
1. Ensure the data/ folder contains the required .csv files.
2. Ensure the trained weights are in the directory
3. Open XaiMethods.ipynb and run the cells sequentially to:
- Load and preprocess the data
- Initialise the LSTM model
- Generate and compare XAI heatmaps for both IG and LRP
