# 1D-CNN Time-Series Forecasting for Energy Consumption

I built this deep learning notebook to train a 1D Convolutional Neural Network (CNN) using PyTorch to forecast hourly grid energy demand (PJME_MW). 

##  How It Works
- **Data Prep:** Automatically downloads the historical workload dataset and structures the timelines into a 24-hour lookback window using data lags.
- **The Architecture:** Processes temporal sequence data through two tiers of `Conv1d` and `MaxPool1d` layers for pattern recognition, then passes them into a dense linear regressor backed by `Dropout` to optimize the output.
- **Evaluation:** Converts the neural network decimal weights back into standard Megawatts (MW) to calculate operational Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE) metrics.
