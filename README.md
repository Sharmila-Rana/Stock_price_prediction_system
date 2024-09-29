### TCS Stock Price Prediction using LSTM

This project focuses on predicting TCS stock prices using a Long Short-Term Memory (LSTM) neural network, a powerful architecture suitable for time series forecasting. The project utilizes historical stock price data and applies machine learning techniques to predict future stock prices.

#### Key Features:
- **Data Preprocessing**: 
   - Loaded and cleaned stock price data from a CSV file using Pandas, filling missing values using forward fill.
   - Applied feature scaling using `MinMaxScaler` to normalize the stock price data between 0 and 1 for better training performance.
   
- **LSTM Model**:
   - Built a sequential LSTM model using TensorFlow and Keras. The model includes multiple LSTM layers with dropout regularization to prevent overfitting.
   - The input is structured as a rolling window of the last 30 days of stock prices, predicting the next day's price.
   
- **Model Training**:
   - Trained the model using 30 epochs and a batch size of 20, optimizing with the Adam optimizer and minimizing mean squared error (MSE).
   - The model was trained on the "Open" prices of the stock.

- **Prediction & Visualization**:
   - The model's predictions are evaluated on both the training data and future time periods.
   - Visualized the actual vs predicted stock prices using Matplotlib, giving insights into model performance.
   - Predicted stock prices for the next 10 days, allowing users to make forecasts based on recent trends.

#### Key Libraries:
- Pandas for data manipulation
- NumPy for numerical computations
- Matplotlib for plotting the results
- TensorFlow & Keras for building and training the LSTM model
- Scikit-learn for data normalization

This project demonstrates the use of machine learning models, specifically LSTM networks, for forecasting future stock prices based on historical data.

    

