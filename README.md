### Stock Price Forecasting Using LSTM

This project is designed to predict stock prices using a Long Short-Term Memory (LSTM) neural network model, which is particularly suited for time-series forecasting. The dataset consists of historical stock prices, and the model is trained to forecast future stock trends based on past performance.

#### Project Workflow:
1. **Data Loading and Preprocessing**:
   - **CSV Data Import**: The project begins by importing stock price data from a CSV file.
   - **Feature Selection**: Focused on the 'Open' price column for training.
   - **Data Reshaping**: Since LSTM models require 3D input, the data was reshaped into sequences of 30-day time windows.
   - **Normalization**: Utilized MinMaxScaler to scale the stock prices between 0 and 1, which is crucial for efficient neural network performance.

2. **Model Architecture**:
   - **Stacked LSTM Layers**: The sequential LSTM model consists of two LSTM layers:
     - First layer with 50 units and `return_sequences=True` to maintain the 3D structure.
     - Second LSTM layer with 20 units, followed by a Dense layer for the final prediction.
   - **Dropout Layers**: Dropout layers with a 20% dropout rate are used between LSTM layers to prevent overfitting and enhance generalization.
   - **Dense Output Layer**: A single neuron in the output layer is used to predict the next day's stock price, with ReLU activation.
   - **Compilation**: The model was compiled using the Adam optimizer, with Mean Squared Error (MSE) as the loss function.

3. **Training the Model**:
   - The LSTM model was trained for 50 epochs with a batch size of 10. 
   - The training data was iteratively passed through the model to reduce the MSE loss.
   - Training performance was visualized by plotting the loss function.

4. **Making Predictions**:
   - After training, the model was tested on the last 30 days of stock data to predict the next stock price.
   - Used inverse transformation to convert the scaled predictions back into their original price range for better interpretability.
   - Forecasted stock prices for the next 10 days, simulating real-world future stock movement.

5. **Visualization & Results**:
   - **Loss Curve**: The loss during training was plotted to visualize how well the model converged.
   - **Prediction vs Actual**: A comparison plot was created to juxtapose the predicted stock prices with actual values from the test data.
   - **Future Forecasting**: Predicted prices for the next 10 days were displayed, helping in visualizing the expected market trend.

6. **Further Improvements**:
   - This project can be extended by incorporating additional features like volume, high/low prices, or technical indicators (e.g., RSI, MACD) for more accurate predictions.
   - Experimenting with other neural network architectures (GRU, CNN-LSTM) could improve performance.

#### Tools and Libraries:
- **Python**: Core language used.
- **TensorFlow & Keras**: For building and training the LSTM neural network.
- **NumPy & Pandas**: For efficient data handling and preprocessing.
- **Matplotlib**: For plotting and visualizing results.
- **Scikit-learn**: For scaling data using MinMaxScaler and evaluating results.

This project demonstrates the power of LSTM for time-series forecasting and can be extended for more advanced stock price predictions and financial modeling.
