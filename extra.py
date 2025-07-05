
def LSTM_ALGO(df: pd.DataFrame):
    # Parse date for indexing
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')
    
    # Prepare training and testing datasets (80% train, 20% test)
    dataset_train = df.iloc[0:int(0.8*len(df)), :]
    dataset_test = df.iloc[int(0.8*len(df)):, :]
    
    # Feature scaling
    training_set = df[['close']].values
    sc = MinMaxScaler(feature_range=(0, 1))
    training_set_scaled = sc.fit_transform(training_set)
    
    # Prepare input data with 7 timesteps
    X_train, y_train = [], []
    for i in range(7, len(training_set_scaled)):
        X_train.append(training_set_scaled[i-7:i, 0])
        y_train.append(training_set_scaled[i, 0])
    
    # Convert to numpy arrays
    X_train, y_train = np.array(X_train), np.array(y_train)
    
    # Reshape input data to 3D for LSTM [samples, timesteps, features]
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    
    # LSTM model
    regressor = Sequential()
    regressor.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    regressor.add(Dropout(0.1))
    regressor.add(LSTM(units=50, return_sequences=True))
    regressor.add(Dropout(0.1))
    regressor.add(LSTM(units=50, return_sequences=True))
    regressor.add(Dropout(0.1))
    regressor.add(LSTM(units=50))
    regressor.add(Dropout(0.1))
    regressor.add(Dense(units=1))
    
    # Compile the model
    regressor.compile(optimizer='adam', loss='mean_squared_error')
    
    # Train the model
    regressor.fit(X_train, y_train, epochs=25, batch_size=32, verbose=0)
    
    # Prepare testing data
    real_stock_price = dataset_test[['close']].values
    dataset_total = pd.concat((dataset_train['close'], dataset_test['close']), axis=0)
    inputs = dataset_total[len(dataset_total) - len(dataset_test) - 7:].values.reshape(-1, 1)
    inputs = sc.transform(inputs)
    
    X_test = []
    for i in range(7, len(inputs)):
        X_test.append(inputs[i-7:i, 0])
    
    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    
    # Predict stock prices
    predicted_stock_price = regressor.predict(X_test)
    predicted_stock_price = sc.inverse_transform(predicted_stock_price)
    
    # Plotting the results
    fig = plt.figure(figsize=(7.2, 4.8), dpi=65)
    plt.plot(real_stock_price, label='Actual Price')  
    plt.plot(predicted_stock_price, label='Predicted Price')
    plt.legend(loc='best')
    plt.savefig('static/graph/LSTM.png')
    plt.close(fig)
    
    # Calculate RMSE
    error_lstm = math.sqrt(mean_squared_error(real_stock_price, predicted_stock_price))
    
    # Forecasting the next day's price
    X_forecast = np.array([inputs[-7:, 0]])
    X_forecast = np.reshape(X_forecast, (1, X_forecast.shape[1], 1))
    forecasted_stock_price = regressor.predict(X_forecast)
    forecasted_stock_price = sc.inverse_transform(forecasted_stock_price)
    
    lstm_pred = forecasted_stock_price[0, 0]
    
    return lstm_pred, error_lstm


@app.post('/predict/lstm')
async def predict_lstm(request : Request ,stock_symbols: List[str] = Form(...)):
    conn = sqlite3.connect(config.DB_PATH)
    query = f"""
        SELECT s.symbol, sp.date, sp.close, sp.open, sp.high, sp.low, sp.volume
        FROM stock s
        JOIN stock_price sp ON s.id = sp.stock_id
        WHERE s.symbol IN ({','.join(['?'] * len(stock_symbols))})
        ORDER BY sp.date ASC
    """
    df = pd.read_sql_query(query, conn, params=stock_symbols)
    conn.close()

    if df.empty:
        raise HTTPException(status_code=404, detail='No stock data found for provided symbols')
    
    timestamp = datetime.datetime.now().timestamp()

    results = {}
    for symbol in stock_symbols:
        company_data = df[df['symbol'] == symbol]
        prediction, rmse = LSTM_ALGO(company_data)
        results[symbol] = {'prediction': prediction, 'rmse': rmse, 'model' : 'LSTM',"timestamp":timestamp}

    return templates.TemplateResponse("predictions.html",{"predictions":results, "request":request})

def LIN_REG_ALGO(df: pd.DataFrame):
    # Number of days to forecast in the future
    forecast_out = 7

    # Prepare the dataset
    df['close after n days'] = df['close'].shift(-forecast_out)
    df_new = df[['close', 'close after n days']]

    # Structure data for train, test & forecast
    y = np.array(df_new.iloc[:-forecast_out, -1]).reshape(-1, 1)  # Labels
    X = np.array(df_new.iloc[:-forecast_out, 0:-1])  # Features
    X_to_be_forecasted = np.array(df_new.iloc[-forecast_out:, 0:-1])  # Future data

    # Splitting the data into training and testing sets
    X_train = X[:int(0.8 * len(X))]
    X_test = X[int(0.8 * len(X)):]
    y_train = y[:int(0.8 * len(y))]
    y_test = y[int(0.8 * len(y)):]

    # Feature Scaling (Normalization)
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    X_to_be_forecasted = sc.transform(X_to_be_forecasted)

    # Training the Linear Regression model
    clf = LinearRegression(n_jobs=-1)
    clf.fit(X_train, y_train)

    # Testing the model
    y_test_pred = clf.predict(X_test)
    y_test_pred = y_test_pred * 1.04  # Small adjustment factor

    # Visualization
    fig = plt.figure(figsize=(7.2, 4.8), dpi=65)
    plt.plot(y_test, label='Actual Price')
    plt.plot(y_test_pred, label='Predicted Price')
    plt.legend(loc='lower right')
    plt.savefig('static/graph/LR.png')
    plt.close(fig)

    # Calculate RMSE
    error_lr = math.sqrt(mean_squared_error(y_test, y_test_pred))

    # Forecasting future prices
    forecast_set = clf.predict(X_to_be_forecasted)
    forecast_set = forecast_set * 1.04  # Apply the adjustment
    lr_pred = forecast_set[0, 0]

   
    print("##############################################################################")
    print("Tomorrow's Closing Price Prediction by Linear Regression: ", lr_pred)
    print("Linear Regression RMSE:", error_lr)
    print("##############################################################################")

    return lr_pred, error_lr

@app.post('/predict/linear-regression')
async def predict_linear_regression(request:Request,stock_symbols: List[str] = Form(...)):
    conn = sqlite3.connect(config.DB_PATH)
    query = f"""
        SELECT s.symbol, sp.date, sp.close, sp.open, sp.high, sp.low, sp.volume
        FROM stock s
        JOIN stock_price sp ON s.id = sp.stock_id
        WHERE s.symbol IN ({','.join(['?'] * len(stock_symbols))})
        ORDER BY sp.date ASC
    """
    df = pd.read_sql_query(query, conn, params=stock_symbols)
    conn.close()

    if df.empty:
        raise HTTPException(status_code=404, detail='No stock data found for provided symbols')

    results = {}
    for symbol in stock_symbols:
        company_data = df[df['symbol'] == symbol]
        prediction, rmse = LIN_REG_ALGO(company_data)
        results[symbol] = {'prediction': prediction, 'rmse': rmse, 'model': 'Linear Regression'}

    return templates.TemplateResponse("predictions.html",{"predictions":results, "request":request})


# Data preparation helper function

def prepare_data(data, time_steps):
    X, y = [], []
    for i in range(len(data) - time_steps):
        X.append(data[i:(i + time_steps), 0])
        y.append(data[i + time_steps, 0])
    return np.array(X), np.array(y)

# Function to find the best ARIMA parameters
def find_best_arima(data, p_range=(0, 3), d_range=(0, 2), q_range=(0, 3)):
    best_aic = np.inf
    best_order = None
    
    for p, d, q in product(range(*p_range), range(*d_range), range(*q_range)):
        try:
            model = ARIMA(data, order=(p, d, q))
            results = model.fit()
            if results.aic < best_aic:
                best_aic = results.aic
                best_order = (p, d, q)
        except Exception as e:
            continue
    
    return best_order

def hybrid_model(train_data, test_data, time_steps=60):
    # Ensure data is a numpy array and handle missing values
    train_df = np.array(train_data).reshape(-1, 1)
    train_df = pd.DataFrame(train_df).ffill().values

    # Get best ARIMA model parameters using grid search
    best_order = find_best_arima(train_df)
    print(f"Best ARIMA order: {best_order}")

    # Initial ARIMA model training
    arima_model = ARIMA(train_df, order=best_order)
    arima_results = arima_model.fit()

    # Get ARIMA residuals
    arima_residuals = train_df - arima_results.fittedvalues.reshape(-1, 1)
    arima_residuals = np.nan_to_num(arima_residuals)

    # Prepare data for LSTM
    scaler = MinMaxScaler()
    residuals_scaled = scaler.fit_transform(arima_residuals)

    X, y = prepare_data(residuals_scaled, time_steps)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    # LSTM model
    lstm_model = Sequential([
        LSTM(units=50, return_sequences=True, input_shape=(X.shape[1], 1)),
        LSTM(units=50),
        Dense(units=1)
    ])
    lstm_model.compile(optimizer='adam', loss='mean_squared_error')
    lstm_model.fit(X, y, epochs=50, batch_size=32, verbose=0)

    # Make predictions for test data
    predictions = []
    test_data = np.array(test_data).reshape(-1, 1)
    combined_data = np.vstack((train_df, test_data))

    for i in range(len(test_data)):
        # ARIMA prediction
        arima_forecast = arima_results.forecast(steps=1)

        # LSTM prediction
        last_60_days = scaler.transform(combined_data[-(time_steps+1):-1])
        X_test = np.array([last_60_days])
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
        lstm_prediction = lstm_model.predict(X_test)
        lstm_prediction = scaler.inverse_transform(lstm_prediction)

        # Combine predictions
        hybrid_prediction = arima_forecast + lstm_prediction[0][0]
        predictions.append(hybrid_prediction[0])

        # Update ARIMA model by re-fitting with new data
        updated_data = np.vstack((train_df, test_data[:i+1]))
        arima_model = ARIMA(updated_data, order=best_order)
        arima_results = arima_model.fit()

    return np.array(predictions)

@app.post("/predict/hybrid")
async def predict_hybrid(request:Request,stock_symbols: List[str] = Form(...)):
    conn = sqlite3.connect(config.DB_PATH)
    query = f"""
        SELECT s.symbol, sp.date, sp.close
        FROM stock s
        JOIN stock_price sp ON s.id = sp.stock_id
        WHERE s.symbol IN ({','.join(['?'] * len(stock_symbols))})
        ORDER BY sp.date ASC
    """
    df = pd.read_sql_query(query, conn, params=stock_symbols)
    conn.close()

    if df.empty:
        raise HTTPException(status_code=404, detail='No stock data found for provided symbols')

    results = {}
    for symbol in stock_symbols:
        company_data = df[df['symbol'] == symbol]
        train_data = company_data['close'][:-30]  # Use most of the data for training
        test_data = company_data['close'][-30:]   # Use last 30 days for testing

        predictions = hybrid_model(train_data, test_data)
        timestamp = datetime.datetime.now().timestamp()

        # Plotting the results
        plt.figure(figsize=(10, 6))
        plt.plot(company_data['date'][-30:], test_data, label='Actual Prices')
        plt.plot(company_data['date'][-30:], predictions, label='Predicted Prices')
        plt.title(f'{symbol} Stock Price Prediction')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        graph_path = f'static/graph/hybrid_{timestamp}.png'
        plt.savefig(graph_path)
        plt.close()

        results[symbol] = {
            'prediction': predictions[-1],
            'model': 'Hybrid',
            'graph_url': graph_path
        }

    return templates.TemplateResponse("predictions.html",{"predictions":results, "request":request})
