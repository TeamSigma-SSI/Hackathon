import pandas as pd
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.stats.diagnostic import acorr_ljungbox
import matplotlib.pyplot as plt
import scipy.stats as stats
import seaborn as sns
import numpy as np
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler


# Given WPI data
data = {
    "Year": list(range(1990, 2025)),
    "WPI": [
        204.858, 270.5749, 189.7332, 191.8816, 211.5574, 251.0384, 196.83, 217.4527,
        189.7584, 212.3109, 237.3597, 231.4219, 199.6758, 225.0317, 251.2593, 195.8788,
        233.3279, 227.4168, 205.5142, 187.632, 213.5787, 265.2407, 240.5312, 193.2754,
        219.8234, 186.5167, 181.9963, 194.6301, 273.2243, 179.8054, 191.5618, 185.2952,
        240.9461, 199.493, 242.5522
    ]
}

df = pd.DataFrame(data)
df.set_index("Year", inplace=True)

# Perform ADF test
adf_result = adfuller(df["WPI"])

# Extract results
adf_statistic = adf_result[0]
p_value = adf_result[1]
critical_values = adf_result[4]

# Print results
print(f"ADF Statistic: {adf_statistic}")
print(f"p-value: {p_value}")
print("Critical Values:")
for key, value in critical_values.items():
    print(f"   {key}: {value}")

# Conclusion
if p_value < 0.05:
    print("The data is stationary (reject the null hypothesis).")
else:
    print("The data is non-stationary (fail to reject the null hypothesis).")

# Plot ACF
plt.figure(figsize=(6, 4))
plot_acf(df["WPI"], lags=16)
plt.title("Autocorrelation Function (ACF)")
plt.show()

# Plot PACF
plt.figure(figsize=(6, 4))
plot_pacf(df["WPI"], lags=16, method='ywm')
plt.title("Partial Autocorrelation Function (PACF)")
plt.show()

wpi_values = df['WPI']

# Fit ARMA(1,1) Model
arma_11 = ARIMA(df["WPI"], order=(1, 0, 1)).fit()

# Fit ARMA(1,3) Model
arma_13 = ARIMA(df["WPI"], order=(1, 0, 3)).fit()

# Print model summaries
print("ARMA(1,1) Summary:")
print(arma_11.summary())

print("\nARMA(1,3) Summary:")
print(arma_13.summary())

# Perform Ljung-Box test for ARMA(1,1)
ljungbox_11 = acorr_ljungbox(arma_11.resid, lags=[10], return_df=True)
print("Ljung-Box Test for ARMA(1,1):")
print(ljungbox_11)

# Perform Ljung-Box test for ARMA(1,3)
ljungbox_13 = acorr_ljungbox(arma_13.resid, lags=[10], return_df=True)
print("\nLjung-Box Test for ARMA(1,3):")
print(ljungbox_13)

# Plot residuals for both models as scatter plots (dots)
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(range(len(arma_11.resid)), arma_11.resid, color="blue", alpha=0.6)
plt.axhline(0, linestyle="--", color="black")
plt.title("Residuals for ARMA(1,1)")
plt.xlabel("Time")
plt.ylabel("Residuals")

plt.subplot(1, 2, 2)
plt.scatter(range(len(arma_13.resid)), arma_13.resid, color="red", alpha=0.6)
plt.axhline(0, linestyle="--", color="black")
plt.title("Residuals for ARMA(1,3)")
plt.xlabel("Time")
plt.ylabel("Residuals")

plt.tight_layout()
plt.show()

# Function to check normality of residuals
def check_residual_normality(model_resid, model_name):
    resid = model_resid
    shapiro_test = stats.shapiro(resid)

    print(f"\nShapiro-Wilk Test for {model_name}:")
    print(f"Statistic: {shapiro_test.statistic}, p-value: {shapiro_test.pvalue}")

    plt.figure(figsize=(12, 5))

    # Histogram
    plt.subplot(1, 2, 1)
    sns.histplot(resid, kde=True, bins=20, color="blue")
    plt.title(f"Residual Histogram ({model_name})")

    # QQ Plot
    plt.subplot(1, 2, 2)
    stats.probplot(resid, dist="norm", plot=plt)
    plt.title(f"QQ Plot ({model_name})")

    plt.tight_layout()
    plt.show()

# Run normality check for ARMA(1,1) and ARMA(1,3)
check_residual_normality(arma_11.resid, "ARMA(1,1)")
check_residual_normality(arma_13.resid, "ARMA(1,3)")

# Define train-test split (e.g., last 5 years for testing)
train_size = int(len(df) * 0.8)  # 80% train, 20% test
train, test = df["WPI"][:train_size], df["WPI"][train_size:]

# Print sizes
print(f"Train size: {len(train)}, Test size: {len(test)}")

# Fit ARMA(1,1) model on train data
arma_11_model = ARIMA(train, order=(1, 0, 1)).fit()

# Forecast for test period
forecast_11 = arma_11_model.forecast(steps=len(test))

# Compute RMSE for ARMA(1,1)
rmse_11 = np.sqrt(mean_squared_error(test, forecast_11))
print(f"RMSE for ARMA(1,1): {rmse_11}")

# Fit ARMA(1,3) model on train data
arma_13_model = ARIMA(train, order=(1, 0, 3)).fit()

# Forecast for test period
forecast_13 = arma_13_model.forecast(steps=len(test))

# Compute RMSE for ARMA(1,3)
rmse_13 = np.sqrt(mean_squared_error(test, forecast_13))
print(f"RMSE for ARMA(1,3): {rmse_13}")

# Fit SES model on train data
ses_model = SimpleExpSmoothing(train).fit()

# Forecast using SES
forecast_ses = ses_model.forecast(steps=len(test))

# Compute RMSE for SES
rmse_ses = np.sqrt(mean_squared_error(test, forecast_ses))
print(f"RMSE for SES: {rmse_ses}")

# Create lag features (shifted values) for supervised learning
df["Lag_1"] = df["WPI"].shift(1)
df.dropna(inplace=True)  # Remove NaN values from shifting

# Train-test split
train_size = int(len(df) * 0.8)
train_X, test_X = df[["Lag_1"]].iloc[:train_size], df[["Lag_1"]].iloc[train_size:]
train_y, test_y = df["WPI"].iloc[:train_size], df["WPI"].iloc[train_size:]

# Scale data for SVM (standardization helps SVM perform better)
scaler_X = StandardScaler()
scaler_y = StandardScaler()
train_X_scaled = scaler_X.fit_transform(train_X)
test_X_scaled = scaler_X.transform(test_X)
train_y_scaled = scaler_y.fit_transform(train_y.values.reshape(-1,1)).ravel()
test_y_scaled = scaler_y.transform(test_y.values.reshape(-1,1)).ravel()

# Train SVM model
svm_model = SVR(kernel="rbf", C=100, gamma=0.1, epsilon=0.1)
svm_model.fit(train_X_scaled, train_y_scaled)

# Predict for test data
forecast_svm_scaled = svm_model.predict(test_X_scaled)
forecast_svm = scaler_y.inverse_transform(forecast_svm_scaled.reshape(-1, 1)).ravel()

# Compute RMSE for SVM
rmse_svm = np.sqrt(mean_squared_error(test_y, forecast_svm))
print(f"RMSE for SVM: {rmse_svm}")

# Initialize and fit the KNN model
knn_model = KNeighborsRegressor(n_neighbors=5)  # You can tune n_neighbors
knn_model.fit(train_X, train_y)

# Predict on test set
forecast_knn = knn_model.predict(test_X)

from sklearn.metrics import mean_squared_error

rmse_knn = np.sqrt(mean_squared_error(test_y, forecast_knn))
print(f"KNN RMSE: {rmse_knn:.2f}")

# Generate future Lag_1 values (use last known value to create future points)
future_lags = [df["Lag_1"].iloc[-1]]  # Start with the last available value

# Predict next 5 years iteratively
future_knn = []
for _ in range(5):  # 5 years into the future
    next_pred = knn_model.predict(np.array(future_lags[-1]).reshape(-1, 1))[0]
    future_knn.append(next_pred)
    future_lags.append(next_pred)  # Use the predicted value for the next iteration

# Convert future years into an array
future_years_knn = np.arange(df.index[-1] + 1, df.index[-1] + 6)

from sklearn.ensemble import RandomForestRegressor

# Initialize and train the Random Forest model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(train_X, train_y)

from sklearn.metrics import mean_squared_error
import numpy as np

# Predict on test set
forecast_rf = rf_model.predict(test_X)

# Compute RMSE
rmse_rf = np.sqrt(mean_squared_error(test_y, forecast_rf))
print(f"Random Forest RMSE: {rmse_rf:.2f}")

# Generate future Lag_1 values (start with the last known value)
future_lags_rf = [df["Lag_1"].iloc[-1]]

# Predict next 5 years iteratively
future_rf = []
for _ in range(5):  # Predict for next 5 years
    next_pred = rf_model.predict(np.array(future_lags_rf[-1]).reshape(-1, 1))[0]
    future_rf.append(next_pred)
    future_lags_rf.append(next_pred)  # Use predicted value for next iteration

# Convert future years into an array
future_years_rf = np.arange(df.index[-1] + 1, df.index[-1] + 6)

plt.figure(figsize=(12, 6))

# Plot actual data
plt.plot(df.index, df["WPI"], label="Actual Data", color="black", marker="o")

# Test set predictions
plt.plot(test.index, forecast_11, label=f"ARMA(1,1) (RMSE: {rmse_11:.2f})", color="blue", linestyle="--", marker="o")
plt.plot(test.index, forecast_13, label=f"ARMA(1,3) (RMSE: {rmse_13:.2f})", color="red", linestyle="--", marker="d")
plt.plot(test.index, forecast_ses, label=f"SES (RMSE: {rmse_ses:.2f})", color="green", linestyle="--", marker="^")
plt.plot(test.index, forecast_svm, label=f"SVM (RMSE: {rmse_svm:.2f})", color="purple", linestyle="--", marker="x")
plt.plot(test.index, forecast_knn, label=f"KNN (RMSE: {rmse_knn:.2f})", color="orange", linestyle="--", marker="s")
plt.plot(test.index, forecast_rf, label=f"Random Forest (RMSE: {rmse_rf:.2f})", color="brown", linestyle="--", marker="P")

# Future forecasts
plt.plot(future_years_rf, future_arma_11, label="Future ARMA(1,1)", color="blue", linestyle="-.")
plt.plot(future_years_rf, future_arma_13, label="Future ARMA(1,3)", color="red", linestyle="-.")
plt.plot(future_years_rf, future_ses, label="Future SES", color="green", linestyle="-.")
plt.plot(future_years_rf, future_svm_preds, label="Future SVM", color="purple", linestyle="-.")
plt.plot(future_years_rf, future_knn, label="Future KNN", color="orange", linestyle="-.")
plt.plot(future_years_rf, future_rf, label="Future Random Forest", color="brown", linestyle="-.")

# Labels and title
plt.xlabel("Year")
plt.ylabel("WPI")
plt.title("WPI Prediction: Test Data & Future Forecast")
plt.legend()
plt.show()

import matplotlib.pyplot as plt

plt.style.use("seaborn-v0_8-dark")  # Applying a clean style

fig, ax = plt.subplots(figsize=(12, 6))

# Plot actual test data
ax.plot(test.index, test_y, label="Actual WPI", color="black", marker="o", linestyle="-", linewidth=2)

# Plot different model predictions on test data
ax.plot(test.index, forecast_11, label=f"ARMA(1,1) (RMSE: {rmse_11:.2f})", color="blue", linestyle="--", marker="o")
ax.plot(test.index, forecast_13, label=f"ARMA(1,3) (RMSE: {rmse_13:.2f})", color="red", linestyle="--", marker="d")
ax.plot(test.index, forecast_ses, label=f"SES (RMSE: {rmse_ses:.2f})", color="green", linestyle="--", marker="^")
ax.plot(test.index, forecast_svm, label=f"SVM (RMSE: {rmse_svm:.2f})", color="purple", linestyle="--", marker="x")
ax.plot(test.index, forecast_knn, label=f"KNN (RMSE: {rmse_knn:.2f})", color="orange", linestyle="--", marker="s")
ax.plot(test.index, forecast_rf, label=f"Random Forest (RMSE: {rmse_rf:.2f})", color="brown", linestyle="--", marker="P")

# Labels, Title & Grid
ax.set_xlabel("Year", fontsize=12)
ax.set_ylabel("WPI", fontsize=12)
ax.set_title("Actual vs. Predicted WPI (Test Data)", fontsize=14, fontweight="bold")
ax.legend(fontsize=10, loc="best")
ax.grid(True, linestyle="--", alpha=0.6)

plt.show()

fig, ax = plt.subplots(figsize=(12, 6))

# Future forecast plots
ax.plot(future_years_rf, future_arma_11, label="Future ARMA(1,1)", color="blue", linestyle="-", marker="o")
ax.plot(future_years_rf, future_arma_13, label="Future ARMA(1,3)", color="red", linestyle="-", marker="d")
ax.plot(future_years_rf, future_ses, label="Future SES", color="green", linestyle="-", marker="^")
ax.plot(future_years_rf, future_svm_preds, label="Future SVM", color="purple", linestyle="-", marker="x")
ax.plot(future_years_rf, future_knn, label="Future KNN", color="orange", linestyle="-", marker="s")
ax.plot(future_years_rf, future_rf, label="Future Random Forest", color="brown", linestyle="-", marker="P")

# Labels, Title & Grid
ax.set_xlabel("Year", fontsize=12)
ax.set_ylabel("Predicted WPI", fontsize=12)
ax.set_title("Future WPI Forecast (Next 5 Years)", fontsize=14, fontweight="bold")
ax.legend(fontsize=10, loc="best")
ax.grid(True, linestyle="--", alpha=0.6)

plt.show()

# Generate future predictions for all models
future_arma_11 = np.array(arma_11.forecast(steps=future_steps))  # ARMA(1,1)
future_arma_13 = np.array(arma_13.forecast(steps=future_steps))  # ARMA(1,3)
future_ses = np.array(ses_model.forecast(steps=future_steps))  # SES
future_svm_preds = []  # SVM iterative approach

future_X = np.array([df["WPI"].iloc[-1]])  # Start with last known value
for _ in range(future_steps):
    future_X_scaled = scaler_X.transform([[future_X[0]]])  # Ensure correct shape
    future_y_scaled = svm_model.predict(future_X_scaled)
    future_y = scaler_y.inverse_transform(future_y_scaled.reshape(-1, 1)).ravel()
    future_svm_preds.append(future_y[0])
    future_X = np.array([future_y[0]])  # Use predicted value as next input

# Print Predictions
print("Future WPI Predictions for Next 5 Years:")
print(f"Years: {future_years_knn.ravel()}")
print(f"ARMA(1,1): {future_arma_11}")
print(f"ARMA(1,3): {future_arma_13}")
print(f"SES: {future_ses}")
print(f"SVM: {future_svm_preds}")
print(f"KNN: {future_knn}")

# Create a DataFrame for neater formatting
future_df = pd.DataFrame({
    "Year": future_years_knn.ravel(),
    "ARMA(1,1)": future_arma_11,
    "ARMA(1,3)": future_arma_13,
    "SES": future_ses,
    "SVM": future_svm_preds,
    "KNN": future_knn
})

# Print the table
print("\nFuture WPI Predictions for Next 5 Years:\n")
print(future_df.to_string(index=False))  # Print without DataFrame index