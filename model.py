import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
st.set_page_config(page_title="Stock Predictor", layout="centered", initial_sidebar_state="auto")
# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv("BSE_30.csv")
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values(by='Date')
    return df

df = load_data()

# UI layout
st.title("📈 Stock Price Predictor - BSE 30")
symbols = df["Symbol"].unique()
selected_symbol = st.selectbox("Choose a Stock Symbol", sorted(symbols))

# Filter and preprocess
data = df[df["Symbol"] == selected_symbol].copy()
data['Prev_Close'] = data['Close'].shift(1)
data.dropna(inplace=True)

if data.shape[0] < 50:
    st.warning("Not enough data to build a reliable model.")
    st.stop()

X = data[['Open', 'High', 'Low', 'Volume', 'Prev_Close']]
y = data['Close']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Models
models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(alpha=1.0),
    "Lasso Regression": Lasso(alpha=0.1),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42)
}

# Train and evaluate
metrics = []
best_model = None
best_r2 = -np.inf
best_name = ""
best_preds = None

st.subheader(f"🔍 Model Evaluation for {selected_symbol}")
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    metrics.append((name, mae, rmse, r2))

    if r2 > best_r2:
        best_model = model
        best_preds = y_pred
        best_r2 = r2
        best_name = name

# Show metrics
metrics_df = pd.DataFrame(metrics, columns=["Model", "MAE", "RMSE", "R²"]).sort_values(by="R²", ascending=False)
st.dataframe(metrics_df.set_index("Model").style.background_gradient(cmap='Greens', axis=0))

# Plot best model predictions
st.subheader(f"📊 {best_name} Predictions vs Actual")
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(y_test.values, label="Actual", color="blue")
ax.plot(best_preds, label="Predicted", color="orange")
ax.set_xlabel("Samples")
ax.set_ylabel("Closing Price")
ax.set_title(f"{selected_symbol} - Actual vs Predicted ({best_name})")
ax.legend()
st.pyplot(fig)

# Save model
model_file = f"{selected_symbol}_{best_name.replace(' ', '_')}.pkl"
joblib.dump(best_model, model_file)
st.success(f"✅ Best model saved as `{model_file}`")
