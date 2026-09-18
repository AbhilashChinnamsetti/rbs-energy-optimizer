from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
import joblib


def train_model(df, target_col="energy_consumption", random_state=42):
    X = df[["traffic_load", "active_cells", "time_of_day", "ambient_temp", "qos_latency", "qos_throughput"]]
    y = df[target_col]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)
    model = RandomForestRegressor(n_estimators=100, random_state=random_state)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    metrics = {"r2": float(r2_score(y_test, preds)), "mae": float(mean_absolute_error(y_test, preds))}
    return model, metrics


def save_model(model, path):
    joblib.dump(model, path)


def load_model(path):
    return joblib.load(path)
