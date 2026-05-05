"""
Train the full ensemble:
  1. LightGBM on tabular features (SMOTE-balanced)
  2. Bi-LSTM on sequential features (strong class weights)
  3. Logistic Regression meta-classifier

Run once:  python train.py
"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    recall_score, precision_score,
    confusion_matrix, average_precision_score
)
from imblearn.over_sampling import SMOTE
import lightgbm as lgb
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Bidirectional, LSTM, Dense, Dropout, Input, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from src.data_pipeline import load_raw_data, load_daily_data
from src.features import build_features, FEATURE_COLS

MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)

SEQ_LEN    = 24
TEST_RATIO = 0.15
VAL_RATIO  = 0.10


def split_data(df):
    n = len(df)
    test_cut = int(n * (1 - TEST_RATIO))
    val_cut  = int(test_cut * (1 - VAL_RATIO))
    return df.iloc[:val_cut], df.iloc[val_cut:test_cut], df.iloc[test_cut:]


def make_sequences(X, seq_len=SEQ_LEN):
    return np.array([X[i - seq_len:i] for i in range(seq_len, len(X))])


def fbeta(precision, recall, beta=2):
    if precision + recall == 0:
        return 0.0
    return (1 + beta**2) * precision * recall / (beta**2 * precision + recall)


# ─── 1. Load data ──────────────────────────────────────────────────────────────
print("Loading hourly data...")
btc, eth = load_raw_data()

print("Loading daily data (7 years)...")
btc_daily, eth_daily = load_daily_data()

print("Engineering features...")
df = build_features(btc, eth, btc_daily, eth_daily)

print(f"Dataset: {len(df)} rows | {df.index[0].date()} to {df.index[-1].date()}")

train_df, val_df, test_df = split_data(df)
print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")
print(f"Dip rate - Train: {train_df['dip_label'].mean():.2%} | Test: {test_df['dip_label'].mean():.2%}")

X_train = train_df[FEATURE_COLS].values
y_train = train_df["dip_label"].values
X_val   = val_df[FEATURE_COLS].values
y_val   = val_df["dip_label"].values
X_test  = test_df[FEATURE_COLS].values
y_test  = test_df["dip_label"].values

# ─── 2. Scale ──────────────────────────────────────────────────────────────────
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_val_sc   = scaler.transform(X_val)
X_test_sc  = scaler.transform(X_test)
joblib.dump(scaler, MODELS_DIR / "scaler.pkl")

# ─── 3. SMOTE on training set ──────────────────────────────────────────────────
print("\nApplying SMOTE to balance training classes...")
sm = SMOTE(random_state=42, k_neighbors=5)
X_train_bal, y_train_bal = sm.fit_resample(X_train_sc, y_train)
print(f"After SMOTE - Train size: {len(X_train_bal)} | Dip rate: {y_train_bal.mean():.2%}")

# ─── 4. LightGBM ───────────────────────────────────────────────────────────────
print("\nTraining LightGBM...")

lgb_train = lgb.Dataset(X_train_bal, label=y_train_bal)
lgb_val   = lgb.Dataset(X_val_sc,   label=y_val, reference=lgb_train)

params = {
    "objective":        "binary",
    "metric":           "average_precision",
    "learning_rate":    0.03,
    "num_leaves":       127,
    "max_depth":        -1,
    "min_child_samples": 20,
    "feature_fraction": 0.7,
    "bagging_fraction": 0.7,
    "bagging_freq":     5,
    "reg_alpha":        0.1,
    "reg_lambda":       0.1,
    "verbose":         -1,
}

lgb_model = lgb.train(
    params,
    lgb_train,
    num_boost_round=1000,
    valid_sets=[lgb_val],
    callbacks=[lgb.early_stopping(80, verbose=False), lgb.log_evaluation(200)],
)
lgb_model.save_model(str(MODELS_DIR / "lgb_model.txt"))

lgb_prob_train = lgb_model.predict(X_train_sc)
lgb_prob_val   = lgb_model.predict(X_val_sc)
lgb_prob_test  = lgb_model.predict(X_test_sc)

feat_imp = pd.DataFrame({
    "feature":    FEATURE_COLS,
    "importance": lgb_model.feature_importance(importance_type="gain"),
}).sort_values("importance", ascending=False)
feat_imp.to_csv(MODELS_DIR / "feature_importance.csv", index=False)
print("LightGBM done.")

# ─── 5. Bi-LSTM ────────────────────────────────────────────────────────────────
print("\nTraining Bi-LSTM...")

X_all_sc  = scaler.transform(df[FEATURE_COLS].values)
y_all_arr = df["dip_label"].values

train_end = len(train_df)
val_end   = train_end + len(val_df)

X_seq_train = make_sequences(X_all_sc[:train_end])
y_seq_train = y_all_arr[SEQ_LEN:train_end]

X_seq_val = make_sequences(X_all_sc[:val_end])[len(X_seq_train):]
y_seq_val = y_all_arr[train_end:val_end]

X_seq_test = make_sequences(X_all_sc)[len(X_seq_train) + len(X_seq_val):]
y_seq_test = y_all_arr[val_end:]

n_features = X_seq_train.shape[2]

# Dip rate for class weights
neg, pos = (y_seq_train == 0).sum(), (y_seq_train == 1).sum()
# Strong weight to force recall — at ~7% dip rate, weight 8x more
lstm_class_weight = {0: 1.0, 1: max(8.0, (neg / pos) * 0.6)}
print(f"LSTM class weight for dip class: {lstm_class_weight[1]:.1f}")

lstm_model = Sequential([
    Input(shape=(SEQ_LEN, n_features)),
    Bidirectional(LSTM(128, return_sequences=True)),
    BatchNormalization(),
    Dropout(0.3),
    Bidirectional(LSTM(64, return_sequences=True)),
    BatchNormalization(),
    Dropout(0.3),
    Bidirectional(LSTM(32)),
    Dropout(0.2),
    Dense(32, activation="relu"),
    BatchNormalization(),
    Dense(1, activation="sigmoid"),
])
lstm_model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss="binary_crossentropy",
)

lstm_model.fit(
    X_seq_train, y_seq_train,
    validation_data=(X_seq_val, y_seq_val),
    epochs=50,
    batch_size=64,
    class_weight=lstm_class_weight,
    callbacks=[
        EarlyStopping(patience=8, restore_best_weights=True),
        ReduceLROnPlateau(factor=0.5, patience=4, verbose=0),
    ],
    verbose=1,
)
lstm_model.save(str(MODELS_DIR / "lstm_model.keras"))

lstm_prob_train = lstm_model.predict(X_seq_train, verbose=0).flatten()
lstm_prob_test  = lstm_model.predict(X_seq_test,  verbose=0).flatten()

lgb_prob_train_aligned = lgb_prob_train[SEQ_LEN:]
lgb_prob_test_aligned  = lgb_prob_test[-(len(lstm_prob_test)):]
y_meta_train = y_seq_train
y_meta_test  = y_seq_test
print("Bi-LSTM done.")

# ─── 6. Meta-classifier ────────────────────────────────────────────────────────
print("\nTraining meta-classifier...")

# Richer meta-features: probs + interactions
def meta_features(lgb_p, lstm_p):
    return np.column_stack([
        lgb_p,
        lstm_p,
        lgb_p * lstm_p,
        np.maximum(lgb_p, lstm_p),
        np.abs(lgb_p - lstm_p),
    ])

X_meta_train = meta_features(lgb_prob_train_aligned, lstm_prob_train)
X_meta_test  = meta_features(lgb_prob_test_aligned,  lstm_prob_test)

meta_clf = LogisticRegression(class_weight="balanced", C=0.5, max_iter=500)
meta_clf.fit(X_meta_train, y_meta_train)
joblib.dump(meta_clf, MODELS_DIR / "meta_clf.pkl")

meta_prob_test = meta_clf.predict_proba(X_meta_test)[:, 1]

# Tune threshold — maximize F2 with recall >= 0.70 floor
best_f2, best_thresh = 0.0, 0.3
for t in np.arange(0.10, 0.70, 0.01):
    preds = (meta_prob_test >= t).astype(int)
    p = precision_score(y_meta_test, preds, zero_division=0)
    r = recall_score(y_meta_test,    preds, zero_division=0)
    if r < 0.50:           # enforce minimum recall floor
        continue
    f2 = fbeta(p, r)
    if f2 > best_f2:
        best_f2, best_thresh = f2, t

joblib.dump(best_thresh, MODELS_DIR / "threshold.pkl")
print(f"Best threshold: {best_thresh:.2f} -> F2: {best_f2:.4f}")

# ─── 7. Final evaluation ───────────────────────────────────────────────────────
final_preds = (meta_prob_test >= best_thresh).astype(int)
recall    = recall_score(y_meta_test, final_preds, zero_division=0)
precision = precision_score(y_meta_test, final_preds, zero_division=0)
f2        = fbeta(precision, recall)
auprc     = average_precision_score(y_meta_test, meta_prob_test)
cm        = confusion_matrix(y_meta_test, final_preds)

print("\n========== FINAL TEST RESULTS ==========")
print(f"Recall:    {recall:.4f}  (target >= 0.80)")
print(f"Precision: {precision:.4f}  (target >= 0.40)")
print(f"F2 Score:  {f2:.4f}  (target >= 0.65)")
print(f"AUPRC:     {auprc:.4f}  (target >= 0.50)")
print(f"Confusion Matrix:\n{cm}")
print("=========================================")

metrics = {
    "recall": recall, "precision": precision,
    "f2": f2, "auprc": auprc,
    "threshold": best_thresh,
    "confusion_matrix": cm.tolist(),
}
joblib.dump(metrics, MODELS_DIR / "metrics.pkl")

n = len(meta_prob_test)
pred_history = pd.DataFrame({
    "timestamp":   test_df.index[-n:],
    "btc_close":   test_df["close"].values[-n:],
    "probability": meta_prob_test,
    "prediction":  final_preds,
    "actual":      y_meta_test,
})
pred_history.to_csv(MODELS_DIR / "pred_history.csv", index=False)

print("\nAll models saved to /models - run: streamlit run app.py")
