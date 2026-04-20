import pandas as pd
import numpy as np
import pickle
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

FEATURES = ['temperature', 'humidity', 'signal_strength',
            'battery_level', 'packet_loss_rate', 'response_time']

def train():
    os.makedirs('model', exist_ok=True)

    df = pd.read_csv('data/wsn_dataset.csv')
    X  = df[FEATURES]
    y  = df['fault_type']

    le        = LabelEncoder()
    y_encoded = le.fit_transform(y)

    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=15,
        min_samples_split=5,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc    = accuracy_score(y_test, y_pred)
    cm     = confusion_matrix(y_test, y_pred)

    print(f"\nAccuracy : {acc*100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    # Persist artifacts
    with open('model/rf_model.pkl',       'wb') as f: pickle.dump(model,  f)
    with open('model/scaler.pkl',         'wb') as f: pickle.dump(scaler, f)
    with open('model/label_encoder.pkl',  'wb') as f: pickle.dump(le,     f)
    np.save('model/confusion_matrix.npy', cm)
    np.save('model/test_accuracy.npy',    np.array([acc]))

    print("\nModel saved -> model/")

if __name__ == '__main__':
    train()
