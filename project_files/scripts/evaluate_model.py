
import argparse
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, roc_auc_score

def load_test_data(test_data_path):
    data = np.load(test_data_path)
    X_test = data['X']
    y_test = data['y']
    return X_test, y_test

def evaluate_model(model_path, test_data_path):
    print(f"Loading model from: {model_path}")
    model = tf.keras.models.load_model(model_path)

    print(f"Loading test data from: {test_data_path}")
    X_test, y_test = load_test_data(test_data_path)

    print("Running predictions...")
    y_pred_probs = model.predict(X_test).flatten()
    y_pred = (y_pred_probs >= 0.5).astype(int)

    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_probs)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"ROC AUC: {auc:.4f}")

def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained model on test data.")
    parser.add_argument('--model-path', type=str, required=True, help='Path to the trained model (.h5)')
    parser.add_argument('--test-data', type=str, required=True, help='Path to the test data (.npz)')

    args = parser.parse_args()

    evaluate_model(args.model_path, args.test_data)

if __name__ == "__main__":
    main()