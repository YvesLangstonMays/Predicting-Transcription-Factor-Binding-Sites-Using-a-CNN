import argparse
import os
import json
import yaml
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical

def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def load_data(data_dir):
    X_train = np.load(os.path.join(data_dir, "X_train.npy"))
    y_train = np.load(os.path.join(data_dir, "y_train.npy"))
    X_val = np.load(os.path.join(data_dir, "X_val.npy"))
    y_val = np.load(os.path.join(data_dir, "y_val.npy"))
    return (X_train, y_train), (X_val, y_val)

def build_model(model_type, config):
    input_shape = tuple(config["input_shape"])
    num_classes = config["num_classes"]

    if model_type == "cnn":
        model = Sequential([
            Conv1D(64, kernel_size=8, activation="relu", input_shape=input_shape),
            MaxPooling1D(pool_size=2),
            Dropout(0.3),
            Flatten(),
            Dense(64, activation="relu"),
            Dropout(0.3),
            Dense(num_classes, activation="softmax")
        ])
    elif model_type == "baseline":
        model = Sequential([
            Flatten(input_shape=input_shape),
            Dense(32, activation="relu"),
            Dense(num_classes, activation="softmax")
        ])
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model

def main():
    parser = argparse.ArgumentParser(description="Train TF binding site model.")
    parser.add_argument("--tf", required=True, help="Transcription factor name")
    parser.add_argument("--data-dir", required=True, help="Path to processed data directory")
    parser.add_argument("--output-dir", required=True, help="Where to save the model and history")
    parser.add_argument("--config", default="config.yaml", help="Path to config YAML file")
    parser.add_argument("--model-type", choices=["cnn", "baseline"], default="cnn", help="Model architecture")
    parser.add_argument("--batch-size", type=int, default=32, help="Training batch size")
    parser.add_argument("--epochs", type=int, default=30, help="Max training epochs")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    config = load_config(args.config)
    (X_train, y_train), (X_val, y_val) = load_data(args.data_dir)

    y_train_cat = to_categorical(y_train, num_classes=config["num_classes"])
    y_val_cat = to_categorical(y_val, num_classes=config["num_classes"])

    model = build_model(args.model_type, config)

    early_stop = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)

    history = model.fit(
        X_train, y_train_cat,
        validation_data=(X_val, y_val_cat),
        batch_size=args.batch_size,
        epochs=args.epochs,
        callbacks=[early_stop]
    )

    model.save(os.path.join(args.output_dir, "model.h5"))
    with open(os.path.join(args.output_dir, "history.json"), "w") as f:
        json.dump(history.history, f)

if __name__ == "__main__":
    main()