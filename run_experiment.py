"""
Main experiment runner for CNN-based diabetic retinopathy classification
using the APTOS 2019 fundus image dataset.
"""

from src.config import Config
from src.dataset import load_aptos_dataset
from src.model import build_cnn_model
from src.train import train_model
from src.evaluate import evaluate_model


def main():
    cfg = Config()

    print("\n==============================")
    print("CNN Diabetic Retinopathy Classification")
    print("==============================\n")

    print("[INFO] Loading and splitting dataset...")
    data = load_aptos_dataset(cfg)

    print("[INFO] Building CNN model...")
    model = build_cnn_model(
        input_shape=data["input_shape"],
        num_classes=data["num_classes"],
        cfg=cfg
    )

    model.summary()

    print("[INFO] Training model...")
    history = train_model(
        model=model,
        x_train=data["x_train"],
        y_train=data["y_train"],
        x_valid=data["x_valid"],
        y_valid=data["y_valid"],
        cfg=cfg
    )

    print("[INFO] Evaluating model on test set...")
    evaluate_model(
        model=model,
        history=history,
        x_test=data["x_test"],
        y_test_encoded=data["y_test_encoded"],
        label_encoder=data["label_encoder"],
        cfg=cfg
    )

    model.save(cfg.MODEL_SAVE_PATH)
    print(f"[INFO] Trained model saved at: {cfg.MODEL_SAVE_PATH}")
    print("[INFO] Experiment completed successfully.")


if __name__ == "__main__":
    main()
