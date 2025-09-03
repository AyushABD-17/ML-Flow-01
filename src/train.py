import mlflow
import mlflow.sklearn
import hydra
from omegaconf import DictConfig
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_curve, auc, confusion_matrix
from sklearn.linear_model import LogisticRegression
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# import xgboost
from xgboost import XGBClassifier


@hydra.main(version_base=None, config_path="../configs", config_name="training")
def main(cfg: DictConfig):
    # Load dataset
    dataset_path = Path(hydra.utils.get_original_cwd()) / cfg.dataset.path
    data = pd.read_csv(dataset_path)

    X, y = data.drop(cfg.dataset.target, axis=1), data[cfg.dataset.target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=cfg.training.test_size,
        random_state=cfg.training.random_state
    )

    # MLflow setup
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("models_experiment")

    with mlflow.start_run():
        if cfg.model.type == "logistic_regression":
            model = LogisticRegression(
                C=cfg.model.params.C,
                max_iter=cfg.model.params.max_iter
            )
        elif cfg.model.type == "xgboost":
            model = XGBClassifier(
                n_estimators=cfg.model.params.n_estimators,
                learning_rate=cfg.model.params.learning_rate,
                max_depth=cfg.model.params.max_depth,
                subsample=cfg.model.params.subsample,
                colsample_bytree=cfg.model.params.colsample_bytree,
                random_state=cfg.model.params.random_state,
                use_label_encoder=False,
                eval_metric="logloss"
            )
        else:
            raise ValueError(f"Unsupported model: {cfg.model.type}")

        # Train
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        acc = accuracy_score(y_test, preds)

        # Log params & metrics
        mlflow.log_params(cfg.model.params)
        mlflow.log_metric("accuracy", acc)

        # Save model
        mlflow.sklearn.log_model(model, "model")

        # --- ARTIFACT 1: Prediction Histogram ---
        plt.figure()
        plt.hist(preds)
        plt.title("Prediction Distribution")
        plot_path = "pred_hist.png"
        plt.savefig(plot_path)
        mlflow.log_artifact(plot_path)

        
        if len(set(y_test)) == 2:  # Only for binary classification
            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(X_test)[:, 1]
                fpr, tpr, _ = roc_curve(y_test, probs)
                roc_auc = auc(fpr, tpr)

                plt.figure()
                plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
                plt.plot([0, 1], [0, 1], "r--")
                plt.xlabel("False Positive Rate")
                plt.ylabel("True Positive Rate")
                plt.title("ROC Curve")
                plt.legend()
                roc_path = "roc_curve.png"
                plt.savefig(roc_path)
                mlflow.log_artifact(roc_path)

                mlflow.log_metric("roc_auc", roc_auc)

        
        cm = confusion_matrix(y_test, preds)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title("Confusion Matrix")
        cm_path = "confusion_matrix.png"
        plt.savefig(cm_path)
        mlflow.log_artifact(cm_path)

        print(f"Accuracy: {acc}")


if __name__ == "__main__":
    main()
