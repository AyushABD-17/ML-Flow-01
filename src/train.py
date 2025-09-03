import mlflow
import mlflow.sklearn
import hydra
from omegaconf import DictConfig
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


@hydra.main(version_base=None, config_path="../configs", config_name="training")
def main(cfg: DictConfig):
    
    dataset_path = Path(hydra.utils.get_original_cwd()) / cfg.dataset.path
    data = pd.read_csv(dataset_path)

    X, y = data.drop(cfg.dataset.target, axis=1), data[cfg.dataset.target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=cfg.training.test_size,
        random_state=cfg.training.random_state
    )
    

# Make sure all runs go to project-root/mlruns
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    

    mlflow.set_experiment("models experiment")

    with mlflow.start_run():
        if cfg.model.type == "logistic_regression":
            model = LogisticRegression(
                C=cfg.model.params.C,
                max_iter=cfg.model.params.max_iter
            )
        else:
            raise ValueError(f"Unsupported model: {cfg.model.type}")

        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        acc = accuracy_score(y_test, preds)

        
        mlflow.log_params(cfg.model.params)
        mlflow.log_metric("accuracy", acc)

        
        mlflow.sklearn.log_model(model, "model")

        
        plt.figure()
        plt.hist(preds)
        plt.title("Prediction Distribution")
        plot_path = "pred_hist.png"
        plt.savefig(plot_path)
        mlflow.log_artifact(plot_path)

        print(f"Accuracy: {acc}")


if __name__ == "__main__":
    main()
