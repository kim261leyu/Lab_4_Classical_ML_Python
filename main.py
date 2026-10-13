import pandas as pd
import matplotlib.pyplot as plt
from tabulate import tabulate

from src.data_loader import load_data, prepare_data
from src.models import get_random_forest, get_gradient_boosting
from src.metrics import evaluate_model

from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay


def main():
    df = load_data("data/telemetry_dataset.csv")
    X_train, X_test, y_train, y_test = prepare_data(df)
    
    rf = get_random_forest()
    gb = get_gradient_boosting()

    res_rf = evaluate_model("Random Forest", rf, X_train, X_test, y_train, y_test)
    res_gb = evaluate_model("Gradient Boosting", gb, X_train, X_test, y_train, y_test)


    print(pd.DataFrame([res_rf, res_gb]).to_markdown(index=False))

  
    ConfusionMatrixDisplay.from_estimator(rf, X_test, y_test)
    plt.title("Random Forest")
    plt.savefig("plots/cm_rf.png")

    ConfusionMatrixDisplay.from_estimator(gb, X_test, y_test)
    plt.title("Gradient Boosting")
    plt.savefig("plots/cm_gb.png")
    
    fig, ax = plt.subplots()
    RocCurveDisplay.from_estimator(rf, X_test, y_test, ax=ax, name="Random Forest")
    RocCurveDisplay.from_estimator(gb, X_test, y_test, ax=ax, name="Gradient Boosting")
    ax.plot([0, 1], [0, 1], linestyle="--", label="Random guess")
    ax.legend()
    plt.savefig("plots/roc.png")

if __name__ == "__main__":
    main()