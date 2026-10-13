import time
import tracemalloc
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

def evaluate_model(name, model, X_train, X_test, y_train, y_test):
    tracemalloc.start()
    st = time.perf_counter() 
    
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]
    exec_time = (time.perf_counter() - st) * 1000.0
    _, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    accuracy = accuracy_score(y_test, preds)
    precision = precision_score(y_test, preds)
    recall = recall_score(y_test, preds)
    f1 = f1_score(y_test, preds)
    roc_auc = roc_auc_score(y_test, probs)
    
    return {"name": name, "exec_time": exec_time, 
            "peak_mem": peak_mem / 1024.0, "accuracy_score": accuracy, 
            "precision_score": precision, "recall_score": recall,
            "f1_score": f1, "roc_auc_score": roc_auc}
            
            
if __name__ == "__main__":
    from src.data_loader import load_data, prepare_data
    from src.models import get_random_forest

    df = load_data("data/telemetry_dataset.csv")
    X_train, X_test, y_train, y_test = prepare_data(df)
    print(evaluate_model("Random Forest", get_random_forest(), X_train, X_test, y_train, y_test))