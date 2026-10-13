import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_data(path):
    data = pd.read_csv(path)
    return data
    
def prepare_data(df):
    X = df.drop(columns=["OOM_Risk"])
    y = df["OOM_Risk"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    return X_train, X_test, y_train, y_test
    
if __name__ == "__main__":
    df = load_data("data/telemetry_dataset.csv")
    print(df.shape)

    X_train, X_test, y_train, y_test = prepare_data(df)
    print(X_train.shape, X_test.shape)
    print(y_train.value_counts(normalize=True))
    print(y_test.value_counts(normalize=True))
    print(X_train.mean(axis=0).round(3))