import pandas as pd
from sklearn.datasets import make_classification


def generate_data(
    n_samples: int = 1000, n_features: int = 10, random_state: int = 42
):
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=7,
        n_redundant=3,
        random_state=random_state,
    )
    feature_names = [f"feature_{i}" for i in range(n_features)]
    df = pd.DataFrame(X, columns=feature_names)
    df["target"] = y
    return df


if __name__ == "__main__":
    df = generate_data()
    df.to_csv("data/dataset.csv", index=False)
    print(f"Dataset generado exitosamente con dimensiones: {df.shape}")