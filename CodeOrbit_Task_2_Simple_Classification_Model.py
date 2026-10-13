"""CodeOrbit Task 2: Simple Classification Model
Machine Learning Internship - CodeOrbit Tech
"""
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    ConfusionMatrixDisplay, precision_score, recall_score
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def main():
    iris = load_iris()
    X = pd.DataFrame(iris.data, columns=iris.feature_names)
    y = pd.Series(iris.target, name="species")

    print("=" * 55)
    print("CodeOrbit Task 2: Simple Classification Model")
    print("=" * 55)
    print("\nDataset preview:")
    print(X.head())
    print("\nDataset shape:", X.shape)
    print("\nSpecies counts:")
    print(y.map(dict(enumerate(iris.target_names))).value_counts())

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, random_state=42)
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print("\nModel Evaluation")
    print("-" * 35)
    print(f"Accuracy : {accuracy_score(y_test, y_pred):.3f}")
    print(f"Precision: {precision_score(y_test, y_pred, average='weighted', zero_division=0):.3f} (weighted)")
    print(f"Recall   : {recall_score(y_test, y_pred, average='weighted', zero_division=0):.3f} (weighted)")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=iris.target_names, zero_division=0))

    results = pd.DataFrame({
        "Actual Species": [iris.target_names[i] for i in y_test.iloc[:10]],
        "Predicted Species": [iris.target_names[i] for i in y_pred[:10]],
    })
    print("Sample Predictions:")
    print(results.to_string(index=False))

    cm = confusion_matrix(y_test, y_pred)
    display = ConfusionMatrixDisplay(cm, display_labels=iris.target_names)
    display.plot(cmap="Blues", values_format="d")
    plt.title("CodeOrbit Task 2 - Confusion Matrix")
    plt.tight_layout()
    output_file = "confusion_matrix.png"
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    print(f"\nConfusion matrix saved as: {output_file}")
    plt.show()


if __name__ == "__main__":
    main()
