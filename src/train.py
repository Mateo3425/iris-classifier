import argparse
import os

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
 
def get_args():
    parser = argparse.ArgumentParser(description="Train Iris decision tree and save confusion matrix.")
    parser.add_argument("--test-size", type=float, default=0.2, help="Fraction for test split (0–1).")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed for splitting/model.")
    parser.add_argument("--outdir", type=str, default="outputs", help="Folder to save outputs.")
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    os.makedirs(args.outdir, exist_ok=True)
    # Next step: load data and split

    # Load the Iris dataset
    iris = load_iris()
    X, y = iris.data, iris.target

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state, stratify=y
    )

    # Train the Decision Tree model
    model = DecisionTreeClassifier(random_state=args.random_state)
    model.fit(X_train, y_train)

        # Make predictions
    y_pred = model.predict(X_test)

    # Evaluate accuracy
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc:.2f}")

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=iris.target_names)
    disp.plot(cmap="Blues")

    # Save figure
    fig_path = os.path.join(args.outdir, "confusion_matrix.png")
    plt.savefig(fig_path)
    plt.close()
    print(f"Confusion matrix saved to {fig_path}")
