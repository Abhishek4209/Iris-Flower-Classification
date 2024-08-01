# Iris Flower Classification Model

This project demonstrates the classification of Iris flowers using various machine learning algorithms. The Iris dataset is a classic dataset in machine learning and consists of 150 samples of Iris flowers, with 4 features: sepal length, sepal width, petal length, and petal width. The goal is to classify the flowers into one of three species: Iris setosa, Iris versicolor, or Iris virginica.

## Algorithms Used

The following machine learning algorithms are implemented in this project:

- Logistic Regression
- Support Vector Classifier (SVC)
- Decision Tree Classifier
- Random Forest Classifier
- Gradient Boosting Classifier
- AdaBoost Classifier
- K-Nearest Neighbors Classifier

## Requirements

- Python 3.12
- scikit-learn
- pandas
- numpy
- matplotlib (optional, for visualization)

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/iris-flower-classification.git
    ```
2. Navigate to the project directory:
    ```bash
    cd iris-flower-classification
    ```
3. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. Load the dataset:
    ```python
    from sklearn.datasets import load_iris
    import pandas as pd

    iris = load_iris()
    X = pd.DataFrame(iris.data, columns=iris.feature_names)
    y = pd.Series(iris.target)
    ```

2. Initialize the classifiers:
    ```python
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
    from sklearn.neighbors import KNeighborsClassifier

    classifiers = {
        'LogisticRegression': LogisticRegression(),
        'SVC': SVC(),
        'DecisionTreeClassifier': DecisionTreeClassifier(),
        'RandomForestClassifier': RandomForestClassifier(),
        'GradientBoostingClassifier': GradientBoostingClassifier(),
        'AdaBoostClassifier': AdaBoostClassifier(),
        'KNeighborsClassifier': KNeighborsClassifier(algorithm='auto', p=2)
    }
    ```

3. Train and evaluate the classifiers:
    ```python
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    for name, clf in classifiers.items():
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"{name} accuracy: {accuracy:.2f}")
    ```

## Results

After training and evaluating the classifiers, you can expect output similar to:
