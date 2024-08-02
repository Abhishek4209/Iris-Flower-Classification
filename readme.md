
# Iris Flower Classification Model

![Iris Flower Classification](image.png)
This project implements several machine learning models to classify the Iris flower dataset. The models used include Logistic Regression, Support Vector Classifier (SVC), Decision Tree Classifier, Random Forest Classifier, Gradient Boosting Classifier, AdaBoost Classifier, and K-Neighbors Classifier.

## Deployment URL:
```bash
https://iris-flower-classification-zqcs.onrender.com

```


## Table of Contents

- [Introduction](#introduction)
- [Dataset](#dataset)
- [Models Used](#models-used)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Contributing](#contributing)
- [Contact](#contact)

## Introduction

The Iris flower dataset is a classic dataset used in pattern recognition. It consists of 150 samples from each of three species of Iris flowers (Iris setosa, Iris virginica, and Iris versicolor). Four features were measured from each sample: the lengths and the widths of the sepals and petals.

## Dataset

The dataset used in this project is the Iris flower dataset which is available in the `https://www.kaggle.com/datasets/arshid/iris-flower-dataset`

## Models Used

The following machine learning models have been implemented and compared:

- Logistic Regression: `LogisticRegression()`
- Support Vector Classifier (SVC): `SVC()`
- Decision Tree Classifier: `DecisionTreeClassifier()`
- Random Forest Classifier: `RandomForestClassifier()`
- Gradient Boosting Classifier: `GradientBoostingClassifier()`
- AdaBoost Classifier: `AdaBoostClassifier()`
- K-Neighbors Classifier: `KNeighborsClassifier(algorithm='auto', p=2)`

## Installation

To install the necessary packages and dependencies, run the following command:

```bash
pip install -r requirements.txt
```

Ensure you are using Python version 3.12.

## Usage

To train and evaluate the models, run the `app.py` script:

```bash
python app.py.py
```

This script will load the dataset, preprocess the data, train each model, and output the evaluation metrics for each model.

## Results

The results of the models will be displayed in the Logs folder, showing the accuracy, precision, recall, and F1-score for each model.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License.

## Contact

Author: Abhishek Upadhyay  
Email: [abhishekupadhyay9336@gmail.com](mailto:abhishekupadhyay9336@gmail.com)
```
