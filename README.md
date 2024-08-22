# CS236299 Project 1: Text Classification - Answer Type Prediction

This repository contains the implementation of Project Segment 1 for the CS236299 course. The focus of this project is on text classification, specifically predicting the answer type of queries in the ATIS (Airline Travel Information System) dataset.

## Table of Contents

- [Project Overview](#project-overview)
- [Goals](#goals)
- [Implementation Details](#implementation-details)
- [Setup](#setup)
- [Usage](#usage)
- [Evaluation](#evaluation)
- [Discussion](#discussion)
- [License](#license)

## Project Overview

The primary objective of this project is to classify natural language queries into their respective answer types. Given a query, the task is to predict the type of answer it seeks, such as a flight ID, fare information, or city name. The dataset used for this task is the ATIS dataset, which contains annotated queries related to airline travel.

## Goals

1. Implement a **Majority Baseline** for text classification.
2. Implement a **Naive Bayes Classifier** for text classification.
3. Implement a **Logistic Regression Classifier** for text classification.
4. Implement a **Multilayer Perceptron (MLP) Classifier** for text classification.
5. Compare the performance of these classifiers and discuss their strengths and weaknesses.

## Implementation Details

### 1. Majority Baseline

- **Majority Baseline**: A simple classifier that always predicts the most frequent class in the training data. This serves as a baseline to compare the performance of more complex models.

### 2. Naive Bayes Classifier

- **Naive Bayes Method**: Implements the Naive Bayes classification algorithm, which models the probability of each class given the input features (words in the query). Add-δ smoothing is used to handle zero probabilities.
- **Two Conceptions**: The Naive Bayes model is implemented using two approaches—index representation and bag-of-words representation.

### 3. Logistic Regression Classifier

- **Logistic Regression**: A linear model that predicts the probability of each class using a linear combination of input features followed by a softmax function. The model is trained using gradient descent to minimize the cross-entropy loss.

### 4. Multilayer Perceptron (MLP) Classifier

- **MLP Architecture**: Extends the logistic regression model by adding a hidden layer with a nonlinear activation function (sigmoid). The model consists of two layers: an input-to-hidden layer and a hidden-to-output layer.
- **Training**: The MLP is trained using the Adam optimizer and cross-entropy loss, similar to logistic regression.

### 5. Comparison of Models

- **Performance Evaluation**: The models are compared based on their accuracy on the test set. The strengths and weaknesses of each model are discussed, including their handling of complex queries and computational efficiency.

## Setup

To set up the project, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/SamiHam162/NLP236299-Project1.git
   cd NLP236299-Project1
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
3. Download the necessary datasets and scripts as outlined in the `project1_classification_solution.ipynb` file.

## Usage

1. **Majority Baseline**:
   - Implement and evaluate the majority baseline to establish a performance benchmark.
2. **Naive Bayes Classifier**:
   - Train and evaluate the Naive Bayes model on the ATIS dataset.
3. **Logistic Regression Classifier**:
   - Train and evaluate the logistic regression model on the ATIS dataset.
4. **Multilayer Perceptron (MLP) Classifier**:
   - Train and evaluate the MLP model on the ATIS dataset.
5. **Comparison**:
   - Compare the performance of the models and analyze their results.

## Evaluation

The evaluation of the models is based on their accuracy in predicting the correct answer type for the test queries. The results are compared to the baseline to determine the effectiveness of each model.

## Discussion

The final section of the project involves a discussion on the strengths and weaknesses of the different classification models implemented, particularly in the context of the ATIS dataset. The discussion also includes insights into the impact of different model architectures on classification accuracy.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
