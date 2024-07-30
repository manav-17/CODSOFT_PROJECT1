# CODSOFT_PROJECT1

TITANIC SURVIVAL PREDICTION

AIM - Use the Titanic dataset to build a model that predicts whether a
passenger on the Titanic survived or not.

TECHNOLOGIES - 

a.Pandas: A powerful library for data manipulation and analysis in Python. It is used to load the Titanic dataset, handle missing values, and preprocess the data.

b.NumPy: A library for numerical computing in Python, often used for array operations. It is indirectly used through Pandas and for handling numerical data in the code.

c.Scikit-learn (sklearn): A widely used machine learning library in Python. The following components from Scikit-learn are utilized in your code:

1.train_test_split: A function to split the dataset into training and testing sets.

2.LabelEncoder: A class used to convert categorical variables into numerical format.

3.SimpleImputer: A class used to fill missing values in the dataset.

4.RandomForestClassifier: A machine learning algorithm used for classification tasks.

5.accuracy_score, classification_report, and confusion_matrix: Functions for evaluating the performance of the model.

d.Machine Learning: The code demonstrates a typical workflow for a binary classification task, specifically predicting survival on the Titanic based on various features.

e.Data Preprocessing: The code includes several preprocessing steps:
1.Dropping unnecessary columns.
2.Filling missing values using the SimpleImputer.
3.Encoding categorical variables using LabelEncoder.

f.Model Training and Evaluation: The code trains a Random Forest model on the training data, makes predictions on the test data, and evaluates the model using accuracy, classification report, and confusion matrix.

g.Matplotlib: A plotting library for creating static, interactive, and animated visualizations in Python. It is used to create histograms, bar plots, and other visualizations in your code.

h.Seaborn: A data visualization library based on Matplotlib that provides a high-level interface for drawing attractive statistical graphics. It is used for creating count plots and heatmaps in your code.

i.Data Visualization: The code includes multiple plots to visualize data distributions, model performance, and feature importance. This is an essential step in understanding the results and insights from the model.
