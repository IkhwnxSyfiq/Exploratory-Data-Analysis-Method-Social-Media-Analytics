Mini Project: Social Media Analytics for Brand Sentiment on Twitter
https://img.shields.io/badge/Python-3.x-blue https://img.shields.io/badge/Scikit--learn-ML_Library-orange https://img.shields.io/badge/NLP-Sentiment%2520Analysis-green

This repository contains the code, report, and presentation for a university mini-project on Big Data Analytics. The project focuses on building a machine learning pipeline to perform sentiment analysis on tweets related to major brands, with a special focus on Tesla. The goal is to classify tweets into Positive, Negative, Neutral, or Irrelevant categories to monitor public opinion and brand perception.

📄 Project Overview
This project was completed as part of the Big Data Analytics course at Universiti Kuala Lumpur (UniKL British Malaysian Institute). The objective was to apply data preprocessing, exploratory data analysis (EDA), and machine learning techniques to a real-world big data problem.

👥 Team Members
Name	Student ID
Muhammad Ikhwan Syafiq bin Norsham	51221221125
Muhammad Waiz bin Nor Kamal	51221221053
Ahmad Syahmi bin Ahmad Fauzi	51221221003
🎯 Objectives
To create a dataset of tweets about major brands (e.g., Tesla, Google, Apple).

To clean and preprocess the raw Twitter data.

To perform exploratory data analysis (EDA) to understand data distribution and characteristics.

To train and evaluate a machine learning model to accurately classify tweet sentiment.

To deploy the model for automatic sentiment extraction from new tweets.

📊 Dataset
The project uses two CSV files:

twitter_training.csv: The primary dataset for training and testing the model.

twitter_validation.csv: A separate dataset used for final model validation.

Dataset Features:

Tweet_ID: Unique identifier for the tweet.

Entity: The brand or entity the tweet is about (e.g., Tesla, Google).

Sentiment: The labelled sentiment (Positive, Negative, Neutral, Irrelevant).

Tweet_content: The raw text of the tweet.

🔧 Methodology & Implementation
The project follows a standard machine learning pipeline:

1. Data Preprocessing & Cleaning
Removed duplicates and NaN values.

Removed URLs and emojis using Regular Expressions (Regex).

Tokenisation and Lemmatisation performed using spaCy to break text into words and reduce them to their base form.

Removed stop words and punctuation.

2. Exploratory Data Analysis (EDA)
Analysed the distribution of sentiment labels across the dataset.

Visualised the length of tweets for each sentiment category using KDE plots.

Compared the volume of tweets per brand and per sentiment using bar charts.

3. Feature Engineering
Text data was transformed into numerical features using TF-IDF (Term Frequency-Inverse Document Frequency) vectorisation from scikit-learn.

4. Model Training
A Random Forest Classifier was chosen for its effectiveness in handling high-dimensional data and robustness to overfitting.

The data was split into a training set (80%) and a testing set (20%).

5. Model Evaluation
The model was evaluated using standard metrics: Precision, Recall, F1-Score, and Accuracy.

Performance was assessed on both the test split and a separate validation dataset.

📈 Results
The trained Random Forest model achieved excellent performance:

On Test Data:

Accuracy: 0.91

The detailed classification report showed high F1-scores across all sentiment classes.

On Validation Data:

Accuracy: 0.95

The model generalised very well to unseen data, demonstrating its robustness.

Sample Prediction:

Input Text: "Well, that's not helping to reassure me my data is safe with @google"

Expected Sentiment: Negative

Predicted Sentiment: Negative ✅

✅ Conclusion
The project successfully developed a highly accurate sentiment analysis model for Twitter data. The Random Forest classifier, coupled with careful TF-IDF feature extraction and thorough data preprocessing, proved to be a powerful combination for this task. The model's high performance on a separate validation set confirms its ability to generalize and its potential for real-world application in social media monitoring and brand management.

🔮 Future Work
Deployment: Integrate the model into a web application or a real-time Twitter stream using the Twitter API.

Advanced Models: Experiment with deep learning models, such as LSTMs or BERT, for potentially higher accuracy in nuanced language.

Multi-Platform Analysis: Extend the analysis to other social media platforms like Facebook or Instagram.

Aspect-Based Sentiment Analysis: Identify not just overall sentiment, but also the specific product features or services people are talking about.
