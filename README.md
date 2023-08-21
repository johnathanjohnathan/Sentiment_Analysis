# Sentiment_Analysis

## Use Case Summary

### Objective Statement
- To get insight how many hotel that has positive and negative review
- To get insight how many hotel based on their rating
- Create modeling using Machine Learning to classify hotel based on their review

### Challenges
- Large size of data, can not be maintained by excel spreadsheet.

### Methodology / Analytic Technique
- Descriptive Analysis: to find out information on current conditions based on the data that has been collected.
- Graph Analysis: provide information from the graph.
- Modelling : Machine Learning Classification.

### Expected Outcome
- Know how many hotel that has positive and negative review
- Know how many hotel based on their rating
- Create modeling using Machine Learning to classify hotel based on their review

## Business Understanding
hotel, building that provides lodging, meals, and other services to the traveling public on a commercial basis. Hotel review is one of the most important aspect when a customer want to book the room. Because with the review customer could know how the hotel is. Thus in this project has some business questions using the hotel review data :

- How many hotel that has positive and negative review?
- How many hotel based on their rating ?
- How to produce modeling using Machine Learning to predict customer churn?

## Data Understanding
### Source Data
Review Hotel Jakarta = https://github.com/faridasroful/Sentiment-Analysis-Hotel-Review/blob/main/reviewHotelJakarta.csv

The dataset has 4 columns and 210 rows

### Data Dictionary
- hotel name : the name of the hotel
- name : the name of the reviewer
- rating : the rating that the reviewer give to the hotel with number between 1 and 5
- review : the review that reviewer give to the hotel based on the rating they gave

### Data Preparation
- Programming Language : Python Version 3.9.12
- Libraries : Pandas, Numpy, Matplotlib, Seaborn, Sklearn, Imblearn

## Data Profiling
We load "reviewHotelJakarta" dataset

## Data Cleansing
There is no missing value in the dataset

## Preprocessing Data
- In this part we make sure all of the review as lower case and labeling the review into 2 class there are 1 and 0
- In this part we adding additional features - length of, and percentage of punctuations in the text
- In this part we do tokenization
- In this part we do Lemmatization and Removing Stopwords

## EDA
- Number of Hotel by Rating
- Number of Hotel by label

## Feature Extraction from Text
- Vectorizer : TF-IDF

TF-IDF stands for term frequency-inverse document frequency and it is a measure, used in the fields of information retrieval (IR) and machine learning, that can quantify the importance or relevance of string representations (words, phrases, lemmas, etc) in a document amongst a collection of documents (also known as a corpus).

## Modeling Data : Logistic Regression
Logistic Regression is a classification technique used in machine learning. It uses a logistic function to model the dependent variable. The dependent variable is dichotomous in nature, in this case is customer who churn or not. We are using deafult threshold in this model to classify the churn customer and not churn customer.

## Evaluation : Confusion Matrix
- A confusion matrix is a technique for summarizing the performance of a classification algorithm.
- Classification accuracy alone can be misleading if you have an unequal number of observations in each class or if you have more than two classes in your dataset.
- Calculating a confusion matrix can give you a better idea of what your classification model is getting right and what types of errors it is making.

### Classification Report
- From the table above we can see the precision for 0 variable is 0.62 and 1 variable is 0.71
- From the table above we can see the recall for 0 variable is 0.24 and 1 variable is 0.93
- From the table above we can see the f1- score for 0 variable is 0.34 and 1 variable is 0.80
- From the table above we can see the accuracy for this model 0.70

### Confusion Matrix
There are 4 category in this matrix:
- True Positive, the number of true positive is 39
- True Negative, the number of true negative is 5
- False Negative, the number of false negative is 3
- False Positive, the number of false positive is 16

## Building a Model with Cross Validation
Cross-Validation is a statistical method of evaluating and comparing learning algorithms by dividing data into two segments: one used to learn or train a model and the other used to validate the model.

## Hyperparameter Tuning in Logistic Regression
Hyperparameter tuning consists of finding a set of optimal hyperparameter values for a learning algorithm while applying this optimized algorithm to any data set. That combination of hyperparameters maximizes the model's performance, minimizing a predefined loss function to produce better results with fewer errors.

## Building a Model with Hyperparameter Turning
After we get the params we build a new model using best params

### Classification Report
- From the table above we can see the precision for 0 variable is 0.62 and 1 variable is 0.88
- From the table above we can see the recall for 0 variable is 0.24 and 1 variable is 0.90
- From the table above we can see the f1- score for 0 variable is 0.34 and 1 variable is 0.89
- From the table above we can see the accuracy for this model 0.0.86
### Confusion Matrix
There are 4 category in this matrix:
- True Positive, the number of true positive is 38
- True Negative, the number of true negative is 16
- False Negative, the number of false negative is 4
- False Positive, the number of false positive is 5

### Predictions
Now we train the model using tf-idf dataset, and see the score 0.8253968253968254

## Result
How many hotel that has positive and negative review?
- From the barchart we can see hotel with label 1 have value 70%. And with label 0 has 30%. From this chart we can give recommendation to increase the service of hotel with Label 0, because 30 is a high number.
How many hotel based on their rating ?
- From the result above we can see that hotel with 5 ratings have the highest value with 47.14% and hotel 4 ratings are 22.86%. And hotel with 1 rating have higher value than 2 rating with 10.95%. from this chart we can give a recommendation to increase the service of 1 rating hotel, so they can get more good review.
How to produce modeling using Machine Learning to predict customer churn?
- To produce modeling using machine learning can use Logistic Regression. Logistic Regression is a classification technique used in machine learning. It uses a logistic function to model the dependent variable. The dependent variable is dichotomous in nature, in this case is hotel who has positive or negative reviews. We are using deafult threshold in this model to classify the hotel.
After using evaluation using this model we decided to use hyperparameter tunning to increase the model performance

## Recommendation
- based on rating we can give a recommendation to increase the service of 1 rating hotel, so they can get more good review.
- based on label we can give recommendation to increase the service of hotel with Label 0, because 30% is a high number.
