# Classification of Patient's Condition & Drug Recommendation

## Project Overview

Discover the future of healthcare with our Flask web application, an innovative platform that classifies medical conditions from symptoms and recommends top-rated drugs. Leveraging advanced analytics, this project aims to assist healthcare professionals in staying updated with the latest treatments and medications by analyzing user reviews and predicting medical conditions.


## Introduction

In this project, we create a recommendation system using Natural Language Processing (NLP) and machine learning techniques to predict medical conditions based on textual reviews and suggest the top 3 drugs with high ratings. We utilize the Drug Reviews dataset from the UCI ML Repository to build and evaluate models, and provide a web application interface using Flask.

## Software and Tools

1. [Github Account](https://github.com) - For version control and collaboration.
2. [VSCode IDE](https://code.visualstudio.com/) - For coding and development.
3. Python Libraries:
   - `Flask` for the web application.
   - `pandas` for data manipulation.
   - `scikit-learn` for machine learning.
   - `nltk` for text processing.
   - `BeautifulSoup` for HTML parsing.
   - `pickle` for model serialization.

## Project Details

### Dataset

The dataset consists of 161,297 user reviews with 7 columns:
- Drug names
- Medical conditions
- Textual reviews
- Ratings given by users
- Date of the review
- Total number of useful counts

We focus on four key medical conditions:
1. Birth Control
2. Depression
3. Pain
4. Anxiety

### Data Preprocessing

1. **Load the Dataset**: Read the dataset files and combine them.
2. **Clean the Data**:
   - Remove irrelevant columns.
   - Handle missing values.
   - Filter for selected medical conditions.
3. **Text Preprocessing**:
   - Remove HTML tags.
   - Convert text to lowercase and remove non-alphabetic characters.
   - Remove stop words and apply lemmatization.
4. **Feature Extraction**:
   - Use TF-IDF vectorization with both unigrams and bigrams.
   - Set `max_df=0.8` to exclude terms that appear in more than 80% of the documents.

### Model Building

1. **Train-Test Split**: Use 80% of the data for training and 20% for testing.
2. **Model Training**:
   - **Multinomial Naive Bayes**: Effective for textual data.
   - **Passive-Aggressive Classifier**: Selected for better accuracy.
3. **Evaluation**:
   - Measure accuracy and confusion matrix.
   - The Passive-Aggressive Classifier was chosen based on performance.
  

In the model building phase, we used two primary algorithms:

1. **Multinomial Naive Bayes Algorithm**: This algorithm is well-suited for textual data and often performs effectively in text classification tasks.

2. **Passive-Aggressive Classifier**: This algorithm showed superior performance compared to the Multinomial Naive Bayes Algorithm in terms of accuracy.

Based on the evaluation results, the Passive-Aggressive Classifier demonstrated higher accuracy and thus, was selected as the primary model for predicting medical conditions.


### Drug Recommendation

Based on the predicted medical condition, recommend the top 3 drugs from the dataset with:
- Ratings >= 9
- Useful count >= 100


