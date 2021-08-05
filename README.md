# W207_movies

## Table of Contents  
- [Overview](#overview)  
- [EDA Notebooks](#eda)
- [Content Recommender Model Notebooks](#content-recommender-model-notebooks)
    - [Data Pre-Processing Notebooks](#content-processing-notebooks)
    - [Model Notebook](#content-model-notebook)
    - [Order and Instructions](#content-notebook-order)
- [Collaborative Filtering Model Notebooks](#collaborative-filtering-model-notebooks)
    - [Data Pre-Processing Notebook](#collaborative-processing-notebook)
    - [Model Notebooks](#collaborative-model-notebooks)
    - [Order and Instructions](#collaborative-notebook-order)

## Overview <a name="overview"></a>
In our project, we seek to evaluate the hit rate of two common recommender models, a content-based recommender model and a collaborative filtering recommender model, and determine the extent to which we can improve recommendation diversity without impacting hit rate performance.

## EDA Notebooks <a name="eda"></a>
1. `movies_eda.ipynb`
2. `ratings_eda.ipynb`

## Content Recommender Model Notebooks <a name="content-recommender-model-notebooks"></a>
### Data Pre-Processing Notebooks <a name="content-processing-notebooks"></a>
#### `credits_keywords_processing.ipynb`

Extracts the cast names, crew names, and keywords from the credits.csv and keywords.csv respective files. Creates a new dataframe, credits_keywords.csv, with columns of these extracted values as text strings.

#### `movies_process_to_temp.ipynb`

Clean the original columns in the movies_metadata.csv file and add in the extracted text fields from credits_keywords.csv. Write this dataframe to a new csv file, movies_temp.csv.

#### `ratings_processing.ipynb`

Add Imdb field into the ratings.csv file to enable filtering of both ratings.csv and movies_temp.csv, binarize ratings (convert sores <4 to 0 and scores >= 4 to 1), drop movies in both ratings.csv and movies_temp.csv that are not in the other dataset, and filter ratings.csv to only include ratings from users who provided 30 or more ratings. Split this filtered ratings.csv file into the following files: test (the  most recent rating for filtered users), dev (the second-to-last rating for filtered users), and train (all remaining ratings for filtered users). Write the test, dev, and train dataframes to new csv files. Write the filtered dataframe of movies_temp to a new csv file, movies_final.csv.

### Model Notebook <a name="content-model-notebook"></a>
#### `content_based.ipynb`

Convert all movies_final columns to numeric form for classifier testing and drop unneeded columns, and convert the dataframe to a CSR sparse matrix. Evaluate a baseline classifier that assigns a random probability to each label and the following classifiers (Cosine Similarity, Bernoulli Naive Bayes, Random Forest, SVM, K-Nearest Neighbors, Logistic Regression, Gaussian Mixture Model, Ensemble Model) on a random sample of the dev set and compare the positive case hit rates and f1 scores. For the Bernoulli Naive Bayes, Random Forest, SVM, K-Nearest Neighbors, Logistic Regression, Gaussian Mixture Models, iterate over common parameters to test changes in positive case hit rate and f1-score. For the classifier with the highest positive case hit rate, keep the parameters for which it received its highest f1-score, and evaluate positive case hit rate, f1-score, and novelty score on a random sample of test users and ratings.

### Order and Instructions <a name="#content-notebook-order"></a>

1. `credits_keywords_processing.ipynb`

   Replace the path to the credits.csv file and the path to the path to the keywords.csv file with the respective paths to these files   within the raw_data folder on your local machine.
   
2. `movies_process_to_temp.ipynb`

   Replace the path to the movies_metadata.csv file with the path to this file within raw_data on your local machine. For credits_keywords.csv, replace the path with the path to this file within the clean_data folder on your local machine.
   
3. `ratings_processing.ipynb`

   Replace the path_raw_data and path_clean_data paths with the paths to the raw_data and clean_data folders, respectively, on your local machine.
   
4. `content_based.ipynb`
   
   Replace the path_raw_data and path_clean_data paths with the paths to the raw_data and clean_data folders, respectively, on your local machine.


## Collaborative Filtering Model Notebooks <a name="collaborative-filtering-model-notebooks"></a>

### Data Pre-Processing Notebooks <a name="collaborative-processing-notebook"></a>
#### `ratings_cf_preprocessing.ipynb`
- Filters ratings to those users who have 30+ ratings
- Conducts Train/Dev/Test split based on leave one out approach 
- Makes evaluation pools of 100 movies per user for dev and test data
- Writes pre-processed data 

### Model Notebooks <a name="collaborative-model-notebooks"></a>
#### `matrix_factorization_models.ipynb`
- Fits SVD, SVD++ and NMF models
- Evaluates models based on RMSE and customly define HR@10 metric
- Introduces diversity into models rankings using Novelty as a deversity metric

#### `deep_learning_models.ipynb`
- Fits a Keras embedding model using various embedding depths
- Evaluates models based on RMSE and customly define HR@10 metric

### Order and Instructions <a name="#collaborative-notebook-order"></a>

1. `ratings_cf_preprocessing.ipynb`

   Replace the `path_raw_data` and `path_clean_data` paths with the paths to the raw_data and clean_data folders, respectively, on your local machine.
   
2. `matrix_factorization_models.ipynb`
   
   Replace the `path_raw_data` and `path_clean_data` paths with the paths to the raw_data and clean_data folders, respectively, on your local machine.

3. `deep_learning_models.ipynb`
   
   Replace the `path_clean_data` paths with the path clean_data folder on your local machine.
