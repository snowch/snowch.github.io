# Alternating Least Squares (ALS) for Movie Recommendations

*Building a collaborative filtering recommendation system with Apache Spark*

---

## Overview

This tutorial demonstrates how to build a movie recommendation system using **Alternating Least Squares (ALS)**, a matrix factorization algorithm implemented in Apache Spark MLlib.

ALS is particularly effective for **collaborative filtering**, where we predict user preferences based on past behavior and the preferences of similar users. It's the same family of techniques used by Netflix, Spotify, and other platforms to recommend content.

## What You'll Learn

- **Exploratory Data Analysis**: Understanding the MovieLens dataset structure and patterns
- **Model Training**: Using Spark's ALS algorithm to learn latent factors
- **Prediction & Evaluation**: Generating recommendations and measuring accuracy

## The Tutorial Notebooks

This is a three-part series that walks through the complete recommendation pipeline:

### [Step 01 - Exploratory Analysis](step_01_exploratory_analysis.ipynb)

- Load and explore the MovieLens dataset
- Analyze rating distributions
- Understand sparsity in the ratings matrix
- Visualize user and movie statistics

### [Step 02 - Train Model](step_02_train_model.ipynb)

- Prepare data for training
- Configure ALS hyperparameters (rank, iterations, regularization)
- Train the collaborative filtering model
- Save the trained model for later use

### [Step 03 - Predict Ratings](step_03_predict_ratings.ipynb)

- Load the trained model
- Generate predictions for user-movie pairs
- Evaluate model performance with RMSE
- Make personalized movie recommendations

## The ALS Algorithm

**Alternating Least Squares** works by factorizing the user-item rating matrix into:
- **User factors**: Latent features representing user preferences
- **Item factors**: Latent features representing movie characteristics

The algorithm alternates between:
1. Fixing item factors and solving for user factors
2. Fixing user factors and solving for item factors

This iterative process minimizes the reconstruction error and learns meaningful latent representations.

## Dataset

The notebooks use the [MovieLens dataset](https://grouplens.org/datasets/movielens/), which contains:
- Ratings from users on movies (scale: 1-5 stars)
- Movie metadata (titles, genres)
- User demographics (in some versions)

## Technologies

- **Apache Spark**: Distributed computing framework
- **PySpark**: Python API for Spark
- **MLlib**: Spark's machine learning library
- **Jupyter Notebooks**: Interactive development environment

## Source Code

The complete source code, including setup instructions and additional resources, is available on GitHub:

**🔗 [Movie Recommender Demo Repository](https://github.com/snowch/movie-recommender-demo/)**

## Prerequisites

To run these notebooks, you'll need:
- Apache Spark installed (version 2.4+)
- PySpark Python package
- Jupyter Notebook or JupyterLab
- Basic familiarity with Python and Spark DataFrames

See the [GitHub repository](https://github.com/snowch/movie-recommender-demo/) for detailed setup instructions.

## Next Steps

After completing this tutorial, you can:
- Experiment with different ALS hyperparameters
- Try implicit feedback models for binary data (clicks, views)
- Implement hybrid recommenders combining content and collaborative filtering
- Deploy the model as a REST API service
- Scale to larger datasets using Spark clusters

---

*This tutorial series originally created for demonstrating Spark MLlib capabilities for recommendation systems.*
