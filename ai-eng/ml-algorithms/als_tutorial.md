---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.18.1
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# Alternating Least Squares (ALS) for Movie Recommendations

*Building a collaborative filtering recommendation system with matrix factorization*

---

## Overview

This tutorial demonstrates how to build a movie recommendation system using **Alternating Least Squares (ALS)**, a matrix factorization algorithm for collaborative filtering. ALS gained attention during the Netflix Prize era and still provides a clear, interpretable baseline, even though many production systems now favor more sophisticated hybrid or deep-learning approaches.

It is adapted from an older tutorial I wrote around a decade ago on creating a movie recommender with Apache Spark on IBM Bluemix (see [movie-recommender-demo](https://github.com/snowch/movie-recommender-demo)), updated here for modern Python workflows and portability.

We'll explore a MovieLens-style dataset (with some interesting rating biases), visualize the sparsity problem in recommendation systems, and understand how ALS factorizes the user-item rating matrix to make predictions.

```{code-cell} ipython3
:tags: [remove-input, remove-output]

import logging
import warnings

logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", message="Matplotlib is building the font cache*")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.sparse import csr_matrix
from sklearn.metrics import mean_squared_error

# Configure matplotlib
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
```

---

## Part 1: The Dataset and Sparsity Problem

We'll use the same MovieLens-style ratings dataset from the original Spark tutorial. It includes some interesting biases in how users rate movies, which makes the sparsity patterns more visible.

```{code-cell} ipython3

from pathlib import Path
from urllib.request import urlretrieve

data_url = "https://raw.githubusercontent.com/snowch/movie-recommender-demo/master/web_app/data/ratings.dat"
data_path = Path("data/ratings.dat")
data_path.parent.mkdir(parents=True, exist_ok=True)

if not data_path.exists():
    urlretrieve(data_url, data_path)

ratings = pd.read_csv(
    data_path,
    sep="::",
    engine="python",
    names=["user", "product", "rating", "timestamp"]
).drop(columns=["timestamp"])

print(f"Total ratings: {len(ratings)}")
print(f"Number of users: {ratings['user'].nunique()}")
print(f"Number of movies: {ratings['product'].nunique()}")
print(f"\nRating distribution:\n{ratings['rating'].value_counts().sort_index()}")
```

### Visualise the ratings matrix using a subset of the data

Let's take a subset of the data.

```{code-cell} ipython3

ratings_subset = ratings.query("user < 20 and product < 20").copy()
ratings_subset
```

Separate the x (user) values and also the y (movie) values for matplotlib.
Also normalise the rating value so that it is between 0 and 1. This is required for coloring the markers.

```{code-cell} ipython3

user = ratings_subset["user"].astype(int)
movie = ratings_subset["product"].astype(int)

min_r = ratings_subset["rating"].min()
max_r = ratings_subset["rating"].max()

def normalise(x):
    rating = (x - min_r) / (max_r - min_r)
    return float(rating)

ratingN = ratings_subset["rating"].apply(normalise)
```

We can now plot the sparse matrix of ratings for this subset of users and movies.

```{code-cell} ipython3

%matplotlib inline
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

min_user = ratings_subset["user"].min()
max_user = ratings_subset["user"].max()
min_movie = ratings_subset["product"].min()
max_movie = ratings_subset["product"].max()

width = 5
height = 5
plt.figure(figsize=(width, height))
plt.ylim([min_user-1, max_user+1])
plt.xlim([min_movie-1, max_movie+1])
plt.yticks(np.arange(min_user-1, max_user+1, 1))
plt.xticks(np.arange(min_movie-1, max_movie+1, 1))
plt.xlabel('Movie ID')
plt.ylabel('User ID')
plt.title('Movie Ratings')

ax = plt.gca()
ax.patch.set_facecolor('#898787') # dark grey background

colors = plt.cm.YlOrRd(ratingN.to_numpy())

plt.scatter(
    movie.to_numpy(),
    user.to_numpy(),
    s=50,
    marker="s",
    color=colors,
    edgecolor=colors)

plt.legend(
    title='Rating',
    loc="upper left",
    bbox_to_anchor=(1, 1),
    handles=[
        mpatches.Patch(color=plt.cm.YlOrRd(0),    label='1'),
        mpatches.Patch(color=plt.cm.YlOrRd(0.25), label='2'),
        mpatches.Patch(color=plt.cm.YlOrRd(0.5),  label='3'),
        mpatches.Patch(color=plt.cm.YlOrRd(0.75), label='4'),
        mpatches.Patch(color=plt.cm.YlOrRd(0.99), label='5')
    ])

plt.show()
```

In the plot, you can see the ratings color code. For example, **User 4** has rated **Movie 2** with the highest **rating of 5**.
Let's inspect the subset to confirm the values ...

```{code-cell} ipython3

ratings_subset
```

The plot is as expected, so we can repeat this with the full data set.

### Visualise the ratings matrix using the full data set

This time we don't need to filter the data.

```{code-cell} ipython3

ratings_full = ratings.copy()
```

Same functions as before ...

```{code-cell} ipython3

user = ratings_full["user"].astype(int)
movie = ratings_full["product"].astype(int)

min_r = ratings_full["rating"].min()
max_r = ratings_full["rating"].max()

def normalise(x):
    rating = (x - min_r) / (max_r - min_r)
    return float(rating)

ratingN = ratings_full["rating"].apply(normalise)
```

Slightly modified chart, for example to print out smaller markers.

```{code-cell} ipython3

%matplotlib inline
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

max_user = ratings_full["user"].max()
max_movie = ratings_full["product"].max()

width = 10
height = 10
plt.figure(figsize=(width, height))
plt.ylim([0, max_user])
plt.xlim([0, max_movie])
plt.ylabel('User ID')
plt.xlabel('Movie ID')
plt.title('Movie Ratings')

ax = plt.gca()
ax.patch.set_facecolor('#898787') # dark grey background

colors = plt.cm.YlOrRd(ratingN.to_numpy())

plt.scatter(
    movie.to_numpy(),
    user.to_numpy(),
    s=1,
    edgecolor=colors)

plt.legend(
    title='Rating',
    loc="upper left",
    bbox_to_anchor=(1, 1),
    handles=[
        mpatches.Patch(color=plt.cm.YlOrRd(0),    label='1'),
        mpatches.Patch(color=plt.cm.YlOrRd(0.25), label='2'),
        mpatches.Patch(color=plt.cm.YlOrRd(0.5),  label='3'),
        mpatches.Patch(color=plt.cm.YlOrRd(0.75), label='4'),
        mpatches.Patch(color=plt.cm.YlOrRd(0.99), label='5')
    ])

plt.show()
```

We can see some clear patterns. Vertical bands can indicate movies that are rated similarly by many users.
Horizontal bands can indicate users who rate movies consistently—pale lines skew negative while darker red lines skew positive.
There are also grey gaps where users have not rated movies, including a subtle arc-shaped region of missing ratings toward the upper-right of the plot.

**Key Observation:** The grey regions represent missing ratings. The goal of a recommender system is to **predict these missing values** based on the patterns in the observed ratings.

```{code-cell} ipython3

df = ratings.rename(columns={"user": "user_id", "product": "movie_id"})
```

---

## Part 2: The ALS Algorithm

### The Matrix Factorization Idea

ALS solves the recommendation problem by factorizing the user-item rating matrix $R$ (size $m \times n$) into two lower-rank matrices:

$$R \approx U \times M^T$$

Where:
- **$U$** (size $m \times k$): User feature matrix—each user represented by $k$ latent factors
- **$M$** (size $n \times k$): Movie feature matrix—each movie represented by $k$ latent factors
- **$k$**: Number of latent factors (rank), typically $k \ll \min(m, n)$

### What Are Latent Factors?

**Latent factors** are hidden features that capture underlying patterns:
- For movies: genre preferences, actor preferences, release era, cinematography style
- For users: taste in comedy, preference for action, tolerance for violence

**Crucially:** We don't specify what these factors mean—the algorithm **learns** them from data!

### Visualizing the Factorization

```{code-cell} ipython3

fig, ax = plt.subplots(figsize=(14, 6))

# Draw the matrices
user_matrix_pos = [0.05, 0.3, 0.15, 0.4]
movie_matrix_pos = [0.4, 0.3, 0.15, 0.4]
rating_matrix_pos = [0.75, 0.2, 0.2, 0.6]

# User matrix
ax_user = fig.add_axes(user_matrix_pos)
user_data = np.random.rand(6, 3)
ax_user.imshow(user_data, cmap='Blues', aspect='auto')
ax_user.set_title('User Features\n(Users × Factors)', fontsize=12, fontweight='bold')
ax_user.set_ylabel('Users (m)', fontsize=10)
ax_user.set_xlabel('Latent Factors (k)', fontsize=10)
ax_user.set_xticks([])
ax_user.set_yticks([])

# Movie matrix
ax_movie = fig.add_axes(movie_matrix_pos)
movie_data = np.random.rand(3, 5)
ax_movie.imshow(movie_data, cmap='Greens', aspect='auto')
ax_movie.set_title('Movie Features^T\n(Factors × Movies)', fontsize=12, fontweight='bold')
ax_movie.set_ylabel('Latent Factors (k)', fontsize=10)
ax_movie.set_xlabel('Movies (n)', fontsize=10)
ax_movie.set_xticks([])
ax_movie.set_yticks([])

# Rating matrix
ax_rating = fig.add_axes(rating_matrix_pos)
rating_data = np.random.rand(6, 5)
ax_rating.imshow(rating_data, cmap='Reds', aspect='auto')
ax_rating.set_title('Rating Matrix\n(Users × Movies)', fontsize=12, fontweight='bold')
ax_rating.set_ylabel('Users (m)', fontsize=10)
ax_rating.set_xlabel('Movies (n)', fontsize=10)
ax_rating.set_xticks([])
ax_rating.set_yticks([])

# Add multiplication and approximation symbols
ax.text(0.25, 0.5, '×', fontsize=40, ha='center', va='center', transform=fig.transFigure)
ax.text(0.62, 0.5, '≈', fontsize=40, ha='center', va='center', transform=fig.transFigure)

ax.axis('off')
plt.show()
```

### How ALS Works: The Algorithm

ALS **alternates** between optimizing user factors and movie factors:

**Algorithm:**
1. **Initialize** $U$ and $M$ with random values
2. **Fix $M$**, solve for $U$ using least squares:
   $$U_i = (M^T M + \lambda I)^{-1} M^T R_i$$
3. **Fix $U$**, solve for $M$ using least squares:
   $$M_j = (U^T U + \lambda I)^{-1} U^T R_j$$
4. **Repeat** steps 2-3 until convergence

**Key Parameters:**
- **$k$ (rank):** Number of latent factors (typically 5-50)
- **$\lambda$ (regularization):** Prevents overfitting (typically 0.01-0.1)
- **iterations:** Number of alternating optimization steps (typically 10-20)

---

## Part 3: Implementing ALS from Scratch

Let's implement a simple version of ALS in NumPy:

```{code-cell} ipython3
class SimpleALS:
    """Simplified ALS implementation for collaborative filtering."""

    def __init__(self, n_factors=5, n_iterations=10, lambda_reg=0.1):
        self.n_factors = n_factors
        self.n_iterations = n_iterations
        self.lambda_reg = lambda_reg

    def fit(self, ratings_df):
        """
        Train the ALS model.

        Parameters:
        -----------
        ratings_df : DataFrame with columns [user_id, movie_id, rating]
        """
        # Create user and movie ID mappings
        self.user_ids = ratings_df['user_id'].unique()
        self.movie_ids = ratings_df['movie_id'].unique()
        self.n_users = len(self.user_ids)
        self.n_movies = len(self.movie_ids)

        self.user_id_map = {uid: idx for idx, uid in enumerate(self.user_ids)}
        self.movie_id_map = {mid: idx for idx, mid in enumerate(self.movie_ids)}

        # Create rating matrix (sparse)
        R = np.zeros((self.n_users, self.n_movies))
        for _, row in ratings_df.iterrows():
            u_idx = self.user_id_map[row['user_id']]
            m_idx = self.movie_id_map[row['movie_id']]
            R[u_idx, m_idx] = row['rating']

        # Initialize user and movie factors
        self.U = np.random.rand(self.n_users, self.n_factors) * 0.01
        self.M = np.random.rand(self.n_movies, self.n_factors) * 0.01

        # Training loop
        self.losses = []
        for iteration in range(self.n_iterations):
            # Fix M, solve for U
            for u in range(self.n_users):
                # Get movies rated by user u
                rated_movies = np.where(R[u, :] > 0)[0]
                if len(rated_movies) == 0:
                    continue

                M_u = self.M[rated_movies, :]
                R_u = R[u, rated_movies]

                # Solve: U[u] = (M_u^T M_u + λI)^-1 M_u^T R_u
                self.U[u, :] = np.linalg.solve(
                    M_u.T @ M_u + self.lambda_reg * np.eye(self.n_factors),
                    M_u.T @ R_u
                )

            # Fix U, solve for M
            for m in range(self.n_movies):
                # Get users who rated movie m
                rating_users = np.where(R[:, m] > 0)[0]
                if len(rating_users) == 0:
                    continue

                U_m = self.U[rating_users, :]
                R_m = R[rating_users, m]

                # Solve: M[m] = (U_m^T U_m + λI)^-1 U_m^T R_m
                self.M[m, :] = np.linalg.solve(
                    U_m.T @ U_m + self.lambda_reg * np.eye(self.n_factors),
                    U_m.T @ R_m
                )

            # Calculate loss (RMSE on observed ratings)
            predictions = self.U @ self.M.T
            mask = R > 0
            loss = np.sqrt(np.mean((R[mask] - predictions[mask]) ** 2))
            self.losses.append(loss)

            if (iteration + 1) % 5 == 0:
                print(f"Iteration {iteration + 1}/{self.n_iterations}, RMSE: {loss:.4f}")

    def predict(self, user_id, movie_id):
        """Predict rating for a user-movie pair."""
        if user_id not in self.user_id_map or movie_id not in self.movie_id_map:
            return np.nan

        u_idx = self.user_id_map[user_id]
        m_idx = self.movie_id_map[movie_id]

        return self.U[u_idx, :] @ self.M[m_idx, :]

    def recommend_top_n(self, user_id, n=5, exclude_rated=True):
        """Get top N movie recommendations for a user."""
        if user_id not in self.user_id_map:
            return []

        u_idx = self.user_id_map[user_id]
        scores = self.U[u_idx, :] @ self.M.T

        if exclude_rated:
            # Set already rated movies to -inf
            for m_idx, movie_id in enumerate(self.movie_ids):
                if movie_id in self.user_id_map:  # Check if rated
                    scores[m_idx] = -np.inf

        top_indices = np.argsort(scores)[::-1][:n]
        recommendations = [(self.movie_ids[idx], scores[idx]) for idx in top_indices]

        return recommendations
```

### Training the Model

```{code-cell} ipython3
# Train the model
model = SimpleALS(n_factors=10, n_iterations=15, lambda_reg=0.1)
model.fit(df)

# Plot training curve
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(model.losses) + 1), model.losses, marker='o', linewidth=2)
plt.xlabel('Iteration', fontsize=12)
plt.ylabel('RMSE', fontsize=12)
plt.title('ALS Training: Loss Over Iterations', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print(f"\nFinal RMSE: {model.losses[-1]:.4f}")
```

---

## Part 4: Making Predictions and Recommendations

### Single Rating Prediction

```{code-cell} ipython3
# Example: Predict a specific user-movie rating
user_id = df['user_id'].iloc[0]
movie_id = df['movie_id'].iloc[0]
actual_rating = df[(df['user_id'] == user_id) & (df['movie_id'] == movie_id)]['rating'].values[0]
predicted_rating = model.predict(user_id, movie_id)

print(f"User {user_id}, Movie {movie_id}")
print(f"  Actual rating: {actual_rating}")
print(f"  Predicted rating: {predicted_rating:.2f}")
print(f"  Error: {abs(actual_rating - predicted_rating):.2f}")
```

### Top-N Recommendations

```{code-cell} ipython3
# Get top 5 recommendations for a user
user_id = df['user_id'].iloc[0]
recommendations = model.recommend_top_n(user_id, n=5)

print(f"\nTop 5 recommendations for User {user_id}:")
for rank, (movie_id, score) in enumerate(recommendations, 1):
    print(f"  {rank}. Movie {movie_id} (predicted rating: {score:.2f})")
```

---

## Part 5: Understanding the Latent Factors

Let's visualize what the model learned:

```{code-cell} ipython3

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# User factors
im1 = ax1.imshow(model.U[:20, :], cmap='coolwarm', aspect='auto')
ax1.set_title('User Latent Factors (First 20 Users)', fontsize=12, fontweight='bold')
ax1.set_xlabel('Latent Factor', fontsize=10)
ax1.set_ylabel('User ID', fontsize=10)
plt.colorbar(im1, ax=ax1, label='Factor Value')

# Movie factors
im2 = ax2.imshow(model.M[:20, :], cmap='coolwarm', aspect='auto')
ax2.set_title('Movie Latent Factors (First 20 Movies)', fontsize=12, fontweight='bold')
ax2.set_xlabel('Latent Factor', fontsize=10)
ax2.set_ylabel('Movie ID', fontsize=10)
plt.colorbar(im2, ax=ax2, label='Factor Value')

plt.tight_layout()
plt.show()
```

**Interpretation:**
- **Positive values** (red): User/movie has high affinity for this latent factor
- **Negative values** (blue): User/movie has low affinity for this latent factor
- **Near zero** (white): Neutral

For example, Factor 1 might represent "action movies" while Factor 2 represents "comedy."

---

## Part 6: Evaluation Metrics

### Root Mean Squared Error (RMSE)

RMSE measures the average prediction error:

$$\text{RMSE} = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (r_i - \hat{r}_i)^2}$$

Where $r_i$ is actual rating and $\hat{r}_i$ is predicted rating.

**Interpretation:**
- RMSE = 0: Perfect predictions
- RMSE = 0.5: Predictions off by ~0.5 stars on average
- RMSE = 1.0: Predictions off by ~1 star on average

```{code-cell} ipython3
# Calculate RMSE on all ratings
predictions = []
actuals = []

for _, row in df.iterrows():
    pred = model.predict(row['user_id'], row['movie_id'])
    if not np.isnan(pred):
        predictions.append(pred)
        actuals.append(row['rating'])

rmse = np.sqrt(mean_squared_error(actuals, predictions))
mae = np.mean(np.abs(np.array(actuals) - np.array(predictions)))

print(f"Evaluation Metrics:")
print(f"  RMSE: {rmse:.4f} (average error in rating units)")
print(f"  MAE:  {mae:.4f} (mean absolute error)")
print(f"\nInterpretation: Predictions are off by ~{rmse:.2f} stars on average")
```

---

## Part 7: Key Takeaways

### Advantages of ALS

✅ **Scalable**: Can handle millions of users and items
✅ **Parallelizable**: User and movie updates are independent
✅ **Interpretable**: Latent factors have semantic meaning
✅ **Effective**: Works well with sparse data

### Limitations

❌ **Cold start problem**: Can't recommend for new users/movies with no ratings
❌ **Implicit feedback**: Designed for explicit ratings (1-5 stars), not clicks/views
❌ **Context-agnostic**: Doesn't consider time, location, or other context

### When to Use ALS

- **E-commerce**: Product recommendations based on purchase history
- **Media platforms**: Movie/music recommendations (Netflix, Spotify)
- **Content sites**: Article recommendations based on reading patterns
- **Social networks**: Friend or content suggestions

### Extensions and Alternatives

- **Implicit ALS**: For binary data (clicks, purchases)
- **Deep Learning**: Neural collaborative filtering (NCF)
- **Hybrid methods**: Combine with content-based features
- **Context-aware**: Incorporate temporal or contextual information

---

## Summary

We've covered:

1. **The Problem**: Predicting missing ratings in sparse user-item matrices
2. **The Solution**: Matrix factorization using Alternating Least Squares
3. **The Algorithm**: Alternating between optimizing user and item factors
4. **Implementation**: Building ALS from scratch in NumPy
5. **Evaluation**: Using RMSE to measure prediction accuracy
6. **Interpretation**: Understanding learned latent factors

**Next Steps:**
- Try on real MovieLens dataset: [grouplens.org/datasets/movielens](https://grouplens.org/datasets/movielens/)
- Experiment with hyperparameters ($k$, $\lambda$, iterations)
- Implement implicit feedback version for binary data
- Compare with deep learning approaches

---

## References

- **Original Paper**: [Netflix Prize](https://www.netflixprize.com/) (2006-2009)
- **Spark MLlib ALS**: [Apache Spark Documentation](https://spark.apache.org/docs/latest/ml-collaborative-filtering.html)
- **MovieLens Dataset**: [GroupLens Research](https://grouplens.org/datasets/movielens/)
- **Source Code**: [Movie Recommender Demo](https://github.com/snowch/movie-recommender-demo/)

*This tutorial demonstrates collaborative filtering concepts. For production systems with millions of users, consider using distributed implementations like Spark MLlib, TensorFlow Recommenders, or PyTorch.*
