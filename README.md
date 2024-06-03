# Movie Recommendation System with TensorFlow Recommenders

This repository contains a movie recommendation system built using TensorFlow Recommenders (TFRS). The system uses the MovieLens 100K dataset to build and evaluate retrieval and ranking models.

## Features

-   **Retrieval Model**: A model to retrieve candidate movies for a given user.
-   **Ranking Model**: A model to rank the retrieved candidate movies based on predicted user ratings.
-   **Two-Tower Architecture**: Utilizes a two-tower architecture for computing user and movie embeddings.
-   **Brute Force Search**: A simple retrieval mechanism using brute-force search for demonstration purposes.

## Requirements

-   Python 3.7+
-   TensorFlow 2.6+
-   TensorFlow Recommenders
-   TensorFlow Datasets
-   NumPy

## Usage

Run the main script to train the models and get movie recommendations:

```bash
python main.py
```

## Code Overview

### `retrieve_movies`

Retrieves a list of movie titles for a given user.

```python
def retrieve_movies(index: tfrs.layers, user_id: str):
    _, titles = index(np.array([user_id]))
    titles = titles.numpy().astype(str).flatten()
    return titles
```

### `rank_movies`

Ranks a list of movie titles for a given user based on predicted ratings.

```python
def rank_movies(titles: List[str], user_id: str, model: tfrs.models.Model):
    ratings = {}
    for movie_title in titles:
        ratings[movie_title] = model(
            {"user_id": np.array([user_id]), "movie_title": np.array([movie_title])}
        )

    for title, score in sorted(ratings.items(), key=lambda x: x[1], reverse=True):
        print(f"{title}: {score}")
    return ratings
```

### `print_ratings`

Prints the ratings of movies in a sorted order.

```python
def print_ratings(ratings: Dict[str, tf.Tensor]):
    for title, score in sorted(ratings.items(), key=lambda x: x[1], reverse=True):
        print(f"{title}: {score}")
```

### `TwoTowerModel`

A two-tower model for computing user and movie embeddings.

```python
class TwoTowerModel(tf.keras.Model):
    def __init__(self, unique_user_ids, unique_movie_titles):
        super().__init__()
        embedding_dimension = 32

        self.user_embeddings = tf.keras.Sequential([
            tf.keras.layers.StringLookup(vocabulary=unique_user_ids, mask_token=None),
            tf.keras.layers.Embedding(len(unique_user_ids) + 1, embedding_dimension),
        ])

        self.movie_embeddings = tf.keras.Sequential([
            tf.keras.layers.StringLookup(vocabulary=unique_movie_titles, mask_token=None),
            tf.keras.layers.Embedding(len(unique_movie_titles) + 1, embedding_dimension),
        ])

        self.ratings = tf.keras.Sequential([
            tf.keras.layers.Dense(256, activation="relu"),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(1),
        ])

    def call(self, inputs):
        user_id, movie_title = inputs
        user_embedding = self.user_embeddings(user_id)
        movie_embedding = self.movie_embeddings(movie_title)
        return self.ratings(tf.concat([user_embedding, movie_embedding], axis=1))
```

### `MovieLensRanking`

A model class for movie ranking.

```python
class MovieLensRanking(tfrs.models.Model):
    def __init__(self, unique_user_ids, unique_movie_titles):
        super().__init__()
        self.ranking_model: tf.keras.Model = TwoTowerModel(unique_user_ids, unique_movie_titles)
        self.task: tf.keras.layers.Layer = tfrs.tasks.Ranking(
            loss=tf.keras.losses.MeanSquaredError(),
            metrics=[tf.keras.metrics.RootMeanSquaredError()],
        )

    def call(self, features: Dict[str, tf.Tensor]) -> tf.Tensor:
        return self.ranking_model((features["user_id"], features["movie_title"]))

    def compute_loss(self, features: Dict[Text, tf.Tensor], training=False) -> tf.Tensor:
        labels = features.pop("user_rating")
        rating_predictions = self(features)
        return self.task(labels=labels, predictions=rating_predictions)
```

### `MovieLensRetrieval`

A model class for movie retrieval.

```python
class MovieLensRetrieval(tfrs.models.Model):
    def __init__(self, unique_user_ids, unique_movie_titles, movies):
        super().__init__()
        embedding_dimension = 32
        self.user_model: tf.keras.Model = tf.keras.Sequential([
            tf.keras.layers.StringLookup(vocabulary=unique_user_ids, mask_token=None),
            tf.keras.layers.Embedding(len(unique_user_ids) + 1, embedding_dimension),
        ])

        self.movie_model: tf.keras.Model = tf.keras.Sequential([
            tf.keras.layers.StringLookup(vocabulary=unique_movie_titles, mask_token=None),
            tf.keras.layers.Embedding(len(unique_movie_titles) + 1, embedding_dimension),
        ])

        metrics = tfrs.metrics.FactorizedTopK(candidates=movies.batch(128).map(self.movie_model))
        self.task: tf.keras.layers.Layer = tfrs.tasks.Retrieval(metrics=metrics)

    def compute_loss(self, features: Dict[Text, tf.Tensor], training=False) -> tf.Tensor:
        user_embeddings = self.user_model(features["user_id"])
        positive_movie_embeddings = self.movie_model(features["movie_title"])
        return self.task(user_embeddings, positive_movie_embeddings)
```

### `main`

The main function to load data, train models, and get recommendations.

```python
def main():
    ratings = tfds.load("movielens/100k-ratings", split="train")

    ratings = ratings.map(lambda x: {
        "movie_title": x["movie_title"],
        "user_id": x["user_id"],
        "user_rating": x["user_rating"],
    })

    tf.random.set_seed(42)
    shuffled = ratings.shuffle(100_000, seed=42, reshuffle_each_iteration=False)

    train = shuffled.take(80_000)
    test = shuffled.skip(80_000).take(20_000)

    movie_titles = ratings.batch(100_000).map(lambda x: x["movie_title"])
    user_ids = ratings.batch(100_000).map(lambda x: x["user_id"])

    unique_movie_titles = np.unique(np.concatenate(list(movie_titles)))
    unique_user_ids = np.unique(np.concatenate(list(user_ids)))

    cached_train = train.shuffle(100_000).batch(4096).cache()
    cached_test = test.batch(4096).cache()

    movies = tfds.load("movielens/100k-movies", split="train")
    movies = movies.map(lambda x: x["movie_title"])
    ret = MovieLensRetrieval(unique_user_ids, unique_movie_titles, movies)
    ret.compile(optimizer=tf.keras.optimizers.Adagrad(learning_rate=0.1))
    ret.fit(cached_train, epochs=3)
    ret.evaluate(cached_test, return_dict=True)

    model = MovieLensRanking(unique_user_ids, unique_movie_titles)
    model.compile(optimizer=tf.keras.optimizers.Adagrad(learning_rate=0.1))

    model.fit(cached_train, epochs=3)
    model.evaluate(cached_test, return_dict=True)

    index = tfrs.layers.factorized_top_k.BruteForce(ret.user_model)
    index.index_from_dataset(movies.batch(100).map(lambda title: (title, ret.movie_model(title))))

    user_id = "42"
    titles = retrieve_movies(index, user_id)
    ratings = rank_movies(titles, user_id, model)
    print_ratings(ratings)
```
