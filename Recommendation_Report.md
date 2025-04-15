# Movie Recommendation System Report

## 1. Introduction

Movie recommendation systems help users discover films they are likely to enjoy by analysing historical rating patterns.  In this project we implemented and compared three classic approaches that span the collaborative‑filtering family:

- **User‑based collaborative filtering** – recommends movies liked by users who rate films similarly to the target user.
- **Item‑based collaborative filtering** – recommends movies that receive similar rating patterns to a film the user likes.
- **Random‑walk (Pixie‑inspired) graph recommendations** – treats the user‑movie interactions as a bipartite graph and uses weighted random walks to surface movies that lie close to the user or a seed movie in this graph.

These complementary perspectives illustrate how neighbourhood methods, item similarity, and graph traversal can all be leveraged to personalise content.

---

## 2. Dataset Description

**MovieLens 100K** is a benchmark data set released by GroupLens that contains explicit 5‑star ratings collected from the MovieLens web site (1997‑1998).

| Statistic | Value       |
| --------- | ----------- |
| Users     | **943**     |
| Movies    | **1682**   |
| Ratings   | **100 000** |

### Raw files

| File     |  | Selected columns                             |                                          |
| -------- | --------- | -------------------------------------------- | ---------------------------------------- |
| `u.data` || `user_id`, `movie_id`, `rating`, `timestamp` |                                          |
| `u.item` || `movie_id`, `title`, `release_date`      |
| `u.user` |     | `user_id`, `age`, `gender`, `occupation` |

### Pre‑processing steps

1. **Load** each file with the appropriate delimiter (tab delimiter for u.data, | for u.item and u.user) and column names using **pandas**.
2. **Convert** Unix timestamps to human‑readable dates (`pd.to_datetime(..., unit='s')`).
3. **Export** the cleaned `ratings`, `movies`, and `users` data frames to `ratings.csv`, `movies.csv`, and `users.csv` for reproducibility.
4. **Pivot** the ratings to build a sparse **user‑movie matrix** (rows = users, columns = movies) that underpins the CF models.

---

## 3. Methodology

### 3.1 User‑Based Collaborative Filtering

- Construct the user‑movie matrix `R`.
- Fill `NaN` with 0 and compute the cosine similarity between every pair of users:

```python
from sklearn.metrics.pairwise import cosine_similarity
user_sim = cosine_similarity(R.fillna(0))
user_sim_df = pd.DataFrame(user_sim, index=R.index, columns=R.index)
```

- For a target user **u**:
  - Select the top‑k most similar users (excluding **u**).
  - Aggregate their ratings with a similarity‑weighted average.
  - Recommend the highest‑scoring unseen movies.

### 3.2 Item‑Based Collaborative Filtering

- Transpose the user‑movie matrix so that rows represent movies.
- Compute the item‑item cosine similarity matrix.
- For a seed movie **m** return the top‑k most similar movies, excluding **m** itself.

### 3.3 Random‑Walk (Pixie) Recommendation

- **Graph construction** – build an undirected bipartite graph `G=(U∪M, E)` where an edge connects user *u* to movie *m* if *u* rated *m*.
- **Weighted walk** – starting from a user (or movie) node, repeatedly:
  1. Choose a neighbouring edge with probability proportional to the normalised rating weight.
  2. Alternate between user and movie layers for `walk_length` steps.
- **Ranking** – count the visits to movie nodes and return the top‑N most visited.

Random walks naturally exploit high‑degree hubs and multi‑hop paths, often surfacing niche but relevant titles that similarity metrics miss.

---

## 4. Implementation Details

### 4.1 Core Functions

| Function                                                       | Purpose                                      |
| -------------------------------------------------------------- | -------------------------------------------- |
| `recommend_movies_for_user(user_id, num=5)`                    | User‑based CF recommender                    |
| `recommend_movies(movie_name, num=5)`                          | Item‑based CF recommender                    |
| `build_graph(ratings, movies)`                                 | Creates adjacency‑list representation of `G` |
| `weighted_pixie_recommend(start_point, walk_length=15, num=5)` | Random‑walk recommender                      |

---

### 4.2  Collaborative‑Filtering Functions
1. **User‑based recommender (`recommend_movies_for_user`)**
   1. Create a sparse *user × movie* matrix with `pivot`.
   2. Replace `NaN` with 0 so cosine similarity is well‑defined.
   3. Compute a *user × user* similarity matrix via `sklearn.metrics.pairwise.cosine_similarity`.
   4. For the target user, pick the top‑`k` most similar neighbours (default `k = 5`).
   5. Aggregate their ratings with a similarity‑weighted average and return the `N` highest‑scoring unseen movies.

2. **Item‑based recommender (`recommend_movies`)**
   1. Transpose the user–movie matrix so movies are rows.
   2. Compute a *movie × movie* cosine similarity matrix.
   3. Given a title, look up its `movie_id`, pull its similarity vector, drop itself, and surface the `N` closest titles.

Both helpers map final `movie_id`s back to human‑readable titles using the pre‑loaded `movies` DataFrame.

---

### 4.3  Building the Adjacency‑List Graph
To support random‑walk recommendations we need an efficient in‑memory representation of the user–movie interaction graph.  We:

1. **Enrich ratings with titles**
   ```python
   ratings_full = ratings.merge(movies[['movie_id', 'title']], on='movie_id')
   ```
2. **Mean‑aggregate duplicates** so each `(user, movie)` pair has a single score.
3. **Normalise for user bias** by subtracting each user’s average rating:
   ```python
   ratings_full['rating'] = ratings_full.groupby('user_id')['rating'] \
                                   .transform(lambda r: r - r.mean())
   ```
4. **Populate a dictionary of sets** (undirected bipartite graph):
   ```python
   graph = defaultdict(set)
   for u, m in ratings_full[['user_id', 'movie_id']].itertuples(index=False):
       graph[u].add(m)   # edge user → movie
       graph[m].add(u)   # mirror edge movie → user
   ```
   This yields O(E) memory where *E* is the number of rating events and supports O(1) neighbour look‑ups.

---

### 4.4  Weighted Random Walks & Ranking
The **`weighted_pixie_recommend`** function implements a simplified Pixie algorithm:

1. **Starting node** – either a `user_id` or a `movie_id`.
2. **Walk parameters** – `walk_length` (default 15) and number of results `num` (default 5).
3. **Transition rule**
   * If we are at a *user* node, choose the next movie with probability proportional to the *absolute* (bias‑corrected) rating that user gave that movie.
   * If we are at a *movie* node, choose the next user with probability proportional to that user’s (absolute) rating of the movie.
4. **Visit tracking** – every time we land on a movie node we increment a counter in `visit_counts[movie_id]`.
5. **Termination** – after `walk_length` steps, sort `visit_counts` in descending order and return the top‑`num` titles.

Because transitions favour higher ratings, the walk naturally drifts toward highly‑rated movies in the local neighbourhood of the starting node, closely mirroring Pixie’s intuition that *“people who like this also strongly like …”*.

---

## 5. Results and Evaluation

### Example Outputs

**User‑based CF (user 10)**

| Rank | Movie                        |
| ---- | ---------------------------- |
| 1    | In the Company of Men (1997) |
| 2    | Misérables, Les (1995)       |
| 3    | Thin Blue Line, The (1988)   |
| 4    | Braindead (1992)             |
| 5    | Boys, Les (1997)             |

**Item‑based CF ("Jurassic Park (1993)")**

| Rank | Movie                                     |
| ---- | ----------------------------------------- |
| 1    | Top Gun (1986)                            |
| 2    | Empire Strikes Back, The (1980)           |
| 3    | Raiders of the Lost Ark (1981)            |
| 4    | Indiana Jones and the Last Crusade (1989) |
| 5    | Speed (1994)                              |

**Pixie (user 1, walk\_length = 15)**

| Rank | Movie                             |
| ---- | --------------------------------- |
| 1    | Toy Story (1995)                  |
| 2    | Welcome to the Dollhouse (1995)   |
| 3    | Emma (1996)                       |
| 4    | Jackie Chan's First Strike (1996) |
| 5    | Silence of the Lambs, The (1991)  |

### Comparative Discussion

| Aspect                    | User‑CF                 | Item‑CF                       | Pixie                               |
| ------------------------- | ----------------------- | ----------------------------- | ----------------------------------- |
| **Personalisation**       | High (user similarity)  | Medium (item similarity)      | High (graph proximity)              |
| **Scalability**           | Expensive on many users | Scales better (items < users) | Tunable via walk length             |
| **Cold‑start**            | Suffers for new users   | Suffers for new items         | Mitigated if connected via users    |
| **Novelty / Serendipity** | Moderate                | Low‑moderate                  | Often higher due to multi‑hop paths |

**Limitations**

- No explicit train/test split – accuracy metrics (MAE, RMSE, precision\@k) were not computed.
- Ratings were binarised to weights in Pixie; further tuning could exploit full rating scale.
- CF models fill missing ratings with 0, which can bias similarity; mean‑centering or implicit feedback weighting would improve robustness.

---

## 6. Conclusion

This project demonstrates how three fundamental recommender paradigms can be implemented with concise Python/pandas code:

- User‑based CF captures community taste alignment but scales poorly.
- Item‑based CF is efficient and interpretable but can lack diversity.
- Pixie‑style random walks exploit the full user‑item graph to surface both popular and niche titles.

### Potential Improvements

- **Hybrid ensembles** that blend CF scores with Pixie visit counts.
- Incorporating **implicit feedback** (e.g., watch counts) and **content features** (genres, tags).
- **Model‑based CF** (matrix factorisation, neural CF) for better generalisation.
- Applying **evaluation metrics** on a held‑out test set to quantify performance.

### Real‑World Applications

- Streaming platforms (Netflix, Disney+) to suggest "Because you watched …" rows.
- E‑commerce product recommendation where items and customers form a bipartite graph.
- News or music apps to diversify feeds via random‑walk exploration.

---



