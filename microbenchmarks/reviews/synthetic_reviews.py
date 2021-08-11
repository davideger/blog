import pandas as pd
import polars as pl
import numpy as np
import platform
import pyarrow
import os

from numpy.random import default_rng

rng = default_rng()


def multinomial_selection(choices, ps, num_samples):
  """Choose num_samples from among the given choices according to the given probabilities."""
  return np.array(choices)[np.argmax(
      rng.multinomial(1, ps, size=num_samples), axis=1)]


def get_stars(restaurant_quality=0., num_reviews=20):
  """Return a distribution of star ratings (1-5) for a given restaurant."""
  restaurant_distributions = [
      [.2, .34, .35, .1, .01],  # poor restaurant
      [.1, .15, .1, .6, .05],  # typical restaurant
      [.01, .0, .01, .08, .90],  # great restaurant
  ]
  dist_idx = np.clip(restaurant_quality * 3., 0, 2).astype(int)
  return np.argmax(
      rng.multinomial(1, restaurant_distributions[dist_idx], size=num_reviews),
      axis=1) + 1.


def get_review_quality(word_counts):
  """Return a distribution of textual quality (0.

  - 1.) for reviews with
  the given word counts.

  """
  quality_distributions = [
      [1., 0., 0., 0., 0.],  # short reviews
      [.1, .15, .1, .6, .05],  # medium reviews
      [.01, .0, .01, .08, .90],  # long reviews
  ]
  dist_indices = np.searchsorted([2, 120], word_counts)
  # TODO: make this more performant
  return np.clip(
      np.array([
          np.argmax(rng.multinomial(1, quality_distributions[dist_idx])) / 4.
          for dist_idx in dist_indices
      ]) + (rng.random(len(word_counts)) * .25), 0., 1.)


def get_age_weeks(num_reviews=20):
  max_age_weeks = 1017
  jitter_weeks = 26
  return np.clip(
      np.random.zipf(1.6, num_reviews), 1, max_age_weeks -
      jitter_weeks) + np.random.randint(0, jitter_weeks, num_reviews)


def random_choice(choices, num_reviews):
  return np.choose(
      np.random.randint(0, len(choices), num_reviews), np.array(choices))


review_aspects = ["atmosphere", "food", "speed", "location", "friendliness"]


def get_topic(num_reviews):
  return random_choice(review_aspects, num_reviews)


def generate_synthetic_review(restaurant_quality, num_reviews):
  # stars               - 1-5
  # reviewer_id         - int
  # review_age_weeks    - int >= 0
  # review_month        - int (1-12)
  # primary_topic       - string (categorical: atmosphere, dish, speed, location, friendliness)
  # kids_involved       - bool
  # for_business        - bool
  # text_length         - int (word count)
  # text_quality        - float (0.0 - 1.0)
  # language            - string (2-letter language code)
  stars = get_stars(restaurant_quality, num_reviews)
  reviewer_id = np.random.randint(1000, 9999, num_reviews)
  review_age_weeks = get_age_weeks(num_reviews)
  review_month = np.ones(num_reviews) * 12 - np.mod(review_age_weeks, 12)
  primary_topic = get_topic(num_reviews)
  kids_involved = np.random.binomial(1, .2, num_reviews).astype(bool)
  for_business = np.random.binomial(1, .4,
                                    num_reviews).astype(bool) & ~kids_involved
  kids_involved = [bool(x) for x in kids_involved]
  for_business = [bool(x) for x in for_business]
  text_length = np.random.randint(0, 1200, num_reviews)
  text_quality = get_review_quality(text_length)
  language = multinomial_selection(["en", "es", "zh", "de", "fr"],
                                   [.4, .1, .3, .15, .05], num_reviews)
  rv = {}
  for v in [
      "stars", "reviewer_id", "review_age_weeks", "review_month",
      "primary_topic", "kids_involved", "for_business", "text_length",
      "text_quality", "language"
  ]:
    rv[v] = locals()[v]
  return rv


def generate_synthetic_reviews_dataset(num_listings=5000):
  num_reviews = rng.zipf(2, num_listings)
  restaurant_goodness = rng.random(num_listings)
  reviews_dfs = []
  reviews_pls = []
  for i in range(num_listings):
    data = generate_synthetic_review(restaurant_goodness[i], num_reviews[i])
    reviews_dfs.append(pd.DataFrame(data))
    npl = pl.DataFrame(data)
    npl["index"] = pl.Series("index", list(range(num_reviews[i])))
    reviews_pls.append(npl)
  return reviews_dfs, reviews_pls


def pd_calculate_per_aspect_star_rating(reviews):
  """Given a dataframe of reviews, return a star rating per aspect.


  The returned rating data frame will have one column per review aspect,
  and a single row with the average star rating for review focusing on
  that aspect.
  """
  star_prior = 3.0
  default_reviews = pd.DataFrame({
      "primary_topic": review_aspects,
      "stars": [star_prior] * len(review_aspects)
  })
  return pd.concat([reviews[["primary_topic", "stars"]], default_reviews
                   ]).groupby("primary_topic").mean().transpose()


def pd_get_aspect_ratings_for_listings(pd_reviews_per_listing):
  rating_summaries = []
  for i in range(len(pd_reviews_per_listing)):
    ratings = pd_calculate_per_aspect_star_rating(pd_reviews_per_listing[i])
    rating_summaries.append(ratings)
  return pd.concat(rating_summaries, axis=0)


def pd_get_aspect_ranks_slower(pd_listing_aspect_ratings, listing_id):
  """Return the rank of this listing globally for a given aspect."""
  # Note - this code is purposefully a bit inefficient to show the outsized
  # slowdown below.
  listing_ratings = pd_listing_aspect_ratings.iloc[listing_id]
  rv = {}
  for aspect in review_aspects:
    rv[aspect] = [(pd_listing_aspect_ratings.loc[:, [aspect]] >
                   listing_ratings[aspect]).sum().iloc[0]]
  return pd.DataFrame(rv)


def pd_get_aspect_ranks_speedy(pd_listing_aspect_ratings):
  """Return the rank of this listing globally for a given aspect."""
  # Note - this code is purposefully a bit inefficient to show the outsized
  # slowdown below.
  num_listings = pd_listing_aspect_ratings.shape[0]
  ordered = {}
  for col in pd_listing_aspect_ratings.columns:
    ordered[col] = pd_listing_aspect_ratings[col].sort_values()

  def get_ranks(listing_id):
    listing_ratings = pd_listing_aspect_ratings.iloc[listing_id]
    rv = {}
    for aspect in review_aspects:
      rv[aspect] = [
          num_listings - np.searchsorted(
              ordered[aspect], listing_ratings[aspect], side="right")
      ]
    return pd.DataFrame(rv)

  return get_ranks


def get_aspect_rank_descriptions(listing_id, num_listings, get_aspect_ranks_fn):
  dl = get_aspect_ranks_fn(listing_id)
  rank_descriptions = [
      f"#{dl.loc[0, aspect]} out of {num_listings} for {aspect}"
      for aspect in dl.columns
  ]
  return "; ".join(rank_descriptions)
