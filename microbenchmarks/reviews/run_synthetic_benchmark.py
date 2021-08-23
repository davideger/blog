import copy
import cProfile
import datetime
import numpy as np
import os
import psutil
import synthetic_reviews
import time
import time_util
import who
from functools import partial


# Here we present two versions of a groupby+head+sort for benchmarking,
# one in Polars (pl) and one in Pandas (pd)
def pl_get_representative_reviews(df, depth):
  return df[df.language == 'en'].sort([
      'stars', 'review_age_weeks'
  ]).groupby('stars').head(depth).sort(['stars', 'review_age_weeks']).index


def pd_get_representative_reviews(df, depth):
  return df[df.language == 'en'].sort_values(
      by=['stars', 'review_age_weeks']).groupby('stars').head(depth).index


def process_listing(df, listing_id, get_reviews_fn, depth,
                    get_ranks_description_fn):
  rv = {'df': df, 'listing_id': listing_id}
  best_reviews = get_reviews_fn(df, depth)
  ranks_description = get_ranks_description_fn(listing_id)
  rv['cpu_id'] = psutil.Process().cpu_num()
  rv['html'] = (f'Listing {listing_id}:: {ranks_description}\n' +
                f'Be sure to check out reviews {best_reviews}')
  return rv


def pd_process_listing(df, listing_id, depth, get_ranks_description_fn):
  return process_listing(df, listing_id, pd_get_representative_reviews, depth,
                         get_ranks_description_fn)


def pl_process_listing(df, listing_id, depth, get_ranks_description_fn):
  return process_listing(df, listing_id, pl_get_representative_reviews, depth,
                         get_ranks_description_fn)


bmsize = 25000
pandas_reviews, polars_reviews = synthetic_reviews.generate_synthetic_reviews_dataset(
    bmsize)

# Memory intensive, comparing this rating to all other average aspect ratings
# for each other restaurant each time.  O(num_aspects * num_restaurants)
pd_listing_aspect_ratings = synthetic_reviews.pd_get_aspect_ratings_for_listings(
    pandas_reviews)
slow_ranker = partial(synthetic_reviews.pd_get_aspect_ranks_slower,
                      pd_listing_aspect_ratings)
fast_ranker = synthetic_reviews.pd_get_aspect_ranks_speedy(
    pd_listing_aspect_ratings)


def noop_description_maker(listing_id):
  return 'This place is fast!'


def slow_rank_description_maker(listing_id):
  return synthetic_reviews.get_aspect_rank_descriptions(listing_id, bmsize,
                                                        slow_ranker)


def fast_rank_description_maker(listing_id):
  return synthetic_reviews.get_aspect_rank_descriptions(listing_id, bmsize,
                                                        fast_ranker)


def cpu_history(processed_listings):
  # a is an array of cpu ids in (presumably) program order.
  if not processed_listings:
      return "(no history - did you not run any code?)"
  a = [r['cpu_id'] for r in processed_listings]
  num_core_changes = ((a - np.roll(a, 1)) != 0.).astype(int).sum()
  return f'Over {len(a)} records, computation jumped core at least {num_core_changes} times.'


import ray

MP_STYLES = [None, 'ray']
MP_STYLE = 'ray'


def pxmap(f, xs, mp_style):
  """Parallel map, implemented with several python parallel execution libraries."""
  if mp_style not in MP_STYLES:
    print(f'Unrecognized mp_style {mp_style}')
  elif mp_style == 'ray':

    @ray.remote
    def g(x):
      return f(x)

    return ray.get([g.remote(x) for x in xs])
  return [f(x) for x in xs]

pandas_results = None
for (label, postprocessing) in [
    ('no post processing', 'noop_description_maker'),
    ('rank finding', 'fast_rank_description_maker'),
]:
  print(
      f'Pandas finding representative reviews for {bmsize} synthetic listings + {label}'
  )
  pandas_results = time_util.timeit(
      f'[process_listing(df, listing_id, pd_get_representative_reviews, d, {postprocessing})'
      'for (listing_id, df) in enumerate(pandas_reviews) for d in [1,3,9]]',
      globals(), locals())

print(f'CPU History: {cpu_history(pandas_results)}')

polars_results = None
for (label, postprocessing) in [
    ('no post processing', 'noop_description_maker'),
    ('rank finding', 'fast_rank_description_maker'),
]:
  print(
      f'Polars finding representative reviews for {bmsize} synthetic listings + {label}'
  )
  polars_results = time_util.timeit(
      f'[process_listing(df, listing_id, pl_get_representative_reviews, d, {postprocessing})'
      'for (listing_id, df) in enumerate(polars_reviews) for d in [1,3,9]]',
      globals(), locals())

print(f'CPU History: {cpu_history(polars_results)}')


pandas_ray_results = None
for (label, postprocessing) in [
    ('no post processing', noop_description_maker),
    ('rank finding', fast_rank_description_maker),
]:
  if not ray.is_initialized():
      ray.init()
  print(
      f'Pandas (ray) finding representative reviews for {bmsize} synthetic listings + {label}'
  )
  xs = [(lid, df, d)
        for (lid, df) in enumerate(pandas_reviews)
        for d in [1, 3, 9]]

  def doit(ldd):
    lid, df, d = ldd
    return process_listing(df, lid, pd_get_representative_reviews, d,
                           postprocessing)

  pandas_ray_results = time_util.timeit(f"pxmap(doit, xs, 'ray')", globals(),
                                        locals())

print(f'CPU History: {cpu_history(pandas_ray_results)}')


