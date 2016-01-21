# Savvy Suds

## Overview
I built a recommendation engine for thebeerexchange.io (BEX), an app enabling users to trade beers you have for beers you want. Looking for Pliny the Younger or Heady Topper? Sign up for BEX. If you just want recommendations, check out my web app at savvysuds.io.

## Dataset
The dataset is comprised by the wishlists and beer trading information of BEX users. Users who had too few beer interactions were filtered out to avoid bias in the relationships between beers for the recommender.

## Model
I used an item-based collaborative filtering model that employs Jaccard similarity to pick beers to recommend. That's a fancy way of saying that my model recommends the beers that users like you enjoyed. The model avoids recommending beers preferred by users with different tastes. which helps to alleviate the popularity problem suffered by many recommenders.
