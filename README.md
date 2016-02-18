# Savvy Suds
www.savvy-suds.com

### Overview
I built a recommendation engine for thebeerexchange.io (BEX), an app enabling users to trade beers they have for beers they want. Looking for Pliny the Younger or Heady Topper? Sign up for BEX. If you just want to know what beer to try next, check out my web app at savvy-suds.com.

### Model
I used an item-based collaborative filtering model that employs Jaccard similarity to pick beers to recommend. That's a fancy way of saying that my model recommends the beers enjoyed by users with tastes similar to yours. The model avoids recommending beers preferred by users with different preferences, which helps to alleviate the popularity bias that generally plagues recommenders.

### Dataset
The dataset is comprised by the wishlists and beer trading information of BEX users. Users who had too few beer interactions were filtered out to avoid bias in the relationships between beers.
