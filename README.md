# Savvy Suds
Check out the web app at www.savvy-suds.com!

### Overview
I built a recommendation engine for thebeerexchange.io (BEX), where users can trade beers they have for beers they want. This model lives on two different platforms. First, I created a web app where you can enter beers you like to see your recommendations. Second, we're in the process of integrating the model into the BEX homepage (in the process of wrapping up A/B tests now). Looking to get a bottle of Pliny the Younger or Heady Topper? Sign up for BEX. If you just want to know what beer to try next, check out my web app at savvy-suds.com.

### Model
I used an item-based collaborative filtering model that employs Jaccard similarity to pick beers to recommend. That's a fancy way of saying that my model recommends the beers enjoyed by users with tastes similar to yours. The model avoids recommending beers preferred by users with different preferences, which helps to alleviate the popularity bias that generally plagues recommenders.

### Dataset
The dataset is comprised by the wishlists and beer trading information of BEX users. Users who had too few beer interactions were filtered out to avoid bias in the relationships between beers.

### Presentation
I gave a talk on this model in late January. Check out the slides here: ow.ly/Ysc05
