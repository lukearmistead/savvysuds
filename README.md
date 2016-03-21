# Savvy Suds (savvysuds.io)
I built an item similarity recommender using craft beer trading data from the 17,000 users of TheBeerExchange.io (BEX). This algorithm lives on two different platforms. 
1. The simple web app I created at savvysuds.io
2. The BEX platform
Looking for some ideas about what beer to drink next? Check out my web app at savvysuds.io. Want to actually get that hard-to-find beer and get richer recommendations? Sign up for BEX.

### Model
I used an item similarity collaborative filtering model that employs Jaccard similarity to pick beers to recommend. That's a fancy way of saying that my model recommends the beers from the wishlists of users like you. The algorithm avoids recommending beers preferred by users with different preferences, which helps to alleviate the popularity bias that generally plagues recommenders.

### Dataset
The dataset is comprised by the wishlists and beer trading information of BEX users. Users who had too few beer interactions were filtered out to avoid bias in the relationships between beers.

### Presentation
I gave a talk on this model in late January. Check out the slides here: www.ow.ly/Ysc05

### Pretty Picture of Beer Preference Clusters
![graph](/other/clusters.png)
