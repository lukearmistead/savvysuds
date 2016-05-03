# Savvy Suds
##### Beer recommendations, courtesy of 20,000 craft beer nerds
I built an item similarity recommender using craft beer trading data from the ~20,000 users of The Beer Exchange (BEX), a craft beer trading marketplace. Currently, this algorithm provides recommendations for users of [BEX](http://www.thebeerexchange.io/) and [Savvy Suds](http://www.savvysuds.io), the simple web app I put together.


*The data science team at Galvanize wrote some nice things about my recommender! Read more here: [galvanize.com/blog/beer-recommender](http://www.galvanize.com/blog/beer-recommender/)*


### Model
I used an item similarity collaborative filtering model that employs Jaccard similarity to choose which beers to recommend. That's a fancy way of saying that my model recommends the beers from the wishlists of users like you. The algorithm avoids suggesting beers preferred by users with different preferences, which helps to alleviate the popularity bias that generally plagues recommenders.

Here's a visual explanation of item similarity:

<img src="/readme_assets/item_similarity_infographic.jpg" width=300/>

*Special thanks to the design wizards at [Galvanize](http://www.galvanize.com/blog/beer-recommender/) for the awesome infographic :)*

### Dataset
The dataset is comprised by the beer wishlists BEX users. A beer can be added to a wishlist in three ways:
1. Adding a beer to the "in search of" list (easy for the user)
2. Proposing a trade for a beer (a bit harder)
3. Completing a trade to receive a beer (really, really hard)

After experimenting with models using different combinations of these signals, I found that the item similarity recommender using only completed trades worked the best.

It takes a lot of effort to complete a trade for a beer. You have to find a trading partner, negotiate your terms, get the beers you promised, and mail them across the country. All of this effort amounts to an incredibly strong signal of interest for the beer the user hopes to get out of the trade.

Users who had too few beer interactions don't provide much information on how beers relate to each other and were thus filtered out of the dataset used to train the model.

### Beer Tastes
The interactions between users and beers reveal some very interesting patterns user tastes. Three distinct communities emerged, IPA lovers, stout lovers, and sour lovers. Interestingly, there was a lot of overlap between these communities, revealing strong diversity in the taste of these communities.


![graph](/readme_assets/graph_viz.png)

### Learn More
##### Press: [galvanize.com/blog/beer-recommender](http://www.galvanize.com/blog/beer-recommender/)
Read Galvanize's take on my recommender.
##### Web app: [savvysuds.io](http://www.savvysuds.io)
Get some ideas about what beer to try next.
##### BEX homepage: [thebeerexchange.io](http://www.thebeerexchange.io/)
The beer trading platform where you can get essentially any beer you want.
##### Slides from my presentation: [ow.ly/Ysc05](www.ow.ly/Ysc05)
I gave a talk about this model in late January.
