import pandas as pd
import graphlab as gl


if __name__ == '__main__':
    # get user input
    user_input = [6988, 4485, 21927]
    user_input = pd.DataFrame({'user_id': [20000 for _ in user_input], 'item_id': user_input})
    beers = pd.read_csv('../data/input/beers.csv')

    # identify user-favored styles for additional filtering
    user_styles = user_input.merge(beers, left_on='item_id', right_on='id')['style']
    print list(user_styles)
    items = beers[beers['style'].isin(list(user_styles))]
    items = gl.SFrame(items)['id']

    # load model and generate recommendations
    model = gl.load_model('../models/item_similarity_model')
    user_input = gl.SFrame(user_input)
    pred = list(model.recommend(users=[20000], items=items, k=5, new_observation_data=user_input, diversity=3)['item_id'])

    # format recommendations for output
    beer_recs = beers[beers['id'].isin(pred)]
    beer_recs = beer_recs[['name', 'brewery_name', 'style', 'score']]
    beer_recs.columns = ['brew', 'brewery', 'style', 'untappd score']
    # beer_recs = beer_recs.to_html(columns=['brew', 'brewery', 'style', 'untappd score'], index=False)
