import csv
import json
import os

import pandas as pd

import process_data as get

if __name__ == '__main__':
    # initialize file paths
    input_path = '../data/input/beers.csv'
    wishlist_input_path = '../data/input/ftiso.csv'
    output_path = 'static/assets/beers.json'

    wishlist = pd.read_csv(wishlist_input_path)
    wishlist = wishlist[['beer_id', 'quantity']].groupby(['beer_id']).sum()

    beers = pd.read_csv(input_path)
    beers = beers.merge(wishlist, left_on='id', right_index=True)
    beers = beers[beers['quantity'] > 1]

    beers['beer_info'] = beers['name'] + ': ' + beers['brewery_name']
    beer_list = beers[['id', 'beer_info']]
    beer_list.index = beer_list.id
    beer_list.drop('id', axis=1)
    beer_list.to_json(output_path, orient='records')
