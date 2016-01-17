import logging
import os
import shutil

import numpy as np
import pandas as pd
import graphlab as gl

import process_data as get

def recs(users, model, ft, user_beers, beers):
    # get recs
    exclude_beers = ft[ft['User ID'].isin(users)][['User ID', 'Beer ID']]
    exclude_beers.columns = [['user_id', 'item_id']]
    exclude_beers = gl.SFrame(exclude_beers)
    recs = model.recommend(users=users, exclude=exclude_beers, diversity=3).to_dataframe()
    recs = recs.merge(beers, left_on='item_id', right_on='ID')[['user_id', 'item_id', 'Name', 'Style', 'Brewery Name']]
    recs['category'] = 'our rec'
    # get user preferences
    user_beers = user_beers.merge(beers, left_on='item_id', right_on='ID')[['user_id', 'item_id', 'Name', 'Style', 'Brewery Name']]
    user_beers['category'] = 'your pref'
    user_beers = user_beers[user_beers['user_id'].isin(users)]

    return pd.concat((user_beers, recs), axis=0).sort(['user_id', 'category'])

def load_rec_data():
    ftiso = pd.read_csv(
        'data/raw_data/ftiso.csv',
        header=None,
        names=[
            'ID',
            'Beer ID',
            'Quantity',
            'Cellar Quantity',
            'User ID',
            'Type',
            'Accessible List',
            'Created',
            'Modified']
    )
    ft = ftiso[ftiso['Type'] == 'ft']
    data = get.model_data(sparse=0,
                       outlier=10000000,
                       iso_rate=1,
                       proposed_rate=1,
                       traded_rate=1,
                       trade_path='data/raw_data/trades.csv',
                       trade_item_path='data/raw_data/trade_items.csv',
                       ftiso_path='data/raw_data/ftiso.csv')
    user_beers = pd.read_csv('data/model_data/all_user_beers')[['user_id', 'item_id']]
    beers = pd.read_csv('data/raw_data/beers.csv')

    return ft, user_beers, beers

if __name__ == '__main__':
    pass
