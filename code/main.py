import logging
import os
import shutil

import numpy as np
import pandas as pd
import graphlab as gl

import process_data as get


log = logging.getLogger(__name__)


MODEL_PATH = '../models/item_similarity_model'
TRADE_PATH = '../data/raw_data/trades.csv'
TRADE_ITEM_PATH = '../data/raw_data/trade_items.csv'
FTISO_PATH = '../data/raw_data/ftiso.csv'
BEERS_PATH = '../data/raw_data/beers.csv'


def save_df(df, output_path):
    """saves a dataframe as a csv after checking if df of that name exists"""
    # check for data
    if os.path.exists(output_path):
        log.warn('%s already exists' % output_path)
        return
    df.to_csv(output_path)
    log.info('saved dataframe as %s' % output_path)


def save_model(output_path, model):
    """saves model in models folder after checking if model path exists"""
    if os.path.exists(output_path):
        shutil.rmtree(output_path)
    model.save(output_path)
    log.info('model saved as %s',output_path)


def build_nmf_model():
    """builds an nmf model from completed trades, including beer side data"""
    # load data
    df = get.model_data(sparse=4,
                        outlier=500,
                        iso_rate=0,
                        proposed_rate=0,
                        traded_rate=1,
                        trade_path=TRADE_PATH,
                        trade_item_path=TRADE_ITEM_PATH,
                        ftiso_path=FTISO_PATH)
    mat = gl.SFrame(df[['user_id', 'item_id']])
    item_data = gl.SFrame(get.item_data())
    # build model
    model = gl.recommender.ranking_factorization_recommender.create(
        mat,
        item_data=item_data,
        nmf=False
    )
    return model


def check_recs(users, model, ft, user_beers, beers):
    """gets model recommendations for a list of users (inputted by ids) in
    a readable, usable format, including metadata bout the beers"""
    # get metadata about
    exclude_beers = ft[ft['User ID'].isin(users)][['User ID', 'Beer ID']]
    exclude_beers.columns = [['user_id', 'item_id']]
    exclude_beers = gl.SFrame(exclude_beers)
    recs = model.recommend(users=users, exclude=exclude_beers, diversity=3) \
        .to_dataframe()
    recs = recs.merge(beers, left_on='item_id', right_on='ID') \
        [['user_id', 'item_id', 'Name', 'Style', 'Brewery Name']]
    recs['category'] = 'our rec'
    # get user preferences
    user_beers = user_beers.merge(beers, left_on='item_id', right_on='ID') \
        [['user_id', 'item_id', 'Name', 'Style', 'Brewery Name']]
    user_beers['category'] = 'your pref'
    user_beers = user_beers[user_beers['user_id'].isin(users)]
    return pd.concat((user_beers, recs), axis=0).sort(['user_id', 'category'])


def main():
    """builds an item similarity model from completed trades
    also samples recommendations and evaluates precision/recall of model"""
    # build and save model
    df = get.model_data(sparse=4,
                        outlier=10000,
                        iso_rate=0,
                        proposed_rate=0,
                        traded_rate=1,
                        trade_path=TRADE_PATH,
                        trade_item_path=TRADE_ITEM_PATH,
                        ftiso_path=FTISO_PATH)
    mat = gl.SFrame(df[['user_id', 'item_id']])
    model = gl.recommender.item_similarity_recommender.create(mat)
    save_model(MODEL_PATH, model)
    # get recommendations for selected users
    ft, user_beers, beers = get.rec_data(FTISO_PATH,
                                          TRADE_PATH,
                                          TRADE_ITEM_PATH,
                                          BEERS_PATH)
    users = [3381, 14239, 6601, 8958, 1440, 880, 57]
    recommends = check_recs(users, model, ft, user_beers, beers)
    # find precision and recall for model
    train, test = gl.recommender.util.random_split_by_user(gl.SFrame(user_beers))
    model = gl.recommender.item_similarity_recommender.create(train)
    pr = model.evaluate(test, metric='precision_recall')
    return recommends, pr


if __name__ == '__main__':
    main()
