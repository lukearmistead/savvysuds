import csv
import logging
import os
import shutil

import numpy as np
import pandas as pd
import graphlab as gl

import process_data as get


log = logging.getLogger(__name__)


# input data paths
TRADE_PATH = '../data/input/trades.csv'
TRADE_ITEM_PATH = '../data/input/trade_items.csv'
FTISO_PATH = '../data/input/ftiso.csv'
BEERS_PATH = '../data/input/beers.csv'
BREWERS_PATH = '../data/input/beers.csv'


# output model and data paths
MODEL_DF_PATH = '../data/output/model.csv'
ALL_USER_BEERS_PATH = '../data/output/user_beers.csv'
RECOMMEND_PATH = '../data/output/recs.json'
IS_MODEL_PATH = '../models/item_similarity_model'
NMF_MODEL_PATH = '../models/item_similarity_model'


def _load_data():
    """create and csvs for training the model and making recommendations"""
    get.model_data(sparse=4,
                    outlier=10000,
                    iso_rate=0,
                    proposed_rate=0,
                    traded_rate=1,
                    trade_path=TRADE_PATH,
                    trade_item_path=TRADE_ITEM_PATH,
                    ftiso_path=FTISO_PATH,
                    output_path=MODEL_DF_PATH)

    # get lists of users with all of their beers & beers to exclude
    get.model_data(sparse=0,
                    outlier=100000000,
                    iso_rate=1,
                    proposed_rate=1,
                    traded_rate=1,
                    trade_path=TRADE_PATH,
                    trade_item_path=TRADE_ITEM_PATH,
                    ftiso_path=FTISO_PATH,
                    output_path=ALL_USER_BEERS_PATH)


def _save_model(output_path, model):
    """saves model in models folder after checking if model path exists"""
    if os.path.exists(output_path):
        shutil.rmtree(output_path)
    model.save(output_path)
    log.info('model saved as %s',output_path)


def check_recs(users, model, ft, user_beers, beers):
    """gets model recommendations for a list of users (inputted by ids) in
    a readable, usable format, including metadata bout the beers"""
    # get metadata about
    exclude_beers = ft[ft['user_id'].isin(users)][['user_id', 'beer_id']]
    exclude_beers.columns = [['user_id', 'item_id']]
    exclude_beers = gl.SFrame(exclude_beers)
    recs = model.recommend(users=users, exclude=exclude_beers, diversity=3) \
        .to_dataframe()
    recs = recs.merge(beers, left_on='item_id', right_on='id') \
        [['user_id', 'item_id', 'name', 'style', 'brewery_name']]
    print recs.head()
    recs['category'] = 'our rec'
    # get user preferences
    user_beers = user_beers.merge(beers, left_on='item_id', right_on='id') \
        [['user_id', 'item_id', 'name', 'style', 'brewery_name']]
    user_beers['category'] = 'your pref'
    user_beers = user_beers[user_beers['user_id'].isin(users)]
    return pd.concat((user_beers, recs), axis=0).sort(['user_id', 'category'])


def recs(users, model, ft, user_beers, beers):
    exclude_beers = ft[ft['user_id'].isin(users)][['user_id', 'beer_id']]
    exclude_beers.columns = [['user_id', 'item_id']]


def nmf_model(output_path, df):
    """builds an nmf model from completed trades, including beer side data"""
    item_data = gl.SFrame(get.item_data(BEERS_PATH, BREWERS_PATH))
    # build model
    model = gl.recommender.ranking_factorization_recommender.create(
        mat,
        item_data=item_data,
        nmf=False
    )
    _save_model(output_path, model)


def is_model(df, output_path):
    """creates and saves best item similarity model"""
    mat = gl.SFrame(df[['user_id', 'item_id']])
    model = gl.recommender.item_similarity_recommender.create(mat)
    _save_model(IS_MODEL_PATH, model)


def get_precision_recall():
    """check the precision and recall for the best model"""
    model_df = gl.SFrame.read_csv(MODEL_DF_PATH)
    train, test = gl.recommender.util.random_split_by_user(gl.SFrame(model_df))
    model = gl.recommender.item_similarity_recommender.create(train)
    pr = model.evaluate(test, metric='precision_recall')
    return recommends, pr


def main():
    # create csvs to train model & generate recommendations
    _load_data()
    model_df = gl.SFrame.read_csv(MODEL_DF_PATH)
    rec_df = gl.SFrame.read_csv(ALL_USER_BEERS_PATH)
    exclude_beers = pd.read_csv(FTISO_PATH)[['user_id', 'beer_id']]
    exclude_beers.columns = ['user_id', 'item_id']
    exclude_beers = gl.SFrame(exclude_beers)

    # build, save, and load model
    is_model(model_df, IS_MODEL_PATH)
    model = gl.load_model(IS_MODEL_PATH)

    # get recommendations & export as json
    recs = model.recommend(users=rec_df['user_id'],
                    # new_observation_data=rec_df,
                    exclude=exclude_beers,
                    diversity=3)
    recs.export_json(RECOMMEND_PATH, orient='records')


if __name__ == '__main__':
    main()
