import logging
import os
import shutil

import numpy as np
import pandas as pd
import graphlab as gl

import process_data as get
import build_model as build

log = logging.getLogger(__name__)


# input data paths
TRADE_PATH = '../data/input/trades.csv'
TRADE_ITEM_PATH = '../data/input/trade_items.csv'
FTISO_PATH = '../data/input/ftiso.csv'
BEERS_PATH = '../data/input/beers.csv'
BREWERS_PATH = '../data/input/breweries.csv'
USERS_PATH = '../data/input/users.csv'


# intermediate csv output paths (used to build recommenders)
IS_DF_PATH = '../data/output/is_model.csv'
NMF_DF_PATH = '../data/output/nmf_model.csv'
POP_DF_PATH = '../data/output/pop_model.csv'


# model output paths
IS_MODEL_PATH = '../models/is_model'
NMF_MODEL_PATH = '../models/nmf_model'
POP_MODEL_PATH = '../models/pop_model'


# model output paths
IS_REC_PATH = '../data/output/is_recs.json'
NMF_REC_PATH = '../data/output/nmf_recs.json'
POP_REC_PATH = '../data/output/pop_recs.json'


def _load_data():
    """create and csvs for training the model and making recommendations"""
    # item similarity model csv
    get.model_data(sparse=4,
                    outlier=10000,
                    iso_rate=0,
                    proposed_rate=0,
                    traded_rate=1,
                    trade_path=TRADE_PATH,
                    trade_item_path=TRADE_ITEM_PATH,
                    ftiso_path=FTISO_PATH,
                    output_path=IS_DF_PATH)

    # get nmf model csv
    get.model_data(sparse=3,
                    outlier=500,
                    iso_rate=0,
                    proposed_rate=0,
                    traded_rate=1,
                    trade_path=TRADE_PATH,
                    trade_item_path=TRADE_ITEM_PATH,
                    ftiso_path=FTISO_PATH,
                    output_path=NMF_DF_PATH)


    # get popularity model csv
    get.model_data(sparse=0,
                    outlier=500,
                    iso_rate=0,
                    proposed_rate=0,
                    traded_rate=1,
                    trade_path=TRADE_PATH,
                    trade_item_path=TRADE_ITEM_PATH,
                    ftiso_path=FTISO_PATH,
                    output_path=POP_DF_PATH)


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


def get_precision_recall():
    """check the precision and recall for the best model"""
    model_df = gl.SFrame.read_csv(MODEL_DF_PATH)
    train, test = gl.recommender.util.random_split_by_user(gl.SFrame(model_df))
    model = gl.recommender.item_similarity_recommender.create(train)
    pr = model.evaluate(test, metric='precision_recall')
    return recommends, pr


def main():
    # create csvs to train models
    _load_data()
    is_df = gl.SFrame.read_csv(IS_DF_PATH)
    nmf_df = gl.SFrame.read_csv(NMF_DF_PATH)
    pop_df = gl.SFrame.read_csv(POP_DF_PATH)
    item_df = gl.SFrame(get.item_data(BEERS_PATH, BREWERS_PATH))
    # list of beers to exclude from recs, by user
    exclude_beers = pd.read_csv(FTISO_PATH)[['user_id', 'beer_id']]
    exclude_beers.columns = ['user_id', 'item_id']
    exclude_beers = gl.SFrame(exclude_beers)
    # build & save models
    build.is_model(is_df, IS_MODEL_PATH)
    build.nmf_model(nmf_df, item_df, NMF_MODEL_PATH)
    build.pop_model(pop_df, POP_MODEL_PATH)
    # load all models
    is_model = gl.load_model(IS_MODEL_PATH)
    nmf_model = gl.load_model(NMF_MODEL_PATH)
    pop_model = gl.load_model(POP_MODEL_PATH)
    # get recommendations & export as json
    users = gl.SFrame.read_csv(USERS_PATH)
    is_recs = is_model.recommend(users=users['id'],
                    exclude=exclude_beers,
                    diversity=3)
    nmf_recs = nmf_model.recommend(users=users['id'],
                    exclude=exclude_beers,
                    diversity=3)
    pop_recs = pop_model.recommend(users=users['id'],
                    exclude=exclude_beers,
                    diversity=3)
    # save recommendations
    is_recs.export_json(IS_REC_PATH, orient='records')
    nmf_recs.export_json(NMF_REC_PATH, orient='records')
    pop_recs.export_json(POP_REC_PATH, orient='records')


if __name__ == '__main__':
    main()
