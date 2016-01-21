import logging
import os
import shutil

import numpy as np
import pandas as pd
import graphlab as gl

import build_model as build
import process_data as get
import check_model as check


log = logging.getLogger(__name__)


TRADE_PATH = '../data/raw_data/trades.csv'
TRADE_ITEM_PATH = '../data/raw_data/trade_items.csv'
FTISO_PATH = '../data/raw_data/ftiso.csv'
MODEL_PATH = '../models/item_similarity_model'


def save_df(df, output_path):
    """saves a dataframe as a csv"""
    # check for data
    if os.path.exists(output_path):
        log.warn('%s already exists' % output_path)
        return
    df.to_csv(output_path)
    log.info('saved dataframe as %s' % output_path)


def save_model(output_path, model):
    """saves model in models folder"""
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
    model = gl.recommender.ranking_factorization_recommender.create(mat, item_data=item_data, nmf=False)
    # check recs
    users = [3381, 14239, 6601, 8958, 1440, 880, 57]
    ft, user_beers, beers = check.load_rec_data()
    recommends = check.recs(users, model, ft, user_beers, beers)
    # # check precision & recall
    # train, test = gl.recommender.util.random_split_by_user(gl.SFrame(user_beers))
    # model = gl.recommender.ranking_factorization_recommender.create(train, item_data=None, nmf=False, verbose=False)
    # pr = model.evaluate(test, metric='precision_recall')
    return model, recommends#, pr


def main():
    """builds an item similarity model from completed trades, including beer side data"""
    # load data
    # if exists thing
    df = get.model_data(sparse=4,
                        outlier=10000,
                        iso_rate=0,
                        proposed_rate=0,
                        traded_rate=1,
                        trade_path=TRADE_PATH,
                        trade_item_path=TRADE_ITEM_PATH,
                        ftiso_path=FTISO_PATH)
    mat = gl.SFrame(df[['user_id', 'item_id']])
    # build model
    model = gl.recommender.item_similarity_recommender.create(mat)
    # save model
    save_model(MODEL_PATH, model)


if __name__ == '__main__':
    main()
