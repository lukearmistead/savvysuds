import logging
import os
import shutil

import numpy as np
import pandas as pd
import graphlab as gl

import build_model as build
import process_data as get
import check_model as check

def save_df(df, output_path):
    """saves a dataframe... duh"""
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


def is_iso_prop():
    # load data
    df = get.model_data(sparse=4,
                        outlier=500,
                        iso_rate=1,
                        proposed_rate=1,
                        traded_rate=1,
                        trade_path='data/raw_data/trades.csv',
                        trade_item_path='data/raw_data/trade_items.csv',
                        ftiso_path='data/raw_data/ftiso.csv')
    mat = gl.SFrame(df[['user_id', 'item_id']])
    # build model
    model = gl.recommender.item_similarity_recommender.create(mat)
    # check recs
    users = [3381, 14239, 6601, 8958, 1440, 880, 57]
    ft, user_beers, beers = check.load_rec_data()
    recommends = check.recs(users, model, ft, user_beers, beers)
    # # check precision & recall
    # train, test = gl.recommender.util.random_split_by_user(gl.SFrame(user_beers))
    # model = gl.recommender.item_similarity_recommender.create(train)
    # pr = model.evaluate(test, metric='precision_recall')
    return model, recommends#, pr


def is_niso_prop():
    # load data
    df = get.model_data(sparse=4,
                        outlier=500,
                        iso_rate=0,
                        proposed_rate=1,
                        traded_rate=1,
                        trade_path='data/raw_data/trades.csv',
                        trade_item_path='data/raw_data/trade_items.csv',
                        ftiso_path='data/raw_data/ftiso.csv')
    mat = gl.SFrame(df[['user_id', 'item_id']])
    # build model
    model = gl.recommender.item_similarity_recommender.create(mat)
    # check recs
    users = [3381, 14239, 6601, 8958, 1440, 880, 57]
    ft, user_beers, beers = check.load_rec_data()
    recommends = check.recs(users, model, ft, user_beers, beers)
    # # check precision & recall
    # train, test = gl.recommender.util.random_split_by_user(gl.SFrame(user_beers))
    # model = gl.recommender.item_similarity_recommender.create(train)
    # pr = model.evaluate(test, metric='precision_recall')
    return model, recommends#, pr


def is_niso_nprop():
    # load data
    df = get.model_data(sparse=4,
                        outlier=500,
                        iso_rate=0,
                        proposed_rate=0,
                        traded_rate=1,
                        trade_path='data/raw_data/trades.csv',
                        trade_item_path='data/raw_data/trade_items.csv',
                        ftiso_path='data/raw_data/ftiso.csv')
    mat = gl.SFrame(df[['user_id', 'item_id']])
    # build model
    model = gl.recommender.item_similarity_recommender.create(mat)
    # check recs
    users = [3381, 14239, 6601, 8958, 1440, 880, 57]
    ft, user_beers, beers = check.load_rec_data()
    recommends = check.recs(users, model, ft, user_beers, beers)
    # # check precision & recall
    # train, test = gl.recommender.util.random_split_by_user(gl.SFrame(user_beers))
    # model = gl.recommender.item_similarity_recommender.create(train)
    # pr = model.evaluate(test, metric='precision_recall')
    return model, recommends#, pr


def mf_iso_prop():
    # load data
    df = get.model_data(sparse=4,
                        outlier=500,
                        iso_rate=1,
                        proposed_rate=1,
                        traded_rate=1,
                        trade_path='data/raw_data/trades.csv',
                        trade_item_path='data/raw_data/trade_items.csv',
                        ftiso_path='data/raw_data/ftiso.csv')
    mat = gl.SFrame(df[['user_id', 'item_id']])
    # build model
    model = gl.recommender.ranking_factorization_recommender.create(mat)
    # check recs
    users = [3381, 14239, 6601, 8958, 1440, 880, 57]
    ft, user_beers, beers = check.load_rec_data()
    recommends = check.recs(users, model, ft, user_beers, beers)
    # # check precision & recall
    # train, test = gl.recommender.util.random_split_by_user(gl.SFrame(user_beers))
    # model = gl.recommender.ranking_factorization_recommender.create(train, item_data=None, nmf=False, verbose=False)
    # pr = model.evaluate(test, metric='precision_recall')
    return model, recommends#, pr


def mf_niso_prop():
    # load data
    df = get.model_data(sparse=4,
                        outlier=500,
                        iso_rate=0,
                        proposed_rate=1,
                        traded_rate=1,
                        trade_path='data/raw_data/trades.csv',
                        trade_item_path='data/raw_data/trade_items.csv',
                        ftiso_path='data/raw_data/ftiso.csv')
    mat = gl.SFrame(df[['user_id', 'item_id']])
    # build model
    model = gl.recommender.ranking_factorization_recommender.create(mat)
    # check recs
    users = [3381, 14239, 6601, 8958, 1440, 880, 57]
    ft, user_beers, beers = check.load_rec_data()
    recommends = check.recs(users, model, ft, user_beers, beers)
    # # check precision & recall
    # train, test = gl.recommender.util.random_split_by_user(gl.SFrame(user_beers))
    # model = gl.recommender.ranking_factorization_recommender.create(train, item_data=None, nmf=False, verbose=False)
    # pr = model.evaluate(test, metric='precision_recall')
    return model, recommends#, pr


def mf_niso_nprop():
    # load data
    df = get.model_data(sparse=4,
                        outlier=500,
                        iso_rate=0,
                        proposed_rate=0,
                        traded_rate=1,
                        trade_path='data/raw_data/trades.csv',
                        trade_item_path='data/raw_data/trade_items.csv',
                        ftiso_path='data/raw_data/ftiso.csv')
    mat = gl.SFrame(df[['user_id', 'item_id']])
    # build model
    model = gl.recommender.ranking_factorization_recommender.create(mat)
    # check recs
    users = [3381, 14239, 6601, 8958, 1440, 880, 57]
    ft, user_beers, beers = check.load_rec_data()
    recommends = check.recs(users, model, ft, user_beers, beers)
    # # check precision & recall
    # train, test = gl.recommender.util.random_split_by_user(gl.SFrame(user_beers))
    # model = gl.recommender.ranking_factorization_recommender.create(train, item_data=None, nmf=False, verbose=False)
    # pr = model.evaluate(test, metric='precision_recall')
    return model, recommends#, pr

def fancy():
    # load data
    df = get.model_data(sparse=4,
                        outlier=500,
                        iso_rate=0,
                        proposed_rate=0,
                        traded_rate=1,
                        trade_path='data/raw_data/trades.csv',
                        trade_item_path='data/raw_data/trade_items.csv',
                        ftiso_path='data/raw_data/ftiso.csv')

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

if __name__ == '__main__':
    # path = 'blah blah'
    # is_1 = is_iso_prop()[0]
    # is_1.save('is_1')
    # # is_2 = is_niso_prop()
    # is_3 = is_niso_nprop()
    # # mf_1 = mf_iso_prop()
    # # mf_2 = mf_niso_prop()
    # mf_3 = mf_niso_nprop()[0]
    #
    # is_1 = is_iso_prop()[0]
    # is_1.save('models/is_1')
    recs = fancy()[1]
    recs.to_csv('thefanciness.csv')
    # fancy.save('models/fancy')


# out = {'is_1': is_1,
# 'is_2': is_2,
# 'is_3': is_3,
# 'mf_1': mf_1,
# 'mf_2': mf_2,
# 'mf_3': mf_3}
    # model, recs = fancy()
# [v[0].to_csv(k + '_recs.csv') for k, v in out.iteritems()]
# [v[1]['precision_recall_overall'][5:10] for k, v in out.iteritems()]
# [k for k, v in out.iteritems()]
