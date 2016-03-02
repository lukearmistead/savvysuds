import csv
import logging
import os
import shutil

import numpy as np
import pandas as pd
import graphlab as gl

import process_data as get
import build_model as build


def is_model(df, output_path):
    """creates and saves best item similarity model"""
    mat = gl.SFrame(df[['user_id', 'item_id']])
    model = gl.recommender.item_similarity_recommender.create(mat)
    _save_model(output_path, model)

def nmf_model(df, output_path):
    """builds an nmf model from completed trades, including beer side data"""
    # load data
    # df = get.model_data(sparse=4,
    #                     outlier=500,
    #                     iso_rate=0,
    #                     proposed_rate=0,
    #                     traded_rate=1,
    #                     trade_path=TRADE_PATH,
    #                     trade_item_path=TRADE_ITEM_PATH,
    #                     ftiso_path=FTISO_PATH)
    mat = gl.SFrame(df[['user_id', 'item_id']])
    item_data = gl.SFrame(get.item_data(BEERS_PATH, BREWERS_PATH))
    # build model
    model = gl.recommender.ranking_factorization_recommender.create(
        mat,
        item_data=item_data,
        nmf=False
    )
    _save_model(output_path, model)


def _save_model(output_path, model):
    """saves model in models folder after checking if model path exists"""
    if os.path.exists(output_path):
        shutil.rmtree(output_path)
    model.save(output_path)
    log.info('model saved as %s',output_path)
