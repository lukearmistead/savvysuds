import logging
import os
import shutil

import numpy as np
import pandas as pd
import graphlab as gl

import process_data as get


log = logging.getLogger(__name__)


def _save_model(output_path, model):
    """saves model in models folder after checking if model path exists"""
    if os.path.exists(output_path):
        shutil.rmtree(output_path)
    model.save(output_path)
    log.info('model saved as %s',output_path)


def nmf_model(df, item_df, output_path):
    """builds an nmf model from completed trades, including beer side data"""
    mat = gl.SFrame(df[['user_id', 'item_id']])
    model = gl.recommender.ranking_factorization_recommender.create(
        mat,
        item_data=item_df,
        nmf=False
    )
    _save_model(output_path, model)


def is_model(df, output_path):
    """creates and saves item similarity model"""
    mat = gl.SFrame(df[['user_id', 'item_id']])
    model = gl.recommender.item_similarity_recommender.create(mat)
    _save_model(output_path, model)


def pop_model(df, output_path):
    """creates and saves popularity model to be used as control"""
    mat = gl.SFrame(df[['user_id', 'item_id']])
    model = gl.recommender.popularity_recommender.create(mat)
    _save_model(output_path, model)
