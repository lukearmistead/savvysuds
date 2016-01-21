import logging
import os
import shutil

import numpy as np
import pandas as pd
import graphlab as gl

import process_data as get
import check_model as check

log = logging.getLogger(__name__)

# class Model(object):
#     def __init__(self,
#     output_path,
#     model_path,
#     params
#     data,
#     items=None
#     ):
#         # output paths
#         self.output_path=output_path
#         # model & relevant params
#         self.params=params
#         self.model=None
#         # input paths
#         self.data=data
#         self.items=items


# def _save_model(output_path, model):
#     """saves model in models folder"""
#     if os.path.exists(output_path):
#         shutil.rmtree(output_path)
#     model.save(output_path)
#     log.info('model saved as %s',output_path)
#
def item_similarity(data,
                    model_path='models/item_similarity',
                    model_params={'similarity_type': 'jaccard',
                                  'verbose': False}
):
    """builds and saves item similarity model"""
    get.model_data(**data_params)
    mat = gl.SFrame.read_csv(data_params['output_path'])
    mat = mat[['user_id', 'item_id']]

    model = gl.recommender.item_similarity_recommender.create(
        mat,
        **model_params
        )
    return model

def factorization(
    model_path='models/simple_factorization',
    simple=True,
    use_item_data = True,
    item_data_output='data/model_data/item_data.csv',
    model_params={'nmf':False,
                  'verbose': False}
):
    """builds and saves factorization model"""
    get.model_data(**data_params)
    mat = gl.SFrame.read_csv(data_params['output_path'])
    if simple:
        mat = mat[['user_id', 'item_id']]
    if use_item_data:
        get.item_data(item_data_output)
        item_data = gl.SFrame.read_csv(item_data_output)
        model = gl.recommender.ranking_factorization_recommender.create(
            mat,
            item_data=item_data, **model_params
        )
    else:
        model = gl.recommender.ranking_factorization_recommender.create(
            mat,
            **model_params
        )
    save_model(model_path, model)
    log.info('saved model as %s' % model_path)

def main():
    get.model_data(**data_params)
    mat = gl.SFrame.read_csv(data_params['output_path'])


if __name__ == '__main__':
    item_sim = Model()
