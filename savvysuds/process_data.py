import argparse
import logging
import os

import numpy as np
import pandas as pd


log = logging.getLogger(__name__)


# trade status classifications
NO_RESPONSE = 0
TRADE_APPROVED = 1
BEER_RECEIVED = 2
TRADE_DECLINED = 3


###############################################
### FUNCTIONS USED TO PREPROCESS MODEL DATA ###
###############################################
def model_data(sparse,
                outlier,
                iso_rate,
                proposed_rate,
                traded_rate,
                trade_path,
                trade_item_path,
                ftiso_path,
                output_path):
    """Return user/item/rating table excluding sparse and outlying users

    Arguments:
    - ratings applied to represent user beer/interactions
      (in search of, proposed_trade, completed_trade)
    - Sparse & outlier: the min & max number of user/beer interactions
    """
    # build combined trade & iso dataframe with ratings
    trades, trade_items, ftiso = _load_csvs(trade_path,
                                           trade_item_path,
                                           ftiso_path)
    all_trades = _get_trade_info(trades, trade_items)
    rated_trades = _get_trade_ratings(all_trades,
                                      proposed_rate=proposed_rate,
                                      traded_rate=traded_rate)
    iso = _get_iso_ratings(ftiso, iso_rate)
    rec_data = pd.concat((rated_trades, iso), axis=0)
    #drop duplicates & trim sparse/outlying data points
    grouped_rec_data = rec_data.groupby(['user_id', 'item_id'], as_index=False).max()
    trimmed_rec_data = _drop_sparse_or_outliers(grouped_rec_data,
                                                sparse, outlier)
    df = trimmed_rec_data
    log.debug('built df with %d observations' % df.count()[0])
    _save_df(df, output_path)


def _load_csvs(trade_path, trade_item_path, ftiso_path):
    """
    load all data relevant to basic recommender and return as tuple of
    pandas dataframes
    """
    trades = pd.read_csv(trade_path)
    trades.columns = ["id", "user1_id", "user2_id", "user1_status", "user2_status", "decline_reason", "user1_address", "user2_address", "user1_tracking", "user2_tracking", "created", "updated"]
    trade_items = pd.read_csv(trade_item_path)
    trade_items.columns = ["id", "trade_id", "user_id", "beer_id", "quantity", "created", "updated"]
    ftiso = pd.read_csv(ftiso_path)
    ftiso.columns = ["id", "beer_id", "quantity", "cellar_quantity", "user_id", "type", "accessible_list", "created", "modified"]
    return trades, trade_items, ftiso


def _get_trade_info(trades, trade_items):
    """Merge trade_items with trades and add recipient column"""
    all_trades = pd.merge(
        trade_items,
        trades,
        left_on='trade_id', right_on='id', how='left')
    # recipient column finds user who wants the beer in trade_items
    all_trades['recipient'] = np.where(
        all_trades['user_id'] == all_trades['user1_id'],
        all_trades['user2_id'], all_trades['user1_id']
    )
    all_trades = all_trades[[
        'user1_id', 'user2_id', 'user1_status', 'user2_status',
        'recipient', 'beer_id', 'user_id'
    ]]
    log.debug('merged trade_items and trades into %d observation df, '
              'added recipient column', all_trades.count()[0])
    return all_trades


def _get_trade_ratings(all_trades, proposed_rate, traded_rate):
    """constructs dataframe with ratings assigned by trade status
    calls _build_trade_mask_w_sender & _build_trade_mask to identify
    types of trades
    """
    # build masks
    m_proposed = _build_trade_mask(all_trades, recipient=TRADE_APPROVED)
    m_rescinded = _build_trade_mask_w_sender(
        all_trades, recipient=TRADE_DECLINED, sender=NO_RESPONSE)
    m_completed = _build_trade_mask(all_trades, recipient=BEER_RECEIVED)

    rated_trades = all_trades

    # populate columns using mask filters
    rated_trades['proposed_trade'] = (m_proposed | m_rescinded) * proposed_rate
    rated_trades['completed_trade'] = m_completed * traded_rate
    # create rating column whose value is the maximum of the
    rated_trades['rating'] = rated_trades[['proposed_trade', 'completed_trade']] \
                            .max(axis=1)
    rated_trades = rated_trades[rated_trades['rating'] != 0]
    # narrow and rename columns
    rated_trades = rated_trades[['recipient', 'beer_id', 'rating']]
    rated_trades.columns = ['user_id', 'item_id', 'rating']
    log.debug('trades found: %d proposed, %d rescinded, %d completed'
              % (sum(m_proposed), sum(m_rescinded), sum(m_completed)))
    return rated_trades


def _build_trade_mask_w_sender(all_trades, recipient, sender):
    """creates mask to filter proposed and completed_trades
    only used in _get_trade_ratings when sender's status is
    relevant (3/0)
    """
    return (
        (all_trades['user1_status'] == recipient) &
        (all_trades['user2_status'] == sender) &
        (all_trades['recipient'] == all_trades['user1_id'])
    ) | (
        (all_trades['user2_status'] == recipient) &
        (all_trades['user1_status'] == sender) &
        (all_trades['recipient'] == all_trades['user2_id'])
        )


def _build_trade_mask(all_trades, recipient):
    """
    creates mask to filter proposed and completed_trades
    only used in _get_trade_ratings
    when we only care about recipient's status
    """
    return (
        (all_trades['user1_status'] == recipient) &
        (all_trades['recipient'] == all_trades['user1_id'])
    ) | (
        (all_trades['user2_status'] == recipient) &
        (all_trades['recipient'] == all_trades['user2_id'])
    )


def _get_iso_ratings(ftiso, iso_rate):
    """find 'in search of' list for each user and assign iso_rate value"""
    iso = ftiso[ftiso['type'] == 'iso'][['user_id', 'beer_id']]
    rating = np.full(iso.count()[0], iso_rate)
    iso['rating'] = rating
    iso.columns = [['user_id', 'item_id', 'rating']]
    iso = iso[iso['rating'] != 0]
    log.debug('found %d isos' % iso.count()[0])
    return iso


def _drop_sparse_or_outliers(rec_data, sparse, outlier):
    """drops users with <= sparse beers and >= outlier beers on profile"""
    beers_per_user = rec_data.groupby('user_id').item_id.transform(len)
    m_sparse = beers_per_user >= sparse
    m_outlier = beers_per_user <= outlier
    trimmed_rec_data = rec_data[m_sparse & m_outlier]
    log.debug('dropped %d sparse users, %d outliers'
              % (sum(m_sparse * 1), sum(m_outlier * 1)))
    return trimmed_rec_data


def _save_df(df, output_path):
    """saves a dataframe as a csv after checking if df of that name exists"""
    # check for data
    if os.path.exists(output_path):
        log.warn('%s already exists' % output_path)
        return
    df.to_csv(output_path)
    log.info('saved dataframe as %s' % output_path)


###############################################
### FUNCTIONS USED TO PREPROCESS ITEM DATA  ###
###############################################
def item_data(beers_path, brewers_path):
    """preprocesses item data to be included in nmf analysis as side data"""
    beers, brewers = _load_item_data(beers_path, brewers_path)
    # join all item tables and narrow to relevant columns
    all_items = beers.merge(brewers, left_on='brewery_id', right_on='id')
    item_data = all_items[['id_x', 'style', 'abv', 'score_x', 'state', 'type', 'score_y', 'url']]
    item_data.columns = ['item_id', 'beer_style', 'abv', 'beer_score', 'brewer_state', 'brewer_type', 'brewer_score', 'url']
    #url to binary
    item_data['url'] = item_data['url'].notnull()
    return item_data.drop('brewer_state', axis=1)


def _load_item_data(beers_path, brewers_path):
    """load all data for basic recommender and return as tuple of dfs"""
    beers = pd.read_csv(beers_path)
    beers.columns = ["id", "name", "style", "description", "abv", "label", "score", "brewery_id", "brewery_name", "created", "modified"]
    brewers = pd.read_csv(brewers_path)
    brewers.columns = ["id", "name", "type", "description", "label", "address", "city", "state", "country", "lat", "lng", "beer_count", "twitter", "facebook", "url", "score", "created", "modified"]
    return beers, brewers


###############################################
###  FUNCTIONS USED TO PREPROCESS REC DATA  ###
###############################################
def rec_data(ftiso_path, trade_path, trade_item_path, beers_path):
    """Loads data to make usable, readable recommendations in main.py"""
    ftiso = pd.read_csv(ftiso_path)
    ftiso.columns = ["id", "beer_id", "quantity", "cellar_quantity", "user_id", "type", "accessible_list", "created", "modified"]
    ft = ftiso[ftiso['type'] == 'ft']
    user_beers = model_data(sparse=0,
                           outlier=10000000,
                           iso_rate=1,
                           proposed_rate=1,
                           traded_rate=1,
                           trade_path=trade_path,
                           trade_item_path=trade_item_path,
                           ftiso_path=ftiso_path)
    user_beers = user_beers[['user_id', 'item_id']]
    beers = pd.read_csv(beers_path)
    beers.columns = ["id", "name", "style", "description", "abv", "label", "score", "brewery_id", "brewery_name", "created", "modified"]
    return ft, user_beers, beers
