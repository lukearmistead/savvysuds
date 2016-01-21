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


def model_data(sparse,
                outlier,
                iso_rate,
                proposed_rate,
                traded_rate,
                trade_path,
                trade_item_path,
                ftiso_path):
    """Return user/item/rating table excluding sparse and outlying users

    Arguments:
    - Ratings applied to represent user beer/interactions
      (in search of, proposed trade, completed trade)
    - Sparse & outlier: the min & max number of user/beer interactions
    """
    # build combined trade & iso dataframe with ratings
    trades, trade_items, ftiso = _load_rec_data(trade_path,
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
    return trimmed_rec_data


def _load_rec_data(trade_path, trade_item_path, ftiso_path):
    """
    load all data relevant to basic recommender and return as tuple of
    pandas dataframes
    """
    trades = pd.read_csv(trade_path)
    trade_items = pd.read_csv(trade_item_path)
    ftiso = pd.read_csv(
        ftiso_path,
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
    return trades, trade_items, ftiso


def _get_trade_info(trades, trade_items):
    """Merge trade_items with trades and add recipient column"""
    all_trades = pd.merge(
        trade_items,
        trades,
        left_on='Trade ID', right_on='ID', how='left')
    # recipient column finds user who wants the beer in trade_items
    all_trades['Recipient'] = np.where(
        all_trades['User ID'] == all_trades['User 1 ID'],
        all_trades['User 2 ID'], all_trades['User 1 ID']
    )
    all_trades = all_trades[[
        'User 1 ID', 'User 2 ID', 'User 1 Status', 'User 2 Status',
        'Recipient', 'Beer ID', 'User ID'
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
    rated_trades['Proposed Trade'] = (m_proposed | m_rescinded) * proposed_rate
    rated_trades['Completed Trade'] = m_completed * traded_rate
    # create rating column whose value is the maximum of the
    rated_trades['Rating'] = rated_trades[['Proposed Trade', 'Completed Trade']] \
                            .max(axis=1)
    rated_trades = rated_trades[rated_trades['Rating'] != 0]
    # narrow and rename columns
    rated_trades = rated_trades[['Recipient', 'Beer ID', 'Rating']]
    rated_trades.columns = ['user_id', 'item_id', 'rating']
    log.debug('trades found: %d proposed, %d rescinded, %d completed'
              % (sum(m_proposed), sum(m_rescinded), sum(m_completed)))
    return rated_trades


def _build_trade_mask_w_sender(all_trades, recipient, sender):
    """creates mask to filter proposed and completed trades
    only used in _get_trade_ratings when sender's status is
    relevant (3/0)
    """
    return (
        (all_trades['User 1 Status'] == recipient) &
        (all_trades['User 2 Status'] == sender) &
        (all_trades['Recipient'] == all_trades['User 1 ID'])
    ) | (
        (all_trades['User 2 Status'] == recipient) &
        (all_trades['User 1 Status'] == sender) &
        (all_trades['Recipient'] == all_trades['User 2 ID'])
        )


def _build_trade_mask(all_trades, recipient):
    """
    creates mask to filter proposed and completed trades
    only used in _get_trade_ratings
    when we only care about recipient's status
    """
    return (
        (all_trades['User 1 Status'] == recipient) &
        (all_trades['Recipient'] == all_trades['User 1 ID'])
    ) | (
        (all_trades['User 2 Status'] == recipient) &
        (all_trades['Recipient'] == all_trades['User 2 ID'])
    )


def _get_iso_ratings(ftiso, iso_rate):
    """find 'in search of' list for each user and assign iso_rate value"""
    iso = ftiso[ftiso['Type'] == 'iso'][['User ID', 'Beer ID']]
    rating = np.full(iso.count()[0], iso_rate)
    iso['Rating'] = rating
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


def item_data(beers_path='../data/raw_data/beers.csv',
              brewers_path='../data/raw_data/breweries.csv'):
    """preprocesses item data for recommender side data"""
    beers, brewers = _load_item_data(beers_path, brewers_path)
    # join all item tables and narrow to relevant columns
    all_items = beers.merge(brewers, left_on='Brewery ID', right_on='ID')
    print all_items.head(2)
    item_data = all_items[['ID_x', 'Style', 'ABV', 'Score_x', 'State', 'Type', 'Score_y', 'URL']]
    item_data.columns = ['item_id', 'beer_style', 'abv', 'beer_score', 'brewer_state', 'brewer_type', 'brewer_score', 'url']
    #url to binary
    item_data['url'] = item_data['url'].notnull()
    df = item_data
    return item_data


def _load_item_data(beers_path, brewers_path):
    """load all data for basic recommender and return as tuple of dfs"""
    beers = pd.read_csv(beers_path)
    brewers = pd.read_csv(brewers_path)
    return beers, brewers


if __name__ == '__main__':
    main()
