import os

import numpy as np
import pandas as pd

# input data paths
TRADE_PATH = '../../data/input/trades.csv'
TRADE_ITEM_PATH = '../../data/input/trade_items.csv'
FTISO_PATH = '../../data/input/ftiso.csv'
BEERS_PATH = '../../data/input/beers.csv'
BREWERS_PATH = '../../data/input/breweries.csv'
USERS_PATH = '../../data/input/users.csv'

# intermediate csv output paths (used to build recommenders)
IS_DF_PATH = 'trade_graph.csv'

# trade status classifications
NO_RESPONSE = 0
TRADE_APPROVED = 1
BEER_RECEIVED = 2
TRADE_DECLINED = 3


def model_data(iso_rate,
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
    df = rated_trades[['user1_id', 'recipient', 'beer_id']]
    df.columns = ['Source', 'Target', 'beer_id']
    df.to_csv(output_path, index=False)


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


if __name__ == '__main__':
    df  = model_data(iso_rate=0,
                    proposed_rate=0,
                    traded_rate=1,
                    trade_path=TRADE_PATH,
                    trade_item_path=TRADE_ITEM_PATH,
                    ftiso_path=FTISO_PATH,
                    output_path=IS_DF_PATH)
