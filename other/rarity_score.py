from __future__ import division

import matplotlib.pyplot as plt
import pandas as pd
import process_data as get
from scipy import stats


FTISO_PATH = '../data/input/ftiso.csv'
BEERS_PATH = '../data/input/beers.csv'
BREWERIES_PATH = '../data/input/breweries.csv'
DISTRIBUTION_PATH = '../data/input/beer_distribution.csv'
SCORE_WEIGHTS = {
    'ratio_score': 0.3,
    'iso_score': 0.3,
    'untappd_score': 0.3,
    'geo_score': 0.0,
    'production_score': 0.1
    }

def weighted_percentile(vector):
    """returns percentile values of a vector
    calculates the percentile of each element of data, with duplicate
    values set to the mean of the chunk of the distribution they occupy
    http://stackoverflow.com/questions/12414043/map-each-list-value-to-its-corresponding-percentile
    """
    return stats.rankdata(vector) / len(vector)


def feature_scale(df, col_name):
    """returns values scaled to 0-1 range as pandas series
    not used
    """
    score_diff = df[col_name] - df[col_name].min()
    norm_score = score_diff / (df[col_name].max() - df[col_name].min())
    return norm_score


def wishlist_scores(data_path):
    '''returns iso/ft ratio score & iso count ratio score as dataframes
    see weighted_percentile for methodology
    higher ratio or iso count means higher score
    '''
    # load data, then calculate iso & ft counts
    ftiso = pd.read_csv(data_path)
    iso = ftiso[ftiso['type'] == 'iso']
    iso = iso.groupby('beer_id', as_index=False).count()
    ft = ftiso[ftiso['type'] == 'ft']
    ft = ft.groupby('beer_id', as_index=False).count()
    # get ratio score
    ft['ratio'] = iso['id'] / ft['id']
    ft['ratio_score'] = weighted_percentile(ft['ratio'])
    # get iso score
    iso['iso_score'] = weighted_percentile(iso['id'])
    return ft[['beer_id', 'ratio_score']], iso[['beer_id', 'iso_score']]


def distribution_scores(data_path):
    """returns distribution scores as dataframe
    scores calculated based on how number of states fall
    distributing to fewer states results in a higher score
    """
    distribution = pd.read_csv(data_path)
    distribution['states'] = distribution \
        .drop(['brewery_id', 'Name'], axis=1) \
        .apply(lambda x: (x != 'N').sum(), axis=1)
    distribution['geo_score'] = weighted_percentile(distribution['states'])
    # inverts distribution scores to punish higher geographic footprint
    distribution['geo_score'] = 1 - distribution['geo_score']
    return distribution[['brewery_id', 'geo_score']]


def production_scores(data_path):
    """scores breweries based on production
    higher production results in a lower score
    """
    breweries = pd.read_csv(data_path)
    breweries['production_score'] = weighted_percentile(
        breweries['beer_count']
        )
    # inverts distribution scores to punish higher production
    breweries['production_score'] = 1 - breweries['production_score']
    return breweries[['id', 'production_score']]


def untappd_scores(data_path):
    """scores beers based on untappd ratings"""
    beers = pd.read_csv(data_path)
    beers['untappd_score'] = weighted_percentile(beers['score'])
    return beers[['id', 'untappd_score']]


def agg_weighted_score(rarity, score_weights, drop_nas=True):
    """returns beer name, id, & score based on the inputed weights
    input - weights as a dictionary
    output - weighted valuation
    """
    rarity['score'] = 0
    if drop_nas:
        rarity = rarity.dropna()
    for category, weight in score_weights.iteritems():
        rarity['score'] += rarity[category] * weight
    rarity['score'] = rarity['score'] / sum(score_weights.values())
    return rarity


def main():
    """returns a dataframe with scores from 0-1 based on iso count, ft/iso,
    distribution, production, and untappd ratings
    note that all hardcodes at beginning of file flow here
    """
    beers = pd.read_csv(BEERS_PATH)
    # initialize output dataframe
    rarity = beers[['name', 'id', 'brewery_id']]
    # fix column names to prevent redundancies during the merges
    rarity.columns = ['name', 'beer', 'brewery']
    # get all scores
    ft, iso = wishlist_scores(FTISO_PATH)
    distribution = distribution_scores(DISTRIBUTION_PATH)
    breweries = production_scores(BREWERIES_PATH)
    beers = untappd_scores(BEERS_PATH)
    # merge all dfs
    beer_scores = [
        (ft, 'beer'),
        (iso, 'beer'),
        (beers, 'beer'),
        (distribution, 'brewery'),
        (breweries, 'brewery')
        ]
    for df, rarity_idx in beer_scores:
        rarity = rarity.merge(df, how='left', left_on=rarity_idx,
            right_on=df.columns[0]).drop(df.columns[0], axis=1)
    agg_rarity = agg_weighted_score(rarity, SCORE_WEIGHTS, drop_nas=True)
    return agg_rarity


if __name__ == '__main__':
    rarity = main()
