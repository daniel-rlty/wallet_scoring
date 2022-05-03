from pycoingecko import CoinGeckoAPI
from pandas import json_normalize
import numpy as np


def get_top_coins():
    """
    get_top_coins get the ethereum erc20 tokens where market cap rank is greater than max_rank on coingecko


    :param max_rank: max_rank for erc20 token on coingecko
    :type max_rank: integer
    :return: dataframe with erc20 id, symbol, name and contract address
    :rtype: dataframe
    """
    cg = CoinGeckoAPI()
    lof_coins = cg.get_coins_list(include_platform=True)
    lof_eth_coins = [x for x in lof_coins if 'ethereum' in x['platforms']]
    lof_eth_coins_w_address = [x for x in lof_eth_coins if len(x['platforms']['ethereum'])]
    df = json_normalize(lof_eth_coins_w_address)[['id', 'symbol', 'name', 'platforms.ethereum']]
    df = df.rename(columns={'platforms.ethereum': 'token_address'})
    df['token_address'] = df['token_address'].apply(lambda x: x.lower())
    ranges = np.arange(0, len(df), 500)
    lof_market_cap = []
    for i in np.arange(len(ranges) - 1):
        ids = ','.join(df['id'].iloc[ranges[i]:ranges[(i + 1)]].values)
        lof_market_cap = lof_market_cap + cg.get_coins_markets(vs_currency='eth', ids=ids, order='id_asc')

    ids = ','.join(df['id'].iloc[ranges[(-1)]:].values)
    lof_market_cap = lof_market_cap + cg.get_coins_markets(vs_currency='eth', ids=ids, order='id_asc')
    df_mc = json_normalize(lof_market_cap)[['id', 'market_cap_rank', 'current_price']]
    df_mc = df_mc.merge(df, how='inner', on=['id'])
    res = df_mc
    return res


df_cg = get_top_coins()