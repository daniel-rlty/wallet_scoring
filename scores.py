import pandas as pd
import matplotlib.pyplot as plt
import requests
import api_tags
import logs
import os
import coingecko_api
import numpy as np
from web3 import Web3
from collections import Counter
from pandas import json_normalize
from dateutil import parser
from datetime import datetime, timezone
from dotenv import load_dotenv

load_dotenv()

ALCHEMY_KEY = os.getenv('ALCHEMY_KEY')
MORALIS_KEY = os.getenv('MORALIS_KEY')
OPENSEA_KEY = os.getenv('OPENSEA_KEY')
null_address = '0x0000000000000000000000000000000000000000'
alchemy_url = f"https://eth-mainnet.alchemyapi.io/v2/{ALCHEMY_KEY}"
web3 = Web3(Web3.HTTPProvider(alchemy_url))


def safe_div(a, b):
    return a / b if b else 0


def is_contract(address):
    code = web3.eth.get_code(web3.toChecksumAddress(address))
    res = bool(web3.toInt(code))
    return res


def get_erc20_balance_from_address_moralis(api_key, address):
    headers = {
        'accept': 'application/json',
        'X-API-Key': api_key,
    }
    params = {
        'chain': 'eth',
    }
    url = f'https://deep-index.moralis.io/api/v2/{address}/erc20'
    response = requests.get(url, headers=headers, params=params)
    if response.status_code != 200:
        raise ValueError(response.json()['message'])
    logs.logger.info(f"ERC20 Token loaded successfully for wallet:{address}")
    json_response = response.json()
    return json_response


def get_df_erc20_from_moralis(json_response):
    res = json_response
    df_erc20 = pd.DataFrame(columns=api_tags.erc20_tags, data=res)
    df_erc20['token_address'] = df_erc20['token_address'].apply(lambda x: x.lower())
    df_cg = coingecko_api.df_cg
    df_erc20 = df_erc20.merge(df_cg, how='inner', on=['token_address'])
    df_erc20['balance'] = df_erc20['balance'].apply(lambda x: int(x)) / 10 ** 18
    res = df_erc20[['id', 'token_address', 'balance', 'current_price']]
    return res


def get_nft_token_id_from_moralis(api_key, address, df_nft):
    df_filter = df_nft[['primary_asset_contracts', 'stats.seven_day_average_price']]
    df_filter = df_filter.rename(columns={'primary_asset_contracts': 'token_address'})
    df_filter = df_filter[df_filter.apply(lambda x: len(x['token_address']) == 1, axis=1)]
    df_filter['token_address'] = df_filter['token_address'].apply(lambda x: x[0]['address'])
    headers = {
        'accept': 'application/json',
        'X-API-Key': api_key,
    }
    params = {
        'chain': 'eth',
        'format': 'decimal',
    }
    url = f'https://deep-index.moralis.io/api/v2/{address}/nft'
    response = requests.get(url, headers=headers, params=params)
    if response.status_code != 200:
        raise ValueError(response.json()['message'])
    logs.logger.info(f"ERC721 TokenID loaded successfully for wallet:{address}")
    json_response = response.json()
    res = json_response['result']
    df_holdings = pd.DataFrame(columns=api_tags.nft_id_tags, data=res)
    df_holdings['token_address'] = df_holdings['token_address'].apply(lambda x: x.lower())
    df_holdings = df_holdings.merge(df_filter, how='inner', on=['token_address'])
    df_holdings = df_holdings.rename(columns={'stats.seven_day_average_price': 'value'})
    df_holdings['block_timestamp'] = np.repeat(datetime.now(timezone.utc), len(df_holdings))
    res = df_holdings[['block_timestamp', 'token_address', 'token_id', 'value']]
    return res


def opensea_retrieving_collections_for_address(address):
    """
    opensea_retrieving_collections request "Retrieving collections" method from Opensea API
    retrieve collections onwed by user


    :param address: ethereum address
    :type address: string
    :return: nft collection owned by user with the number of asset owned for each collection
    :rtype: dict
    """
    url = 'https://api.opensea.io/api/v1/collections'
    headers = {
        'Accept': 'application/json',
        'X-API-KEY': OPENSEA_KEY
    }
    params = (
        ('asset_owner', address),
        ('offset', 0),
        ('limit', 300))
    response = requests.get(url, params=params, headers=headers)
    if response.status_code != 200:
        raise ValueError(response.json()['error'])
    logs.logger.info(f"ERC-721/1155 Token loaded successfully for wallet:{address}")
    return response.json()


# def add_collection_to_database(df_nft):
#    global nft_database
#    batch_collection = df_nft
#    batch_collection.pop('owned_asset_account')
#    to_add = batch_collection[(~batch_collection['slug'].isin(nft_database['slug']))]
#    nft_database = nft_database.append(to_add)


def get_df_nft(json_response, thresold):
    """
    get_df_nft filter nft collection from user and build a dataframe for data manipulation


    :param json_response: opensea response from 'Retrieve collection' method
    :param thresold: volume/price ratio
    :type json_response: dict
    :type thresold: integer
    :return: erc721 and erc1155 user collection for which volume/average price is greater than thresold over
    :rtype: dataframe
    """
    res = json_response
    df_nft = json_normalize(res) if len(res) else pd.DataFrame(columns=api_tags.opensea_tags)
    df_nft = df_nft[(df_nft[f"stats.seven_day_average_price"] != 0)]
    df_nft = df_nft[(df_nft[f"stats.seven_day_volume"] / df_nft[f"stats.seven_day_average_price"] > thresold)]
    df_nft = df_nft[df_nft['slug'] != 'ens']
    return df_nft


def get_nft_transfers_from_moralis(api_key, address):
    headers = {
        'accept': 'application/json',
        'X-API-Key': api_key,
    }
    params = {
        'chain': 'eth',
        'format': 'decimal',
        'direction': 'both',
        'limit': 500,
    }
    url = f'https://deep-index.moralis.io/api/v2/{address}/nft/transfers'
    response = requests.get(url, headers=headers, params=params)
    if response.status_code != 200:
        raise ValueError(response.json()['message'])
    logs.logger.info(f"NFT Transfers loaded successfully for wallet:{address}")
    json_response = response.json()
    return json_response


def get_df_nft_transfers(json_response):
    res = json_response['result']
    df_nft_tx = json_normalize(res) if len(res) else pd.DataFrame(columns=api_tags.nft_transfer_tags)
    hash_count = Counter(df_nft_tx['transaction_hash'])
    df_nft_tx['token_address'] = df_nft_tx['token_address'].apply(lambda x: x.lower())
    df_nft_tx['value'] = df_nft_tx['value'].apply(lambda x: int(x)) / 10 ** 18
    df_nft_tx['value'] = df_nft_tx.apply(lambda x: x['value']/hash_count[x['transaction_hash']], axis=1)
    df_nft_tx['block_timestamp'] = df_nft_tx['block_timestamp'].apply(lambda x: parser.parse(x))
    return df_nft_tx


def eth_balance(address):
    headers = {
        'accept': 'application/json',
        'X-API-Key': MORALIS_KEY,
    }
    params = {
        'chain': 'eth',
    }
    url = f"https://deep-index.moralis.io/api/v2/{address}/balance"
    response = requests.get(url, headers=headers, params=params)
    if response.status_code != 200:
        raise ValueError(response.json()['message'])
    res = int(response.json()['balance']) / 10 ** 18
    return res


def erc20_val(df_erc20):
    res = np.sum(df_erc20['balance'] * df_erc20['current_price'])
    return res


def erc20_count(df_erc20):
    res = len(df_erc20)
    return res


def collection_balance(df_nft):
    res = len(df_nft)
    return res


def nft_balance(df_nft):
    res = np.sum(df_nft['owned_asset_count'])
    return res


def nft_count(df_nft):
    res = np.sum(df_nft['owned_asset_count'])
    return res


def total_nft_value(df_nft, based_price):
    res = np.sum(df_nft[f"stats.{based_price}"] * df_nft['owned_asset_count'])
    return res


def average_nft_holding(df_nft):
    res = np.mean(df_nft['owned_asset_count']) if not df_nft['owned_asset_count'].empty else 0
    return res


def average_nft_price(df_nft, based_price):
    res = np.mean(df_nft[f"stats.{based_price}"]) if not df_nft.empty else 0
    return res


def std_nft_price(df_nft, based_price):
    res = np.std(df_nft[f"stats.{based_price}"])
    return res


def get_nft_mint(df_nft_tx):
    res = df_nft_tx[((df_nft_tx['from_address'] == null_address) & (df_nft_tx['value'] > 0))][
        ['block_timestamp', 'token_address', 'token_id', 'value']]
    return res


def get_nft_buys(address, df_nft_tx):
    res = df_nft_tx[((df_nft_tx['value'] > 0) & (df_nft_tx['from_address'] != null_address)
                     & (df_nft_tx['to_address'] == address.lower())
                     )][
        ['block_timestamp', 'token_address', 'token_id', 'value']]
    return res


def get_nft_sales(address, df_nft_tx):
    res = df_nft_tx[((df_nft_tx['value'] > 0) & (df_nft_tx['from_address'] == address.lower()))][
        ['block_timestamp', 'token_address', 'token_id', 'value']]
    return res


def free_transfers(df_nft_tx):
    res = df_nft_tx[((df_nft_tx['from_address'] != null_address) & (df_nft_tx['to_address'] != null_address)
                     & (df_nft_tx['value'] == 0))][
        ['token_address', 'token_id', 'value']]
    return res


def weekly_tx_fq(df_nft_tx):
    tmp = df_nft_tx.sort_values('block_timestamp')
    duration = datetime.now(timezone.utc) - tmp['block_timestamp'].iloc[0] if not tmp.empty else 0
    res = safe_div(len(tmp), duration.days/7)
    return res


def get_scores(address, df_erc20, df_nft, df_nft_holdings, df_nft_tx):
    """
    get_scores compute all scores from user data (tokens and transactions)


    :param address: user address
    :param df_erc20: erc20 user data
    :param df_nft: erc721/1155 opensea collection
    :param df_nft_holdings: erc721/1155 token address + tokenID holdings
    :param df_nft_tx: erc721 transaction transfers by user
    :type address: string
    :type df_erc20: dataframe
    :type df_nft: dataframe
    :type df_nft_tx: dataframe
    :return: dataframe
    :rtype: dataframe with user scores
    """
    score_keys = [
        'address', 'value_score', 'collection_balance', 'nft_balance',
        'nft_exposition_wealth', 'average_nft_holding', 'average_nft_price', 'std_nft_price', 'nft_volume_traded',
        'minting_ratio', 'sales_on_mint', 'sales_on_buys', 'sales_ratio', 'profit_score', 'avg_holding_time',
        'weekly_tx_fq', 'free_transfers']
    df_res = pd.DataFrame(columns=score_keys)
    df_res.loc[df_res.shape[0]] = np.repeat(None, len(score_keys))
    eth_val = eth_balance(address)
    top_erc20_val = erc20_val(df_erc20)
    nft_val = total_nft_value(df_nft, 'seven_day_average_price')
    nft_mints = get_nft_mint(df_nft_tx)
    nft_sales = get_nft_sales(address, df_nft_tx)
    nft_buys = get_nft_buys(address, df_nft_tx)
    sales_on_mints = nft_mints.merge(nft_sales, how='inner', on=['token_address', 'token_id'])
    sales_on_buys = nft_buys.merge(nft_sales, how='inner', on=['token_address', 'token_id'])
    rotation_on_mints = safe_div(len(sales_on_mints), len(nft_mints))
    rotation_on_buys = safe_div(len(sales_on_buys), len(nft_buys))
    all_nft_buys = pd.concat([nft_mints, nft_buys])
    trades = all_nft_buys.merge(nft_sales, how='inner', on=['token_address', 'token_id'])
    minting_ratio = safe_div(len(nft_mints), len(all_nft_buys))
    sales_ratio = safe_div(len(trades), len(all_nft_buys))
    holdings_on_all_buys = all_nft_buys.merge(df_nft_holdings, how='inner', on=['token_address', 'token_id'])
    trade_n_hold = pd.concat([trades, holdings_on_all_buys])
    nft_money_input = np.sum(trade_n_hold['value_x'])
    nft_money_output = np.sum(trade_n_hold['value_y'])
    holding_time = pd.to_datetime(trade_n_hold['block_timestamp_y']).dt.tz_localize(None) - pd.to_datetime(trade_n_hold['block_timestamp_x']).dt.tz_localize(None)
    df_res.loc[0]['address'] = address
    df_res.loc[0]['value_score'] = nft_val + eth_val + top_erc20_val
    df_res.loc[0]['nft_exposition_wealth'] = safe_div(nft_val, nft_val + eth_val + top_erc20_val)
    df_res.loc[0]['collection_balance'] = collection_balance(df_nft)
    df_res.loc[0]['nft_balance'] = nft_balance(df_nft)
    df_res.loc[0]['average_nft_holding'] = average_nft_holding(df_nft)
    df_res.loc[0]['average_nft_price'] = average_nft_price(df_nft, 'seven_day_average_price')
    df_res.loc[0]['std_nft_price'] = std_nft_price(df_nft, 'seven_day_average_price')
    df_res.loc[0]['nft_volume_traded'] = np.sum(df_nft_tx['value'])
    df_res.loc[0]['sales_on_mint'] = rotation_on_mints
    df_res.loc[0]['sales_on_buys'] = rotation_on_buys
    df_res.loc[0]['minting_ratio'] = minting_ratio
    df_res.loc[0]['sales_ratio'] = sales_ratio
    df_res.loc[0]['profit_score'] = safe_div(nft_money_output, nft_money_input)
    df_res.loc[0]['avg_holding_time'] = np.mean(holding_time).days if not holding_time.empty else 0
    df_res.loc[0]['weekly_tx_fq'] = weekly_tx_fq(df_nft_tx)
    df_res.loc[0]['free_transfers'] = safe_div(len(free_transfers(df_nft_tx)), len(df_nft_tx))
    logs.logger.info(f"Scores computed successfully for wallet:{address}")
    logs.logger.info(f"-------------------------------------------------")
    return df_res


class User:

    def __init__(self, address):
        if not web3.isAddress(address):
            raise ValueError('Please provide hexadecimal address')
        if is_contract(address):
            raise ValueError('Please provide owned ethereum account')
        self.address = address
        self.df_erc20 = pd.DataFrame(columns=api_tags.erc20_tags)
        self.df_nft = pd.DataFrame(columns=api_tags.opensea_tags)
        self.df_nft_tx = pd.DataFrame(columns=api_tags.nft_transfer_tags)
        self.df_scores = pd.DataFrame()
        self.df_nft_holdings = pd.DataFrame()

    def load_erc20_token(self):
        json_response_erc20 = get_erc20_balance_from_address_moralis(MORALIS_KEY, self.address)
        self.df_erc20 = get_df_erc20_from_moralis(json_response_erc20)

    def load_nft(self, thresold):
        json_response_nft = opensea_retrieving_collections_for_address(self.address)
        self.df_nft = get_df_nft(json_response_nft, thresold)

    def load_nft_transfers(self):
        json_response_transfers = get_nft_transfers_from_moralis(MORALIS_KEY, self.address)
        self.df_nft_tx = get_df_nft_transfers(json_response_transfers)

    def load_nft_holdings(self):
        self.df_nft_holdings = get_nft_token_id_from_moralis(MORALIS_KEY, self.address, self.df_nft)

    def load_scores(self):
        self.df_scores = get_scores(address=self.address, df_erc20=self.df_erc20, df_nft=self.df_nft,
                                    df_nft_tx=self.df_nft_tx, df_nft_holdings=self.df_nft_holdings)

    def load_all(self):
        self.load_erc20_token()
        self.load_nft(20)
        self.load_nft_holdings()
        self.load_nft_transfers()
        self.load_scores()

    def get_df_erc20_token(self):
        return self.df_erc20

    def get_df_nft(self):
        return self.df_nft

    def get_df_nft_transfers(self):
        return self.df_nft_tx

    def get_df_scores(self):
        return self.df_scores

    def get_scores(self):
        if self.df_scores.empty:
            self.load_all()
        res = dict(self.df_scores.iloc[0])
        return res


def get_lof_address(api_key):
    from_block = web3.eth.get_block_number()-2000
    to_block = web3.eth.get_block_number()-200
    headers = {
        'accept': 'application/json',
        'X-API-Key': api_key,
    }
    params = {
        'chain': 'eth',
        'from_block': f"{from_block}",
        'to_block': f"{to_block}",
        'format': 'decimal',
    }
    url = f'https://deep-index.moralis.io/api/v2/nft/transfers'
    response = requests.get(url, headers=headers, params=params)
    if response.status_code != 200:
        raise ValueError(response.json()['message'])
    df = json_normalize(response.json()['result'])
    lof_addresses = list(df['from_address'].values) + list(df['to_address'].values)
    res = np.unique([x for x in lof_addresses if x != null_address
                     and x != '0x000000000000000000000000000000000000dead' and not is_contract(x)])
    return res


def get_user_score(address):
    user = User(address)
    res = user.get_scores()
    return res


def get_batch_scores():
    lof_address = get_lof_address(MORALIS_KEY)
    lof_scores = [get_user_score(x) for x in lof_address]
    res = json_normalize(lof_scores)
    return res
