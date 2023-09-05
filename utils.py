## Import Libraries
import numpy as np
import pandas as pd
import MetaTrader5 as mt5
import scipy  as sc

## Download historical market data from MetaTrader5

def get_data_from_mt5(initialize, Ticker, TimeFrame):
    """
    Download historical market data from MetaTrader5.
    
    Parameters:
        initialize: A list containing the login credentials and server information for the MetaTrader5 account.
        The list should be in the format [login, password, server].
        Ticker: A string representing the currency ticker to download.
        TimeFrame: A string representing the time frame of the data to download. 
        Valid values are "1m", "5m", "15m", "30m", "1h", "4h", "1d", "1w".
    
    Returns:
        A pandas DataFrame containing the historical market data.
    
    Examples:
        # Download historical market data from MetaTrader5
        initialize = [123456, 'password', 'MetaTraderServer']
        Ticker = 'EURUSD'
        TimeFrame = '1h'
        data = get_data_from_mt5(initialize, Ticker, TimeFrame)
    """

    # Initialization
    mt5.initialize()
    mt5.login(login=initialize[0],password=initialize[1],server=initialize[2])
    
    
    # Time Frames Definition
    TimeFrames={
                "1m":mt5.TIMEFRAME_M1,
                "5m":mt5.TIMEFRAME_M5,
                "15m":mt5.TIMEFRAME_M15,
                "30m":mt5.TIMEFRAME_M30,
                "1h":mt5.TIMEFRAME_H1,
                "4h":mt5.TIMEFRAME_H4,
                "1d":mt5.TIMEFRAME_D1,
                "1w":mt5.TIMEFRAME_W1,
                }
                

    # Get Data and do Some Proccess
    df = pd.DataFrame(mt5.copy_rates_from_pos(Ticker,  TimeFrames[TimeFrame], 0, 99999))

    df['time']=pd.to_datetime(df['time'], unit='s')
    df.set_index(df['time'],inplace=True)
    df.index = df.index.tz_localize(None)
    del df['time'],df['spread'],df['real_volume']
    df.columns=['Open', 'High', 'Low', 'Close', 'Volume']
    df['Mean']=np.mean(pd.concat((df['Low'],df['High'], df['Close']),axis=1),axis=1)
    df['diff']=df['Mean']-df['Mean'].shift(1)
    df['info']=f'{Ticker}_{TimeFrame}'

    return df.dropna()


def get_dxy_from_mt5(initialize, TimeFrame):

    currency = ['EURUSD', 'USDJPY','GBPUSD', 'USDCAD', 'USDSEK', 'USDCHF']
    dfs = []
    
    for Ticker in currency:
        df = get_data_from_mt5(initialize, Ticker, TimeFrame)
        df=df[['Open', 'High', 'Low', 'Close', 'Volume']]
        dfs.append(df)

    # Calculate the DXY index
    DXY = 50.14348112 * dfs[0]**(-0.576) *dfs[1]**(0.136) *dfs[2]**(-0.119) * dfs[3]**(0.091) *dfs[4]**(0.042) *dfs[5]**(0.036)

    DXY.dropna(inplace=True)

    DXY['Mean']=np.mean(pd.concat((DXY['Low'],DXY['High'], DXY['Close']),axis=1),axis=1)
    DXY['diff']=DXY['Mean']-DXY['Mean'].shift(1)
    DXY['info']=f'USD_{TimeFrame}'
    return DXY.dropna()


def get_country_index_from_mt5(initialize, country, TimeFrame):
    
    dxy = get_dxy_from_mt5(initialize, TimeFrame)
    if country == 'USD':
        country_index =dxy

    else:
        if country in ['CAD', 'JPY', 'SEK', 'CHF']:
            Ticker = 'USD' + country
            df = get_data_from_mt5(initialize, Ticker, TimeFrame)
            df=df[['Open', 'High', 'Low', 'Close', 'Volume']]
            dxy=dxy[['Open', 'High', 'Low', 'Close', 'Volume']]
            country_index = dxy/df
        else:
            Ticker = country + 'USD'
            df = get_data_from_mt5(initialize, Ticker, TimeFrame)
            df=df[['Open', 'High', 'Low', 'Close', 'Volume']]
            dxy=dxy[['Open', 'High', 'Low', 'Close', 'Volume']]
            country_index = df*dxy
            
        country_index['Mean']=np.mean(pd.concat((country_index['Low'],country_index['High'], country_index['Close']),axis=1),axis=1)
        country_index['diff']=country_index['Mean']-country_index['Mean'].shift(1)
        country_index['info']=f'{country}_{TimeFrame}'
    return country_index.dropna()


def preprocess_data(df):
    """
    Preprocesses the input dataframe by dropping missing values, detrending, and normalizing.

    Args:
        df (pd.DataFrame): The input dataframe.

    Returns:
        pd.DataFrame: The preprocessed dataframe.
    """
    
    df2=df.dropna()
    df_preprocessed=pd.DataFrame(index=df2.index)
    for col in df.columns:
        df_preprocessed[col]=sc.signal.detrend(np.log(df2[col].values))
        norm=np.linalg.norm(df_preprocessed[col].fillna(0),2)
        df_preprocessed[col] = df_preprocessed[col]/norm
    
    return df_preprocessed








