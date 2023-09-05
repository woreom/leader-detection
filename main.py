# Imports
from utils import get_data_from_mt5, preprocess_data
from cyclicity_analysis import COOM, OrientedArea
import pandas as pd
from visualazation import plot_leadership


def analyze_leadership(initialize, currencies, TimeFrame, n_sample, plot_results=True):
    """
    Analyzes the leadership relationships between currencies.

    Args:
        initialize(list): list of login, password, server for the MT5 account.
        currencies (list): A list of currency pairs to analyze.
        TimeFrame (str): The time frame of the data.
        n_sample (int): The number of samples to retrieve.
        plot_results(bool): Whether to plot results or not 

    Returns:
       results (dict): A dictionary containing the results of the analysis.
        - 'lead_lag_matrix' (pd.DataFrame)
        - 'sequential_order_dict' (dict)
        - 'accumulated_oriented_area' (pd.DataFrame)
    """
    # Get Data
    df = pd.DataFrame()
    for currency in currencies: 
        df[f'{currency}'] = get_data_from_mt5(initialize, currency, TimeFrame)['Mean'].iloc[-n_sample:]

    # Preprocess data
    X = preprocess_data(df)

    # Determining Pairwise Component Leader Follower Relationships
    oa = OrientedArea(X)
    lead_lag_df = oa.compute_lead_lag_df()

    # Determining the Sequential Order of Time-Series
    coom = COOM(lead_lag_df)
    #eigenvalue_moduli = coom.eigenvalue_moduli
    leading_eigenvector = coom.get_leading_eigenvector()
    sequential_order_dict = coom.compute_sequential_order_dict(leading_eigenvector) 
    print("Sequential Order of data, (Note that index 0 is leader) :")
    print(sequential_order_dict)

    # Determining The accumulated oriented area
    accumulated_oriented_area = pd.DataFrame()
    for i in range(len(sequential_order_dict)-1):
        temp = oa.compute_pairwise_accumulated_oriented_area_df(sequential_order_dict[i], sequential_order_dict[i+1])
        accumulated_oriented_area[str(list(temp.columns)[0])] = temp.to_numpy().reshape(-1,)

    # Plot Results
    if plot_results: plot_leadership(X, lead_lag_df, leading_eigenvector, sequential_order_dict, accumulated_oriented_area)
    
    results=dict()
    results['lead_lag_matrix']=lead_lag_df
    results['seq order']=sequential_order_dict
    results['sequential_order_dict']=accumulated_oriented_area
    
    return results



if __name__ == "__main__":

    # Configuration
    LOGIN = "51545562"
    PASSWORD = "zop7gsit"
    SERVER = "Alpari-MT5-Demo"
    
    initialize = [LOGIN, PASSWORD, SERVER]
    
    # Get Data
    currencies = ['XAUUSD', 'XAGUSD', 'USDCHF']
    TimeFrame='1h'
    n_sample=10000
    plot_results=True
    
    results=analyze_leadership(initialize, currencies, TimeFrame, n_sample, plot_results)





