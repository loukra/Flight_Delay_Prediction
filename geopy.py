import pandas as pd
import airportsdata

def merge_geodata(dataframe, geodata, c_name):
    """Function for merging geodata into dataframe

    Args:
        dataframe (dataframe): target dataframe
        geodata (dataframe): geodata for each airport
        c_name (string): name of column. DEPSTN or ARRSTN 

    Returns:
        returns Pandas dataframe
    """
    df = pd.merge(dataframe, geodata.add_suffix('_' + c_name), left_on=c_name, right_on="iata_" + c_name, how="left")
    return df


def get_geodata(dep, arr):
    """AI is creating summary for get_geodata

    Args:
        dep ([dataframe column]): iata codes for departure airports
        arr ([datafrane column]): iata codes for arrival airports

    Returns:
        [dataframe]: dataframe of geoinformations for each airport
    """
    dep_iata = set(dep.unique())
    arr_iata = set(arr.unique())

    iata = dep_iata | arr_iata

    airports = airportsdata.load("IATA")

    geo_data = [
        (
            airports.get(code, {}).get("iata"),
            airports.get(code, {}).get("country"),
            airports.get(code, {}).get("lat"),
            airports.get(code, {}).get("lon"),
            airports.get(code, {}).get("elevation"),
        )
        for code in iata
    ]

    return pd.DataFrame(
        geo_data, columns=["iata", "country", "lat", "lon", "elevation"]
    )