import re
import pandas as pd


def to_date(string, order: list = ["%Y", "%m", "%d", "%H", "%M", "%S"]):

    spacings = re.findall("\D", string)

    format = "".join([str(a) + str(b) for a, b in zip(order, spacings)])
    format += order[-1]

    date = pd.to_datetime(string, format=format)

    return date


def applier_cwise(dataframe, columns, func):
    for col in columns:
        if dataframe[col].dtype == object:
            dataframe.loc[:, col] = dataframe[col].apply(func)
    

