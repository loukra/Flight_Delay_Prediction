import re
import pandas as pd


def to_date(string, order: list = ["%Y", "%m", "%d", "%H", "%M", "%S"]):

    spacings = re.findall("\D", string)
    format = "".join([str(a) + str(b) for a, b in zip(order, spacings)])
    if len(order) != len(spacings):
        format += order[-1]

    date = pd.to_datetime(string, format=format)

    return date


def applier_cwise(dataframe, columns, func, orders = []):

    for count, col in enumerate(columns):
        if dataframe[col].dtype == object:
            if len(orders) > 1:
                order = orders[count]
            elif len(orders) == 1:
                order = orders[0]
            elif len(orders) == 0:
                order = ["%Y", "%m", "%d", "%H", "%M", "%S"]
            dataframe.loc[:, col] = dataframe[col].apply(func, order = order)
    

