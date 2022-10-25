import pandas as pd
from geopy import load_geodata
from date_pyler import applier_cwise
from date_pyler import to_date
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pickle


def datetime_engineering(df_geo):
    print("Datetime engineering...")
    # columnwise datetype conversion on dataframe
    date_order = [
        ["%Y", "%m", "%d", "%H", "%M", "%S"],
        ["%Y", "%m", "%d", "%H", "%M", "%S"],
        ["%Y", "%m", "%d"],
    ]
    applier_cwise(df_geo, ["STD", "STA", "DATOP"], to_date, orders=date_order)

    df_geo["date_dep_year"] = df_geo.DATOP.dt.year.astype(int)
    df_geo["date_dep_month"] = df_geo.DATOP.dt.month.astype(int)
    df_geo["date_dep_day_year"] = df_geo.DATOP.dt.day_of_year.astype(int)
    df_geo["date_dep_day_week"] = df_geo.DATOP.dt.day_of_week.astype(int)
    df_geo["date_dep_hour"] = df_geo.STD.dt.hour.astype(int)
    df_geo["date_dep_minute"] = df_geo.STD.dt.minute.astype(int)
    df_geo["flight_duration"] = (
        (df_geo.STA - df_geo.STD).astype("timedelta64[m]").astype(int)
    )
    df_geo["min_of_day"] = (df_geo.STD.dt.hour * 60 + df_geo.STD.dt.minute).astype(int)
    return df_geo


def OneHot(
    X_test, X_train=None, column="", drop="first", handle_unknown="ignore", sparse=False
):
    print("One Hot Encoding...")
    if isinstance(X_train, pd.DataFrame):
        print("we are here!")
        onehot = OneHotEncoder(drop="first", handle_unknown="ignore", sparse=False)
        STATUS_onehot_train = onehot.fit_transform(
            X_train[column].to_numpy().reshape(-1, 1)
        )
        STATUS_onehot_df_train = pd.DataFrame(
            STATUS_onehot_train,
            columns=[column + "_" + x for x in onehot.categories_[0][1:]],
        )
        filename = ["models/OneHot_" + column + "_fitted.sav"][0]
        pickle.dump(onehot, open(filename, "wb"))

        STATUS_onehot_test = onehot.transform(X_test[column].to_numpy().reshape(-1, 1))
        STATUS_onehot_df_test = pd.DataFrame(
            STATUS_onehot_test,
            columns=[column + "_" + x for x in onehot.categories_[0][1:]],
        )

        X_train = pd.concat(
            [X_train, STATUS_onehot_df_train.set_index(X_train.index)], axis=1
        )
        X_test = pd.concat(
            [X_test, STATUS_onehot_df_test.set_index(X_test.index)], axis=1
        )

        X_train.drop(column, axis=1, inplace=True)
        X_test.drop(column, axis=1, inplace=True)
        return X_test, X_train
    else:
        
        filename = ["models/OneHot_" + column + "_fitted.sav"][0]
        with open(filename, "rb") as f:
            onehot = pickle.load(f)
        STATUS_onehot_test = onehot.transform(X_test[column].to_numpy().reshape(-1, 1))
        STATUS_onehot_df_test = pd.DataFrame(
            STATUS_onehot_test,
            columns=[column + "_" + x for x in onehot.categories_[0][1:]],
        )

        X_test = pd.concat(
            [X_test, STATUS_onehot_df_test.set_index(X_test.index)], axis=1
        )

        X_test.drop(column, axis=1, inplace=True)
        return X_test


def dframe_engineering(df, mode):
    print("Feature engineering...")
    df_geo = load_geodata(df)
    df_geo = datetime_engineering(df_geo)
    df_geo["exec_airline"] = df_geo.FLTID.str.split().str[0]
    df_geo = df_geo.dropna()

    features = [
        "date_dep_year",
        "date_dep_month",
        "date_dep_day_year",
        "date_dep_day_week",
        "date_dep_hour",
        "date_dep_minute",
        "flight_duration",
        "lat_DEPSTN",
        "lon_DEPSTN",
        "elevation_DEPSTN",
        "lat_ARRSTN",
        "lon_ARRSTN",
        "elevation_ARRSTN",
        "exec_airline",
        "STATUS",
    ]

    X = df_geo[features]
    if "target" in df_geo.columns:
        y = df_geo.target
    if mode == "train":
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=42
        )

        X_test, X_train = OneHot(X_test=X_test, X_train=X_train, column="STATUS")
        X_test, X_train = OneHot(X_test=X_test, X_train=X_train, column="exec_airline")

        scaler = MinMaxScaler()

        X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
        X_test = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

        filename = "models/MinMax_fitted.sav"
        pickle.dump(scaler, open(filename, "wb"))

        y_train = np.where(y_train > 1440 / 2, 1440 / 2, y_train)
        X_train = X_train.drop(columns="date_dep_month")
        X_test = X_test.drop(columns="date_dep_month")

        return X_train, y_train, X_test, y_test
    elif mode == "predict":
        X_test = OneHot(X_test=X, column="STATUS")
        X_test = OneHot(X_test=X_test, column="exec_airline")
        filename = "models/MinMax_fitted.sav"
        with open(filename, "rb") as f:
            scaler = pickle.load(f)
        X_test = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)
        X_test = X_test.drop(columns="date_dep_month")
        return X_test, y
