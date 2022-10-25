import pandas as pd
from geopy import load_geodata
from feature_engineering import dframe_engineering
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import pickle
import sys
import numpy as np

def main(url_train):
    df = pd.read_csv(url_train)
    df = df.set_index("ID")

    X_train, y_train, X_test, y_test = dframe_engineering(df,mode = 'train')

    params = {'subsample': 1,
    'n_estimators': 1000,
    'min_child_weight': 2,
    'max_depth': 4,
    'learning_rate': 0.2,
    'gamma': 0.1,
    'colsample_bytree': 0.75}
    xgb_model = XGBRegressor(**params)

    xgb_model.fit(X_train,y_train)

    y_train_pred = xgb_model.predict(X_train)
    y_train_pred = np.where(y_train_pred < 0, 0, y_train_pred)

    rmse_xgb_train = mean_squared_error(y_train,y_train_pred, squared=False)
    r2_xgb_train = r2_score(y_train,y_train_pred)

    y_test_pred = xgb_model.predict(X_test)
    y_test_pred = np.where(y_test_pred < 0, 0, y_test_pred)
    rmse_xgb_test = mean_squared_error(y_test,y_test_pred, squared=False)
    r2_xgb_test = r2_score(y_test,y_test_pred)

    print('XGB Train Scores:')
    print(f'RMSE: {rmse_xgb_train}')
    print(f'r2: {r2_xgb_train}')

    print('XGB Test Scores:')
    print(f'RMSE: {rmse_xgb_test}')
    print(f'r2: {r2_xgb_test}')


    print("Saving model in the model folder")
    filename = 'models/xgb_model.sav'
    pickle.dump(xgb_model, open(filename, 'wb'))

    filename = 'Data/Train_dataframe.csv'
    train_dframe = pd.concat([X_train, pd.Series(y_train, name ='target')], axis = 1)
    test_dframe = pd.concat([X_test, pd.Series(y_test, name ='target')], axis = 1)

    train_dframe.to_csv(filename),
    filename = 'Data/Test_dataframe.csv'
    test_dframe.to_csv(filename)


if __name__ == '__main__':
    print('Number of arguments:', len(sys.argv), 'arguments.')
    print('Argument List:', str(sys.argv)) 

    #in an ideal world this would validated
    url_train = sys.argv[1]

    main(url_train)



