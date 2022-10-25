import pandas as pd
from geopy import load_geodata
from feature_engineering import dframe_engineering
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import pickle
import sys
import numpy as np

def main(url_test):
    df = pd.read_csv(url_test)
    if 'ID' in df.columns:
        df = df.set_index("ID")
    y_test = None
    
    X_test, y_test = dframe_engineering(df,mode = 'predict')

    filename = 'models/xgb_model.sav'
    with open(filename, 'rb') as f:
        xgb_model = pickle.load(f)

    y_pred = xgb_model.predict(X_test)
    y_pred = np.where(y_pred < 0, 0, y_pred)
    
    if isinstance(y_test, pd.Series):
        rmse_xgb_test = mean_squared_error(y_test,y_pred, squared=False)
        r2_xgb_test = r2_score(y_test,y_pred)

        print('XGB Test Scores:')
        print(f'RMSE: {rmse_xgb_test}')
        print(f'r2: {r2_xgb_test}')

    filename = 'predictions/y_pred.csv'
    pd.Series(y_pred).to_csv(filename)



if __name__ == '__main__':
    print('Number of arguments:', len(sys.argv), 'arguments.')
    print('Argument List:', str(sys.argv)) 

    url_ytest = None
    #in an ideal world this would validated
    url_test = sys.argv[1]


    main(url_test)