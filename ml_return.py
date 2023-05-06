import pandas as pd
# import numpy as np
# from datetime import datetime
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from sklearn.decomposition import PCA
# from sklearn.svm import SVC
# from sklearn.linear_model import LogisticRegression
# from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
# from sklearn.linear_model import Perceptron
# from sklearn.metrics import accuracy_score, classification_report

# pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

def ml_return(ledger):

        # Load Data Files to Dataframe
        df1 = pd.DataFrame(ledger)
        df1 = df1.drop(columns=df1.columns[0], axis=1)  # should remove first col !!
        df2 = pd.read_csv("hw4_data_logreturn.csv", na_values=["#DIV/0!", "#NUM!"])
        df2 = df2.iloc[:, :-8]
        df1.rename(columns={"dt_enter": "Dates"}, inplace=True)
        df1['Dates'] = pd.to_datetime(df1['Dates'])
        df2['Dates'] = pd.to_datetime(df2['Dates'])
        df1['Dates'] = df1['Dates'].dt.strftime('%Y/%-m/%-d')
        df2['Dates'] = df2['Dates'].dt.strftime('%Y/%-m/%-d')
        merged_df = pd.merge(df2, df1[["Dates", "success"]], on='Dates')

        # Data Cleaning
        merged_df.fillna(0, inplace=True)
        df1.iloc[:,4:7] = df1.iloc[:,4:7].fillna(0)

        # Split Dataset
        for i in range((len(merged_df) - 21)):
                df_train = merged_df.iloc[i:21+i]
                df_test = merged_df.iloc[21+i]
                x_train = df_train.drop(["success", "Dates"], axis=1)
                y_train = df_train.success
                x_test = df_test[1:13]
                x_test = x_test.to_frame().T

                # Import RF
                rf = RandomForestClassifier(n_estimators=200, random_state=42)
                rf.fit(x_train, y_train)
                y_predict = rf.predict(x_test)

                # Output Predict
                df1['success'][i] = y_predict[0]
                if y_predict[0] == 1:
                        df1['rtn'][i] *= y_predict[0]
                else:
                        df1['rtn'][i] *= 0

                        new_ledger = df1

        return new_ledger

