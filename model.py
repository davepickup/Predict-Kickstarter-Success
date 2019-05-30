import json
import numpy as np
import pandas as pd
from pandas.io.json import json_normalize
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier


def normalize_json_column(df, column):
    """Extracts json data from specified column and concatanates with original df"""
    column_normalized_df = json_normalize(df[column].apply(json.loads))
    column_normalized_df.columns = list(map(lambda x: "{}.{}".format(column,x),column_normalized_df.columns))
    result = pd.concat([df, column_normalized_df], axis=1, sort=False)
    del result[column]
    return result

xgb_params = {'nthread': 4,
              'objective':'binary:logistic',
              'learning_rate': 0.05,
              'max_depth': 6,
              'min_child_weight': 11,
              'subsample': 0.8,
              'colsample_bytree': 0.8,
              'n_estimators': 1000}

class KickstarterModel:

    def __init__(self):

        self.model = None
        self.scaler = None

    def preprocess_training_data(self, df):
        """Target is 'df.state'. Retrieve relevant json data from category and profile columns
        and join with other df columns. Create new features and select final features. Scale data"""

        # select most important features for preprocessing
        df_red = df[["goal", "name", "blurb", "deadline", "launched_at", "location", "static_usd_rate", "profile", "category"]]
        
        # select json cols, fill NaNs and extract data
        json_cols = ["location", "profile", "category"]
        df_red[json_cols] = df_red[json_cols].fillna(df_red[json_cols].mode().iloc[0])
        normalized = normalize_json_column(df_red, 'category')
        normalized = normalize_json_column(normalized, 'profile')
        normalized = normalize_json_column(normalized, 'location')
        
        # select columns from new normalised dataframe
        df2 = normalized[["goal", "name", "blurb", "deadline", "launched_at", "static_usd_rate", "category.name", "category.id",
                          "category.parent_id", "profile.link_url", "location.country", "location.state"]]
        
        # fill entries in profile.link_url column with 1 and 0
        df2["profile.link_url"] = df2["profile.link_url"].notnull().astype('int')
        
        # feature engineering
        df2["name_count"] = df2["name"].str.len()
        df2["blurb_count"] = df2["blurb"].str.len()
        df2["goal_std"] = df2["goal"] * df2["static_usd_rate"]
        df2["proj_duration"] = df2["deadline"] - df2["launched_at"]
        df2["launched_at"] = pd.to_datetime(df2["launched_at"], unit='s')
        df2["la_month"] = df2["launched_at"].dt.month
        df2["la_year"] = df2["launched_at"].dt.year
        
        # drop unwanted features after feature engineering
        df2.drop(["goal", "name", "blurb", "deadline", "launched_at", "static_usd_rate"], axis=1, inplace=True)
        
        # fill nan values in categorical and numerical columns
        cat_cols = df2.select_dtypes("object").columns
        num_cols = df2.select_dtypes(np.number).columns
        df2[cat_cols] = df2[cat_cols].fillna(df2[cat_cols].mode().iloc[0])
        df2[num_cols] = df2[num_cols].fillna(0)
        
        # encode categorical variables
        df2 = pd.concat([df2.select_dtypes([], ['object']),
                         df2.select_dtypes(['object']).apply(pd.Series.astype, dtype='category')
                         ], axis=1).reindex(df2.columns, axis=1)
        df2[cat_cols] = df2[cat_cols].apply(lambda x: x.cat.codes)

        # assign features to X and labels to y, scale X
        X = df2
        y = df[["state"]]
        y = y['state'].replace(['failed', 'successful'], [0, 1])
        self.scaler = StandardScaler()
        X = self.scaler.fit_transform(X)
        return X, y

    def fit(self, X, y):

        xgb_clf = XGBClassifier(**xgb_params)
        xgb_clf.fit(X, y)

        self.model = xgb_clf

    def preprocess_unseen_data(self, df):

        # select most important features for preprocessing
        df_red = df[["goal", "name", "blurb", "deadline", "launched_at", "location", "static_usd_rate", "profile", "category"]]
        
        # select json cols, fill NaNs and extract data
        json_cols = ["location", "profile", "category"]
        df_red[json_cols] = df_red[json_cols].fillna(df_red[json_cols].mode().iloc[0])
        normalized = normalize_json_column(df_red, 'category')
        normalized = normalize_json_column(normalized, 'profile')
        normalized = normalize_json_column(normalized, 'location')
        
        # select columns from new normalised dataframe
        df2 = normalized[["goal", "name", "blurb", "deadline", "launched_at", "static_usd_rate", "category.name", "category.id",
                          "category.parent_id", "profile.link_url", "location.country", "location.state"]]
        
        # fill entries in profile.link_url column with 1 and 0
        df2["profile.link_url"] = df2["profile.link_url"].notnull().astype('int')
        
        # feature engineering
        df2["name_count"] = df2["name"].str.len()
        df2["blurb_count"] = df2["blurb"].str.len()
        df2["goal_std"] = df2["goal"] * df2["static_usd_rate"]
        df2["proj_duration"] = df2["deadline"] - df2["launched_at"]
        df2["launched_at"] = pd.to_datetime(df2["launched_at"], unit='s')
        df2["la_month"] = df2["launched_at"].dt.month
        df2["la_year"] = df2["launched_at"].dt.year
        
        # drop unwanted features after feature engineering
        df2.drop(["goal", "name", "blurb", "deadline", "launched_at", "static_usd_rate"], axis=1, inplace=True)
        
        # fill nan values in categorical and numerical columns
        cat_cols = df2.select_dtypes("object").columns
        num_cols = df2.select_dtypes(np.number).columns
        df2[cat_cols] = df2[cat_cols].fillna(df2[cat_cols].mode().iloc[0])
        df2[num_cols] = df2[num_cols].fillna(0)
        
        # encode categorical variables
        df2 = pd.concat([df2.select_dtypes([], ['object']),
                         df2.select_dtypes(['object']).apply(pd.Series.astype, dtype='category')
                         ], axis=1).reindex(df2.columns, axis=1)
        df2[cat_cols] = df2[cat_cols].apply(lambda x: x.cat.codes)

        X = df2
        X_test = self.scaler.transform(X)
        return X_test

    def predict(self, X):

        return self.model.predict(X)
