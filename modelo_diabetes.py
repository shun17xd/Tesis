
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score
import joblib

df = pd.read_csv(r"diabetes_012_health_indicators_BRFSS2015_VR V2.csv")
df = df.dropna()
df['target'] = df['Diabetes_012_C']
df = df[['BMI', 'Age', 'HighBP', 'HighChol', 'GenHlth', 'PhysHlth',
         'MentHlth', 'DiffWalk', 'Income', 'Education', 'target']]

class FeatureEngineering(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.epsilon = 1e-5
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        data = X.copy()
        data['HealthRatio'] = (data['GenHlth'] + self.epsilon) / (data['Age'] + self.epsilon)
        data['PhysicalMental'] = (data['PhysHlth'] + data['MentHlth']) / 2
        data['BMIAge'] = data['BMI'] * data['Age']
        return data

class WoEEncoding(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.feature_bins = {
            'BMI': [-np.inf, 25, 30, 35, np.inf],
            'Age': [-np.inf, 30, 45, 60, np.inf],
            'GenHlth': [-np.inf, 2, 3, 4, np.inf]
        }
        self.woe_mappings = {}
    def fit(self, X, y):
        X = X.copy()
        X['target'] = y.values
        for feature, bins in self.feature_bins.items():
            X[f'{feature}_cat'] = pd.cut(X[feature], bins=bins)
            grouped = X.groupby(f'{feature}_cat')['target']
            woe_df = grouped.agg(events=lambda x: (x == 1).sum(),
                                 non_events=lambda x: (x == 0).sum()).reset_index()
            woe_df['event_rate'] = woe_df['events'] / woe_df['events'].sum()
            woe_df['non_event_rate'] = woe_df['non_events'] / woe_df['non_events'].sum()
            woe_df['WOE'] = np.log((woe_df['event_rate'] + 1e-5) / (woe_df['non_event_rate'] + 1e-5))
            self.woe_mappings[feature] = dict(zip(woe_df[f'{feature}_cat'], woe_df['WOE']))
        return self
    def transform(self, X):
        data = X.copy()
        for feature in self.feature_bins.keys():
            data[f'{feature}_cat'] = pd.cut(data[feature], bins=self.feature_bins[feature])
            data[f'{feature}_woe'] = data[f'{feature}_cat'].map(self.woe_mappings[feature])
            data.drop(columns=[f'{feature}_cat'], inplace=True)
        return data

class ColumnSelector(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[self.columns]

X = df.drop(columns=['target'])
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

selected_columns = [
    'BMI', 'Age', 'HighBP', 'GenHlth', 'PhysHlth', 'BMIAge',
    'HealthRatio', 'PhysicalMental', 'BMI_woe', 'Age_woe', 'GenHlth_woe'
]

pipeline = Pipeline([
    ('feature_engineering', FeatureEngineering()),
    ('woe_encoding', WoEEncoding()),
    ('column_selector', ColumnSelector(selected_columns)),
    ('random_forest', RandomForestClassifier(max_depth=6, n_estimators=300, criterion='entropy', random_state=42))
])

pipeline.fit(X_train, y_train)
joblib.dump(pipeline, "modelo_diabetes.pkl")
print("Modelo guardado como modelo_diabetes.pkl")
