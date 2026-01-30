import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import joblib

root_file = r'data\samples'
# years = [2020, 2021, 2022, 2023]
years = [2022]

from_vals = [
    91,92,93,98,99,
    101,102,103,104,105,106,107,108,109,
    121,122,123,124,
    141,142,143,144,145,146,147,148,149,
    161,162,163,164,165,
    181,182,183,184,185,186,187,188,189,
    201,202,203,204
]

to_vals = [
    1,1,1,1,1,
    2,2,2,2,2,2,2,2,2,
    3,3,3,3,
    4,4,4,4,4,4,4,4,4,
    5,5,5,5,5,
    6,6,6,6,6,6,6,6,6,
    7,7,7,7
]

# parent_classes = np.unique(np.array(to_vals))
parent_classes = np.unique(np.array(from_vals))
seed = 1917

def load_data(pyromes, years):
    yearly_frames = []
    for pyrome in pyromes:
        for year in years:
            year_csv_file = root_file + f'/stratified_sample_fbfm40_30m_{pyrome}_{year}.csv'
            year_fuels_sample = pd.read_csv(year_csv_file)
            yearly_frames.append(year_fuels_sample)
    fuels_frame = pd.concat(yearly_frames)
    return fuels_frame

# Load a sample to get feature list
sample_pyrome = 29
sample_data = pd.read_csv(root_file + f'/stratified_sample_fbfm40_30m_{sample_pyrome}_{years[0]}.csv')
feature_list = sample_data.columns.to_list()
feature_list.remove('system:index')
feature_list.remove('.geo')

alphaearth_features = [f'A{str(i).zfill(2)}' for i in range(64)]
label_list = ['FBFM40', 'FBFM40Parent']
feature_list_wo_alphaearth = [feature for feature in feature_list if feature not in (alphaearth_features + label_list)]
train_features = alphaearth_features + feature_list_wo_alphaearth
# Exclude ESP and other unwanted features from training
train_features = [f for f in train_features if f not in ['ESP']]

zones_to_train = [27, 28]
# , 29, 30, 31, 32, 33, 34, 35, 41]

rf = RandomForestClassifier(
    n_estimators=500,
    criterion='entropy',
    max_features='sqrt',
    n_jobs=-1
)

for zone in zones_to_train:
    print(f'Training model for zone {zone}')
    zone_sample = load_data([zone], years)
    
    X = np.nan_to_num(zone_sample[train_features].to_numpy(), 0)
    y = zone_sample['FBFM40'].to_numpy().ravel()
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=seed)
    
    scaler = StandardScaler()
    encoder = LabelEncoder().fit(parent_classes)
    
    X_train_scaled = scaler.fit_transform(X_train)
    y_train_encode = encoder.transform(y_train)
    
    rf.fit(X_train_scaled, y_train_encode)
    rf.feature_names_in_ = np.array(train_features)

    joblib.dump(rf, f'data/rf_zone_{zone}.joblib')
    print(f'Saved data/rf_zone_{zone}.joblib')
