import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score,precision_score,classification_report,confusion_matrix
from sklearn.model_selection import GridSearchCV,train_test_split
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
import xgboost as xgb

root_file = r'data\samples'
dist_file = r'data\samples\fuel_class_distributions_2023.csv'
dist_frame = pd.read_csv(dist_file, index_col=0)

sample_pyromes= [
    29
]
eval_pyromes = [
    28
]

single_pyrome_tests = [
    # 6,
    # 18,
    # 26,
    27,
    28,
    29,
    30,
    31,
    32,
    33,
    34,
    35,
    # 36,
    41
]

selected_pair_neighbor_zones = [
    # 18,
    # 26,
    27,
    28,
    # 26,
    30,
    31,
    34,
    31,
    34,
    35,
    # 36,
    35,
    34
]

year = 2023
years = [
    2020,
    2021,
    2022,
    2023
]

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
seed = 1917

child_classes = np.unique(np.array(from_vals))
parent_classes = np.unique(np.array(to_vals))

def load_data(pyromes,years):
    yearly_frames = []
    for pyrome in pyromes:
        for year in years:
            year_csv_file = root_file + f'/stratified_sample_fbfm40_30m_{pyrome}_{year}.csv'
            year_fuels_sample = pd.read_csv(year_csv_file)
            yearly_frames.append(year_fuels_sample)

    fuels_frame = pd.concat(yearly_frames)

    return fuels_frame

fuels_sample = load_data(sample_pyromes,years)

feature_list = fuels_sample.columns.to_list()
feature_list.remove('system:index')
feature_list.remove('.geo')

alphaearth_features = [f'A{str(i).zfill(2)}' for i in range(64)]
label_list = ['FBFM40','FBFM40Parent']
feature_list_wo_alphaearth = [feature for feature in feature_list if feature not in (alphaearth_features +label_list)]

# def find_closest_zone(zone,all_zones,zone_dists):
#     dists = {}
#     for pair_zone in [pair for pair in all_zones if pair != zone]:
#         dist = np.sqrt(np.sum((zone_dists[zone_dists['zone'] == zone][['1','2','3','4','5','6','7']].to_numpy() - zone_dists[zone_dists['zone'] == pair_zone][['1','2','3','4','5','6','7']].to_numpy())**2))
#         dists[pair_zone] = dist
#     neighbor = min(dists,key=dists.get)
#     return neighbor

def find_closest_zone(zone, all_zones, zone_dists):
    dists = {}
    for pair_zone in [pair for pair in all_zones if pair != zone]:
        dist = np.sqrt(np.sum((zone_dists.loc[zone][['1','2','3','4','5','6','7']].to_numpy() - zone_dists.loc[pair_zone][['1','2','3','4','5','6','7']].to_numpy())**2))
        dists[pair_zone] = dist
    neighbor = min(dists, key=dists.get)
    return neighbor

def train_and_eval_pyrome(zone,selected_pair_zone,all_zones,zone_class_dists,years,model,seed,features,labels,class_list,test_size=0.25):
    zone_sample = load_data([zone],years)

    pair_zone_sample = load_data([selected_pair_zone],years)

    
    closest_distribution_zone = find_closest_zone(zone,all_zones,zone_class_dists)
    print(f'Closest Zone to {zone} was {closest_distribution_zone}')
    closest_distribution_zone_sample = load_data([closest_distribution_zone],years)

    X = np.nan_to_num(zone_sample[features].to_numpy(),0)
    y = zone_sample[labels].to_numpy().ravel()

    X_pair_zone = np.nan_to_num(pair_zone_sample[features].to_numpy(),0)
    y_pair_zone = pair_zone_sample[labels].to_numpy().ravel()

    X_closest_zone = np.nan_to_num(closest_distribution_zone_sample[features].to_numpy(),0)
    y_closest_zone = closest_distribution_zone_sample[labels].to_numpy().ravel()

    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=test_size,random_state=seed)

    scaler = StandardScaler()
    encoder = LabelEncoder().fit(class_list)

    X_train_scaled = scaler.fit_transform(X_train)
    y_train_encode = encoder.transform(y_train)

    X_test_scaled = scaler.transform(X_test)
    y_test_encode = encoder.transform(y_test)

    X_pair_zone_scaled = scaler.transform(X_pair_zone)
    y_pair_zone_encode = encoder.transform(y_pair_zone)

    X_closest_zone_scaled = scaler.transform(X_closest_zone)
    y_closest_zone_encode = encoder.transform(y_closest_zone)


    model.fit(X_train_scaled,y_train_encode)

    in_zone_pred = model.predict(X_test_scaled)
    pair_zone_pred = model.predict(X_pair_zone_scaled)
    closest_zone_pred = model.predict(X_closest_zone_scaled)


    in_zone_accuracy = accuracy_score(y_test_encode,in_zone_pred)
    pair_zone_accuracy = accuracy_score(y_pair_zone_encode,pair_zone_pred)
    closest_zone_accuracy= accuracy_score(y_closest_zone_encode,closest_zone_pred)

    return in_zone_accuracy, pair_zone_accuracy, closest_zone_accuracy

# xgbclf = xgb.XGBClassifier()

rf = RandomForestClassifier(
    n_estimators=500,
    criterion='entropy',
    max_features='sqrt',
    n_jobs=-1
)
knn = KNeighborsClassifier(
    n_neighbors=11,
    metric='cosine',
    weights='distance',
    n_jobs=-1
)

performance_metrics = []
# train_features = alphaearth_features + feature_list_wo_alphaearth
train_features = feature_list_wo_alphaearth

for model_name, model in zip(['RandomForest'],[rf]):
    for zone, pair_zone in zip(single_pyrome_tests, selected_pair_neighbor_zones):
        print(f'Evaluating {model_name} in Zone {zone} with pair zone {pair_zone}')
        in_zone_accuracy, pair_zone_accuracy, closest_zone_accuracy = train_and_eval_pyrome(
                                                                                            zone,
                                                                                            pair_zone,
                                                                                            single_pyrome_tests,
                                                                                            dist_frame,
                                                                                            years,
                                                                                            model,
                                                                                            seed,
                                                                                            train_features,
                                                                                            ['FBFM40Parent'],
                                                                                            parent_classes,
                                                                                            test_size=0.25
                                                                                        )
        performance_metrics_sample = {
                                'zone':zone,
                                'model':model_name,
                                'in_zone':in_zone_accuracy,
                                'pair_zone':pair_zone_accuracy,
                                'closest_zone':closest_zone_accuracy
                            }
        performance_metrics.append(performance_metrics_sample)

performance_df = pd.DataFrame(performance_metrics)
performance_df.to_csv(r'data/performance_metrics_wo_alphaearth.csv',index=False)