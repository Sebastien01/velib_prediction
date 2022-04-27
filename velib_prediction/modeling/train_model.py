import argparse

from joblib import dump
from load import build_dataset
from pipeline import build_features_pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor


parser = argparse.ArgumentParser()
parser.add_argument('station', type=int, help='Station ID to build a model for')
parser.add_argument('target', type=str, help="Target to predict :'docks_available', 'mechanical_available' or 'ebike_available'")
parser.add_argument('model_name', type=str, help="Name of the model to save")

args = parser.parse_args()

STATION_ID = args.station
TARGET = args.target
MODEL_NAME = args.model_name

X,y = build_dataset(station_id=STATION_ID, target=TARGET)
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.75, random_state=1)

time_ft_pipe = build_features_pipeline()

X_train = time_ft_pipe.transform(X_train)
X_test  = time_ft_pipe.transform(X_test)


print("Gridsearching for best params, this might take a while ...")

params = {
 'max_depth': [20, 22, 24,],
 'min_samples_leaf': [1, 2, 4],
 'min_samples_split': [2, 5, 10],
 'n_estimators': range(80,200,15)}

GS_forest = GridSearchCV(RandomForestRegressor(),params,cv=5)
GS_forest.fit(X_train,y_train)

print("Best score :")
print(GS_forest.best_score_) 

model = RandomForestRegressor(**GS_forest.best_params_).fit(X_train, y_train)
print("\nScore on test set :\n", model.score(X_test, y_test))
print("="*20)

print(f"Saving model as {MODEL_NAME}")

dump(model, f'models/{MODEL_NAME}.joblib') 

print('Done')
