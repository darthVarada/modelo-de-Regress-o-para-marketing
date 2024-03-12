# imports
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_predict
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import make_scorer, mean_squared_error, accuracy_score
from scipy.stats.mstats import winsorize
import numpy as np

# load model
def extract_data():
    data = pd.read_csv("MKT.csv")
    return data
#Remove values beyond 3 standard deviations
def data_cleaning(data):
    mean_value = data['newspaper'].mean()
    std_dev = data['newspaper'].std()
    upper_threshold = mean_value + 3 * std_dev
    df_filtered = data[data['newspaper'] <= upper_threshold]
    #limit the values to 3 standard deviations
    data['newspaper'] = winsorize(data['newspaper'], limits=(0, 0.05))
def prepare_features(data):
    X = data[['facebook', 'youtube', 'newspaper']]
    y = data['sales']
    return X,y

def train_split(X,y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=42)
    return X_train, X_test, y_train, y_test

def train_model(X_train, y_train):
    regLinear = LinearRegression().fit(X_train, y_train)
    return regLinear
def predict(X_test, y_test, regLinear):
    y_pred_linear = regLinear.predict(X_test)
    return y_pred_linear


# Define your model and dataset
def linear_regression(X, y):
    model_cross = LinearRegression()  
    return model_cross

def cross_val_score(model_cross, X_cross, y_cross, metric, kf):
    cross_val_results = cross_val_score(model_cross, X_cross, y_cross, cv=kf, scoring=metric)
    return cross_val_results


# Configure cross-validation with KFold
def cross_val_score(model_cross, X_cross, y_cross, metric):
    kf = KFold(n_splits=5, shuffle=True, random_state=42)  # Number of folds (here, 5)
    return kf
# Perform cross-validation and get the mean of the metrics
def cross_val_score(model_cross, X_cross, y_cross, metric, kf):
    cross_val_results = cross_val_predict(model_cross, X_cross, y_cross, cv=kf, method='predict')
    return cross_val_results

# Display the results
def display_results(cross_val_results, y_pred_linear, y_test):
    print(f'Metrics per fold: {cross_val_results}')
    print(f'Mean of the metrics: {cross_val_results.mean()}')

    # Define a threshold to classify predictions as success or failure
    threshold = 2.0  # Example threshold, adjust as needed

    # Convert continuous predictions into binary classes
    y_pred_binary = (y_pred_linear <= threshold).astype(int)
    y_test_binary = (y_test <= threshold).astype(int)

    # Calculate accuracy
    accuracy_binary = accuracy_score(y_test_binary, y_pred_binary)
    print(f'Accuracy: {accuracy_binary:.2f}')

    mean_squared_error(y_test_binary, y_pred_binary)

# Serialize o objeto do modelo
def serialize_object(model_cross):
    with open("trained_classifier.pkl", "wb") as file:
        pickle.dump(model_cross, file)

if __name__ == "__main__":
    data = extract_data()
    data_cleaning(data)
    X_cross, y_cross = prepare_features(data)
    X_train, X_test, y_train, y_test = train_split(X_cross, y_cross)
    regLinear = train_model(X_train, y_train)
    y_pred_linear = predict(X_test, y_test, regLinear)
    model_cross = linear_regression(X_cross, y_cross)
    metric = mean_squared_error
    kf = cross_val_score(model_cross, X_cross, y_cross, metric=metric, kf=5)
    display_results(cross_val_results=kf, y_pred_linear=y_pred_linear, y_test=y_test)
    
    # Serialize o modelo treinado
    serialize_object(regLinear)
