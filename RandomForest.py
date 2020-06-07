from time import *
from pprint import pprint

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from DecisionTree import decision_tree_algorithm, decision_tree_predictions
from helper_functions import localtime_in_sec

# Option Display for Rows n Columns
desired_width = 300
pd.set_option('display.width', desired_width)
pd.set_option('display.max_columns', 100)
#pd.set_option('display.max_rows', 300)


# TODO Bootstrapping
def bootstrapping(data, n_bootstrap):

    bootstrap_indices = np.random.randint(low=0, high=len(data), size=n_bootstrap)
    df_bootstrapped = data.iloc[bootstrap_indices]

    return df_bootstrapped


# TODO Random Forest Model
def random_forest_algorithm(data, n_trees, n_bootstrap, n_features, dt_max_depth):
    forest = []
    for i in range(n_trees):
        df_bootstrapped = bootstrapping(data, n_bootstrap)
        tree = decision_tree_algorithm(df_bootstrapped, max_depth=dt_max_depth, random_subspace=n_features)
        forest.append(tree)

    return forest


# TODO Random Forest Prediction
def random_forest_predictions(data, forest):
    df_predictions = {}
    for i in range(len(forest)):
        col_name = "tree_{}".format(i)
        predictions = decision_tree_predictions(data, tree=forest[i])
        df_predictions[col_name] = predictions

    df_predictions = pd.DataFrame(df_predictions)
    random_forest_predictions = df_predictions.mode(axis=1)[0]

    return random_forest_predictions


# TODO Secondary data init
col_names = ['TotLenFwdPckt', 'FwdPcktLenMax', 'FwdPcktLenMean', 'AvgFwdSgmnSize',
             'SubFlFwdByts', 'InitWinBytsFwd', 'ActvData', 'Label']

dataframe = pd.read_excel("ddos_cicids2017.xlsx", names=col_names)


#TODO Rescaling for Secondary Data
scaled_sec_features = dataframe.copy()

column_names_sec = ['TotLenFwdPckt', 'FwdPcktLenMax', 'FwdPcktLenMean', 'AvgFwdSgmnSize',
             'SubFlFwdByts', 'InitWinBytsFwd', 'ActvData']

sec_features = scaled_sec_features[column_names_sec]
scaler = MinMaxScaler().fit(sec_features.values)
sec_features = scaler.transform(sec_features.values)

scaled_sec_features[column_names_sec] = sec_features
#print(scaled_features)


# TODO Splitting secondary data
train_sekunder, test_sekunder = train_test_split(scaled_sec_features, train_size= 0.25)


# TODO Primary init
primary = pd.read_excel("data_primer.xlsx", names=col_names)


# TODO Rescaling for Primary Data
scaled_prim_features = primary.copy()

column_names_prim = ['TotLenFwdPckt', 'FwdPcktLenMax', 'FwdPcktLenMean', 'AvgFwdSgmnSize',
             'SubFlFwdByts', 'InitWinBytsFwd', 'ActvData']

prim_features = scaled_prim_features[column_names_prim]
scaler = MinMaxScaler().fit(prim_features.values)
prim_features = scaler.transform(prim_features.values)

scaled_prim_features[column_names_prim] = prim_features


# TODO Splitting primary
train_primer, test_primer = train_test_split(scaled_prim_features, train_size= 0.5)

print("DATA TRAINING \n",train_primer)
print("==========")
print("DATA TESTING \n",test_primer)
print("==========")


# TODO Construct forest model
forest = random_forest_algorithm(train_primer, n_trees=50, n_bootstrap=800, n_features=6, dt_max_depth=10)

"""
print("==========")
pprint(forest)
"""

#TODO Starting time
now = localtime_in_sec(localtime)
print("Starting time:", now, "second")
print("==========")
predictions = random_forest_predictions(test_primer, forest)


# TODO Determine Predictions
print("DATA PREDIKSI \n",predictions)
print("==========")
print("DATA AKTUAL \n",test_primer.Label)


# TODO Ending time and Print Duration
# Ending time
later = localtime_in_sec(localtime)
print("==========")
print("Ending time: ",later, "second")
# Print Duration
duration = int(later-now)

# Determine total of benign and Ddos in each predictions
unique_clases, counts_unique_clasess = np.unique(predictions, return_counts=True)


# TODO Confusion matrix
# Determine TN, FP, FN, TP
print("==========")
from sklearn.metrics import confusion_matrix

tn, fp, fn, tp = confusion_matrix(test_primer.Label, predictions).ravel()
print("Berikut adalah confusion matrix : \n",confusion_matrix(test_primer.Label, predictions))
print("==========")
print(" True Negative: ",tn,"\n",
      "False Positive: ",fp,"\n",
      "False Negative: ",fn,"\n",
      "True Positive: ",tp)
print("==========")


# TODO Calculate Testing System
# accuracy, precisioin, recall, f-measure
if (tp + tn + fp + fn) != 0:
    accuracy = ((tp + tn) / (tp + tn + fp + fn)) * 100
else:
    accuracy = 0

if (tp+fp) != 0:
    precision = (tp / (tp + fp)) * 100
else:
    precision = 0

if (tp+fn) != 0:
    recall = (tp / (tp + fn)) * 100
else:
    recall = 0

if (precision+recall) != 0:
    f_measure = 2 * ((precision * recall) / (precision + recall))
else:
    f_measure = 0


# Print Output
print(unique_clases)
print(counts_unique_clasess)
print("==========")
print("duration", duration, "second")
print("akurasi: ", accuracy)
print("presisi: ", precision)
print("recall: ",recall)
print("f-measure: ",f_measure)



"""
n_bootstrap = 800
bootstrapped = bootstrapping(scaled_features, n_bootstrap)
print(bootstrapped)
"""


