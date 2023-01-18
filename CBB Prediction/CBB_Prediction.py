#%%
# Importing all packages that might be used
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
from CBB_DataPrep import *
import xgboost as xge
# Library for optimizing models
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
# %%
# Splitting the data into X and y. Dropped any of the features we are not interested in.
X = Model_DataFrame.drop(columns=['Schl', 'Opp', 'School_x', 'Home_Win', 'School_y'],axis=1)
y = Model_DataFrame['Home_Win']
#%%
# Creating a bar chart to show the count of wins vs losses for the home team
# There seems to be an imbalance so I should address this before splitting the data into X and y
dt = y.value_counts(0)
x_1 = ['no', 'yes']
y_2 = [dt[0], dt[1]]
plt.bar(x_1, y_2)
plt.show
plt.title('College Game Home Wins')
plt.xlabel('Home Win?')
plt.ylabel('Count of Wins')
'''
When running all of the models, there is a clear bias towards selecting the majority value of the home team winning
I need to address this and will probably use SMOTE
'''
# %%
# Splitting the data into a train/test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=100)
#Getting the value counts of each
print(y_train.value_counts(0))
print(y_train.value_counts(1))
#%%
# Addressing the imbalance with SMOTE to see if it improves accruacy
from imblearn.over_sampling import SMOTE
sm = SMOTE(random_state = 1)
X_train_res, y_train_res = sm.fit_resample(X_train, y_train.ravel())
'''
After checking all of the data later and the accuracy scores, using SMOTE might be unnecesary
The accuracy hardly changfes with about a 2% difference overall for all the models with using the original data
generally being more accurate.
'''



# %%
'''Logistic Regression'''
# Comparison between the re sampled data and the original data
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn import metrics

target = ['Away Team Won', 'Home Team Won']
# Fitting and running the model for the resampled data
model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print('Logistic Regression')
print('ORIGINAL DATA\n---------------')
print(classification_report(y_test, y_pred, target_names=target))
# Storing the accuracy score as a variable to use in a plot
Logistic_Accuracy = metrics.accuracy_score(y_test, y_pred)



# %%
'''Classification tree'''
# Currently running with a depth of 5
from sklearn import tree

cif = tree.DecisionTreeClassifier(criterion="entropy", max_depth=5)

cif = cif.fit(X_train,y_train)
### This predicts the response
y_pred = cif.predict(X_test)

### This gives you the accuracy of your predictions
print('Classification tree, Entropy, Max Depth: 5')
print('ORIGINAL DATA\n---------------')
print(metrics.accuracy_score(y_test, y_pred))

# Storing Accruacy to call later for comparison
ClassTree_Accuracy = metrics.accuracy_score(y_test, y_pred)
print(classification_report(y_test, y_pred, target_names=target))




# %%
'''Support Vector Machine'''
from sklearn import svm

#Creating a svm classifier
clf = svm.SVC(kernel='rbf',probability=True)
# ORIGINAL DATA
#Training
clf.fit(X_train, y_train)

#Prediciting
y_pred = clf.predict(X_test)

# SVM Metrics
print('ORIGINAL DATA\n---------------')
print(classification_report(y_test, y_pred, target_names=target))

y_actual = pd.Series(y_test, name='Actual')
y_predicted = pd.Series(y_pred, name='Predicted')

metrics.confusion_matrix(y_actual, y_predicted)
SupportVector_Accuracy = metrics.accuracy_score(y_test, y_pred)





#%%
'''Gradient Boost Ensemble'''
# This is the method that assigns a higher weighted value to wrong predicitions.
from sklearn.ensemble import GradientBoostingClassifier
gradient = GradientBoostingClassifier()
# Learning rate is a range from 0 to 1 that signifiers how much weight the model will put on wrong predicitons

gbe = GradientBoostingClassifier(n_estimators=100, learning_rate=0.8, max_depth=7, random_state=51).fit(X_train, y_train)
y_pred = gbe.predict(X_test)

print('Gradient Boost Ensemble, Learning Rate=0.8, Max Depth: 7')
print('ORIGINAL DATA\n-------------')
print(classification_report(y_test, y_pred, target_names=target))
GradientBoost_Accuracy = metrics.accuracy_score(y_test, y_pred)
# The current configuration above has the highest f1-score for both targets out of all of the models.





#%%
'''Neural Network'''
from sklearn.neural_network import MLPClassifier
#Creating a neural net classifier
nn = MLPClassifier(hidden_layer_sizes=(7,9), random_state=2, max_iter=1000)

#Training the model
nn.fit(X_train, y_train)

#Prediciting the outcomes
y_pred = nn.predict(X_test)

print('Neural Network, Layers: (7, 9)')
print('ORIGINAL DATA\n--------------')
print(classification_report(y_test, y_pred, target_names=target))

#Accuracy Variable
NeuralNet_Accuracy = metrics.accuracy_score(y_test, y_pred)

y_actual = pd.Series(y_test, name='Actual')
y_predicted = pd.Series(y_pred, name='Predicted')

print(metrics.confusion_matrix(y_actual, y_predicted))




# %%
# K-nearest
from sklearn.neighbors import KNeighborsClassifier
# Creating a classifier
knn = KNeighborsClassifier(n_neighbors=5)
# Fitting the model
knn.fit(X_train, y_train)
# Predicting the outcome
y_pred = knn.predict(X_test)
# Printing the classification report
print('K-Nearest Neightbors')
print('ORIGINAL DATA\n-------------')
print(classification_report(y_test, y_pred, target_names=target))
KNN_Accuracy = metrics.accuracy_score(y_test, y_pred)



#%%
from sklearn.ensemble import RandomForestClassifier
# Classifier
random_forest = RandomForestClassifier(n_estimators=100, random_state=101)
random_forest.fit(X_train, y_train)
# Accuracy
y_pred = random_forest.predict(X_test)
# Printing the classification report
print('ORIGINAL DATA\n----------------')
print(classification_report(y_test, y_pred, target_names=target))
RandomForest_Accuracy = metrics.accuracy_score(y_test, y_pred)




#%%
'''XGBoost Model'''
from sklearn.metrics import mean_squared_error
# XGBoost can use three inputs that are all dictionaries
# reg:squarederror for linear regression
# reg:logistic for logistic regression
# binary:logistic for logistyic regression with probabilities
# verbosity has parameters of 0, 1, 2, 3. 0 is nothing, 1 is warning, 2 is info
# eta is a parameter for its learning rate
xgbr = xge.XGBClassifier(objective='binary:logistic', verbosity=0, eta=0.1119, max_depth=4, min_child_weight=6,
                        reg_alpha=41,reg_lambda=0.0824, colsample_bytree=.8035, gamma=2.757)
xgbr.fit(X_train, y_train)
y_pred = xgbr.predict(X_test)
print('ORIGINAL DATA\n----------------')
predictions = [round(value) for value in y_pred]
accuracy = metrics.accuracy_score(y_test, predictions)
print(accuracy)

#%%
# Attempting to optimize the model

space={'max_depth': hp.quniform("max_depth", 3, 18, 1),
        'gamma': hp.uniform ('gamma', 1,9),
        'reg_alpha' : hp.quniform('reg_alpha', 40,180,1),
        'reg_lambda' : hp.uniform('reg_lambda', 0,1),
        'colsample_bytree' : hp.uniform('colsample_bytree', 0.5,1),
        'min_child_weight' : hp.quniform('min_child_weight', 0, 10, 1),
        'eta': hp.uniform('eta',0,1),
        'n_estimators': 180,
        'seed': 0
    }

def objective(space):
    clf=xge.XGBClassifier(objective='binary:logistic',
                    n_estimators =space['n_estimators'], max_depth = int(space['max_depth']), gamma = space['gamma'],
                    reg_alpha = int(space['reg_alpha']),min_child_weight=int(space['min_child_weight']), eta=int(space['eta']),
                    colsample_bytree=int(space['colsample_bytree']),use_label_encoder=False)
    
    evaluation = [( X_train, y_train), ( X_test, y_test)]
    
    clf.fit(X_train, y_train,
            eval_set=evaluation, eval_metric="auc",
            early_stopping_rounds=10,verbose=False)
    

    pred = clf.predict(X_test)
    accuracy = metrics.accuracy_score(y_test, pred>0.5)
    print ("SCORE:", accuracy)
    return {'loss': -accuracy, 'status': STATUS_OK }

trials = Trials()

best_hyperparams = fmin(fn = objective,
                        space = space,
                        algo = tpe.suggest,
                        max_evals = 100,
                        trials = trials)
#%%
# Best Hyperparameters for XGBoost
print(best_hyperparams)

#%%
'ALL REBALANCED DATA MOVED HERE'
# All SMOTE rebalanced data was moved to this cell
# If this is ran before using the prediciton function, the function will use the rebalanced data
# GTenerally, it appeared that using SMOTE did not help the dataset much overall
# Fitting and running the model for the resampled data
'''Logistic Regression'''
model.fit(X_train_res, y_train_res)
y_pred = model.predict(X_test)

print('REBALANCED DATA\n---------------')
print(classification_report(y_test, y_pred, target_names=target))
# Storing the accuracy score as a variable to use in a plot
Logistic_Accuracy_SMOTE = metrics.accuracy_score(y_test, y_pred)

'''Random Forest'''
random_forest.fit(X_train_res, y_train_res)
# Accuracy
y_pred = random_forest.predict(X_test)
# Printing the classification report
print('REBALANCED DATA\n------------------')
print(classification_report(y_test, y_pred, target_names=target))
RandomForest_Accuracy_SMOTE = metrics.accuracy_score(y_test, y_pred)

'''KNN'''
#REBALANCED
# Fitting the model
knn.fit(X_train_res, y_train_res)
# Prediciting if they made the NBA or not
y_pred = knn.predict(X_test)
# Printing the classification report
print('REBALANCED DATA\n-------------')
print(classification_report(y_test, y_pred, target_names=target))
KNN_Accuracy_SMOTE = metrics.accuracy_score(y_test, y_pred)

'''Neural Network'''
nn.fit(X_train_res, y_train_res)

#Prediciting the outcomes
y_pred = nn.predict(X_test)

print('REBALANCED DATA\n----------------')
print(classification_report(y_test, y_pred, target_names=target))
# Accuracy Variable
NeuralNet_Accuracy_SMOTE = metrics.accuracy_score(y_test, y_pred)

y_actual = pd.Series(y_test, name='Actual')
y_predicted = pd.Series(y_pred, name='Predicted')


metrics.confusion_matrix(y_actual, y_predicted)

'''Gradient Boost Ensemble'''
# Refit Data
gbe = GradientBoostingClassifier(n_estimators=100, learning_rate=0.8, max_depth=7, random_state=51).fit(X_train_res, y_train_res)
y_pred = gbe.predict(X_test)
print('REBALANCED DATA\n-------------')
print(classification_report(y_test, y_pred, target_names=target))
GradientBoost_Accuracy_SMOTE = metrics.accuracy_score(y_test, y_pred)

'''Support Vector Machine'''
# REBALANCED
clf.fit(X_train_res, y_train_res)

#Prediciting
y_pred = clf.predict(X_test)
# SVM Metrics
print('REBALANCED DATA\n---------------')
print(classification_report(y_test, y_pred, target_names=target))

y_actual = pd.Series(y_test, name='Actual')
y_predicted = pd.Series(y_pred, name='Predicted')

metrics.confusion_matrix(y_actual, y_predicted)
SupportVector_Accuracy_SMOTE = metrics.accuracy_score(y_test, y_pred)

'''Decision Tree'''
#REBALANCED
cif = cif.fit(X_train_res,y_train_res)
### This predicts the response
y_pred = cif.predict(X_test)

### This gives you the accuracy of your predictions
print('REBALANCED DATA\n--------------')
print(metrics.accuracy_score(y_test, y_pred))
# Storing Accuracy as a variable to be called back later to compare
ClassTree_Accuracy_SMOTE = metrics.accuracy_score(y_test, y_pred)

print(classification_report(y_test, y_pred, target_names=target))



#%%
'''Getting 10 most important features for Random Forest'''
feature_imp = pd.Series(random_forest.feature_importances_, index=X.columns).sort_values(ascending=False)
feature_imp = feature_imp.nlargest(10)
# This is another thing we can reference when talking about the important features.
# May be useful to compare the differences from this model compared to the logistic regression
# Graph of the data above
# Citation: https://www.datacamp.com/tutorial/random-forests-classifier-python
import matplotlib.pyplot as plt
import seaborn as sns
# %matplotlib inline
# Creating a bar plot
# This will show the ten most important features
ax = sns.barplot(x=feature_imp, y=feature_imp.index)
ax.tick_params(axis='y', labelsize=10)
# Adding labels to the graph
plt.xlabel('Feature Importance Score')
plt.ylabel('Features')
plt.title("Visualizing Important Features")
plt.legend()
plt.show()
# %%
'''Getting 10 least important features for Random Forest'''
feature_imp = pd.Series(random_forest.feature_importances_, index=X.columns).sort_values(ascending=False)
feature_imp = feature_imp.nsmallest(10)
# This is another thing we can reference when talking about the important features.
# May be useful to compare the differences from this model compared to the logistic regression
# Graph of the data above
# Citation: https://www.datacamp.com/tutorial/random-forests-classifier-python

# %matplotlib inline
# Creating a bar plot
# This will show the feature importance for the ten least important features
ax = sns.barplot(x=feature_imp, y=feature_imp.index)
ax.tick_params(axis='y', labelsize=10)
# Adding labels to the graph
plt.xlabel('Feature Importance Score')
plt.ylabel('Features')
plt.title("Visualizing Important Features")
plt.legend()
plt.show()
# %%
# Feature Importance List
feature_imp = pd.Series(random_forest.feature_importances_, index=X.columns).sort_values(ascending=False)
print(feature_imp)
# %%
# Visualizing Accuracy Scores for Comparison between SMOTE and the original dataset
from statistics import fmean
SMOTE_Accuracy = [Logistic_Accuracy_SMOTE, ClassTree_Accuracy_SMOTE, GradientBoost_Accuracy_SMOTE,
                  NeuralNet_Accuracy_SMOTE, KNN_Accuracy_SMOTE, RandomForest_Accuracy_SMOTE, SupportVector_Accuracy_SMOTE]
Original_Accuracy = [Logistic_Accuracy, ClassTree_Accuracy, GradientBoost_Accuracy, NeuralNet_Accuracy,
                     KNN_Accuracy, RandomForest_Accuracy, SupportVector_Accuracy]
Names = ['Logistic Regression', 'Classification Tree', 'Gradient Boost', 'Neural Network',
         'K-Nearest Neighbors', 'Random Forest', 'Support Vector Machine']
# This compares the accuracy scores for all the models ran between using SMOTE and using the original dataset
ax = sns.barplot(x=Original_Accuracy, y=Names)
plt.xlabel(f'Accuracy Score')
plt.xlim(0.65,0.82)
plt.ylabel('Models')
plt.title(f'Original Dataset Accruacy Scores\nAverage: {round(fmean(Original_Accuracy), 4)}\nAxis Condensed')
plt.show()

ax = sns.barplot(x=SMOTE_Accuracy, y=Names)
plt.xlabel(f'Accuracy Score')
plt.xlim(0.65,0.82)
plt.ylabel('Models')
plt.title(f'SMOTE Dataset Accruacy Scores\nAverage: {round(fmean(SMOTE_Accuracy), 4)}\nAxis Condensed')
plt.show()
# %%
def CBB_Prediction(home_team, away_team, model_type='xgbr'):
    # setting the model type to lowercase in order for it to be uniform
    # for now, the home and away team inputs need to be entered in their proper casing
    # Some schools are all caps like UCLA while others are proper case like Arizona State
    if (type(away_team) == str) and (type(home_team) == str) and (type(model_type) == str):
        model_type = model_type.lower()
    else:
        raise TypeError('The inputs for home_team and away_team must be strings')

    # Creating a new dataframe that will become the X_test after merging the stats into the teams

    df = pd.DataFrame(columns=['Schl', 'Opp'])
    df = df.append({'Schl': home_team, 'Opp': away_team}, ignore_index=True)
    
 
    df_merged = pd.merge(left=df, right=statsdf, how='left', left_on='Schl', right_on='School')
    df_merged = pd.merge(left=df_merged, right=statsdf, how='left', left_on='Opp', right_on='School')
    

    X_test = df_merged.drop(columns=['Schl', 'Opp', 'School_x', 'School_y'],axis=1)
    
    # Checking the model_type input. By default, it is going to use a Support Vector Machine
    # I added this just to see how different model predicition probabilities were affected
    # These use the parameters for each model entered when they were ran above
    if model_type == 'svm':
        y_pred = clf.predict(X_test)
        prob = clf.predict_proba(X_test)
    elif model_type == 'log':
        y_pred = model.predict(X_test)
        prob = model.predict_proba(X_test)
    elif model_type == 'dtc':
        y_pred = cif.predict(X_test)
        prob = cif.predict_proba(X_test)
    elif model_type == 'gbe':
        y_pred = gbe.predict(X_test)
        prob = gbe.predict_proba(X_test)
    elif model_type == 'nn':
        y_pred = nn.predict(X_test)
        prob = nn.predict_proba(X_test)
    elif model_type == 'xgbr':
        y_pred = xgbr.predict(X_test)
        prob = xgbr.predict_proba(X_test)
    else:
        raise ValueError('Please enter a valid model type. The valid models are:\nsvm\nlog\ndtc\ngbe\nnn\nxgbr')
    
    #Checking if the home team won in order to give the proper probability of the outcome
    if y_pred[0] == 1:
        print(f'The {home_team}(home) are predicted to beat the {away_team} in this matchup')
        #prob = clf.predict_proba(X_test)
        prob = round(prob[0][1],4)*100

        print(f'Estimated Probability of a win: {prob}%')
    else:
        print(f'The {away_team}(away) are predicted to beat the {home_team} in this matchup')
        #prob = clf.predict_proba(X_test)
        prob = round(prob[0][0],4)*100

        print(f'Estimated Probability of a win: {prob}%')

#%%
CBB_Prediction('Kansas', 'UCLA', 'xgbr')
# %%
team_stats['School']
# %%
Model_DataFrame[Model_DataFrame['Opp'] == 'UCLA']
# %%
