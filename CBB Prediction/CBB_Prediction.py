#%%
# Importing all packages that might be used
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
#%%
# Reading all of the data and converting them into dataframes
team_stats = pd.read_html('https://www.sports-reference.com/cbb/seasons/2023-school-stats.html', header=1)
oppenent_stats = pd.read_html('https://www.sports-reference.com/cbb/seasons/2023-opponent-stats.html', header=1)
advanced_stats = pd.read_html('https://www.sports-reference.com/cbb/seasons/2023-advanced-school-stats.html', header=1)
opp_advanced_stats = pd.read_html('https://www.sports-reference.com/cbb/seasons/2023-advanced-opponent-stats.html', header=1)
# read_html turned it into a list that contains the dataframe
# By indexing the 0th position of the list, it gives us the dataframe outside of the lsit
team_stats = team_stats[0]
oppenent_stats = oppenent_stats[0]
advanced_stats = advanced_stats[0]
opp_advanced_stats = opp_advanced_stats[0]
print(oppenent_stats.info())
# %%
# Columns to be dropped
# cols represents the columns dropped for the team stats dataframe. Some of the first few columns are repeated
# in the other dataframes, so we can keep them in the first one, and drop them from the rest
cols = [0,2,3,4,8,11,12,13,14,17,20]
cols2 = [0,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]

team_stats.drop(columns=team_stats.columns[cols], axis=1,inplace=True)
oppenent_stats.drop(columns=oppenent_stats.columns[cols2], axis=1,inplace=True)
advanced_stats.drop(columns=advanced_stats.columns[cols2], axis=1,inplace=True)
opp_advanced_stats.drop(columns=opp_advanced_stats.columns[cols2], axis=1,inplace=True)

# Dropping all null values based on the rows
team_stats.dropna(axis=0, inplace=True)
opp_advanced_stats.dropna(axis=0, inplace=True)
advanced_stats.dropna(axis=0, inplace=True)
oppenent_stats.dropna(axis=0, inplace=True)

#Dropping any duplicate values from the tables. I believe this dropped around 10 rows
team_stats.drop_duplicates('School',inplace=True)
opp_advanced_stats.drop_duplicates('School',inplace=True)
advanced_stats.drop_duplicates('School',inplace=True)
oppenent_stats.drop_duplicates('School',inplace=True)

# %%
# Comparing the information for all of the dataframes and making sure the value counts line up and are all not null
print(team_stats.info())
print(oppenent_stats.info())
print(advanced_stats.info())
print(opp_advanced_stats.info())

#%%
# Merging all of the stats dataframes into one.
# Merged all of them left inter the team_statsdf on school name
statsdf = pd.merge(left=team_stats,right=oppenent_stats,how='inner', on='School',)
statsdf = pd.merge(left=statsdf, right=advanced_stats, on='School', how='inner')
statsdf = pd.merge(left=statsdf, right=opp_advanced_stats,how='inner', on='School')
print(statsdf)
#%%
# Oncew again, dropping any null values that were created with the merge
statsdf.dropna(axis=0, inplace=True)
statsdf.info()

#%%
# Since most of the data types listed above are "object", we need to convert those into a numeric form
# we use label encoding to do this

from sklearn.preprocessing import LabelEncoder
#Initialize
le = LabelEncoder()

# Used a for loop since everything needed to be transformed into a numerical type aside from the first column, school
columns = statsdf.keys()
for i in columns:
    if i == 'School':
        pass
    else:
        statsdf[i] = le.fit_transform(statsdf[i])

statsdf.info()

# %%
gamesdf = pd.read_csv('C:\\Users\\Timothy Jordan\\Desktop\\Pandas Practice\\CBB Prediction\\college_games.csv')
gamesdf.info()
# %%
# Defining a list of columns to be dropped from the gamesdf. These are columns we are not interested in
drop_col = ['Year', 'Date', 'Unnamed: 2', 'Rk.', 'Unnamed: 5', 'Rk..1'
            , 'Unnamed: 9', 'PTS', 'OPP', 'MOV', 'Unnamed: 13', 'OT']
gamesdf.drop(columns=drop_col, axis=1, inplace=True)
gamesdf.dropna(axis=0,inplace=True)
gamesdf.info()
# %%
# Renaming the unnamed column to Home_Win since this is going to be our target variable
gamesdf.rename(columns={'Unnamed: 8': 'Home_Win'}, inplace=True)
# Felt better by transforming it into a binary outcome of 1 and 0 with 1 representing a Home Team Win
gamesdf['Home_Win'] = le.fit_transform(gamesdf['Home_Win'])
gamesdf.head(10)
# %%
# Merging the individual games with the stats for the teams that played in those games
# First merging the home team stats and then the away teasm stats
# Used a left join and joined on the school names
Model_DataFrame = pd.merge(left=gamesdf, right=statsdf, how='left', left_on='Schl', right_on='School')
Model_DataFrame = pd.merge(left=Model_DataFrame, right=statsdf, how='left', left_on='Opp', right_on='School')
print(Model_DataFrame.info())
Model_DataFrame.head(5)
# %%
# Dropping null values once again from the final dataframe and checking to see if everything looks right
Model_DataFrame.dropna(axis=0,inplace=True)
# Had to use verbose and show_counts since it did not default to it for whatever reason
Model_DataFrame.info(verbose=True, show_counts=True)
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
plt.xlim(0.70,0.82)
plt.ylabel('Models')
plt.title(f'Original Dataset Accruacy Scores\nAverage: {round(fmean(Original_Accuracy), 4)}')
plt.show()

ax = sns.barplot(x=SMOTE_Accuracy, y=Names)
plt.xlabel(f'Accuracy Score')
plt.xlim(0.70,0.82)
plt.ylabel('Models')
plt.title(f'SMOTE Dataset Accruacy Scores\nAverage: {round(fmean(SMOTE_Accuracy), 4)}')
plt.show()
# %%
def CBB_Prediction(home_team, away_team, model_type='log'):
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
    else:
        raise ValueError('Please enter a valid model type. The valid models are:\nsvm\nlog\ndtc\ngbe\nnn')
    
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
CBB_Prediction('Arizona', 'Southern California')
# %%
team_stats['School']
# %%
Model_DataFrame[Model_DataFrame['Opp'] == 'UCLA']
# %%
