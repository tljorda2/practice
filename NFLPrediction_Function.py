# Initial NFl Predicition
#%%
# Package imports for overall prject
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from timeit import default_timer as timer

#%%
# Reading the team data csv for the 2022-2023 season
team_statsdf = pd.read_csv('C:\\Users\\Timothy Jordan\\Desktop\\Pandas Practice\\NFL Predicition model\\TeamSeasonStats2022.csv')
team_statsdf = team_statsdf.sort_values('W', ascending=False)
print(team_statsdf)
""" for string in team_statsdf.Tm:
    if "*" in string or "+" in string:
        string2 = string[:-1]
        index1 = team_statsdf.loc[team_statsdf['Tm'] == string]
        team_statsdf.at[index1,'Tm'] = string2 """
for index, row in team_statsdf.iterrows():
    team_statsdf.loc[index, "Tm"] = row["Tm"].replace("*", "").replace("+", "")
print(team_statsdf.Tm)

#%%
# Making a graph to display all the teams and their offensive and defensive ratings
# I am going to plot OSRS and DSRS as they are aggregate team stats that give a decent view of 
# How the team has been doing overall in the season.
fig, ax = plt.subplots()
# This creates plot lines to make it easier to visualize where each team is
ax.axhline(0,color='grey',linewidth=1.5)

ax.axvline(0, color='grey', linewidth=1.5)
    

plt.scatter(team_statsdf['OSRS'], team_statsdf['DSRS'], marker=None, color='black')
plt.xlabel('OSRS')
plt.ylabel('DSRS')
plt.ylim(-8,8)
plt.xlim(-8,8)
plt.title('NFL Team OSRS / DSRS')

# This is a for loop that iterates through each team and adds the annotation to the graph
n = team_statsdf.Tm.values.tolist()
y = team_statsdf.OSRS.values.tolist()
z = team_statsdf.DSRS.values.tolist()
for i in range(len(n)):
    ax.annotate(n[i], (y[i], z[i]), xytext=(y[i]+0.2, z[i]+0.2),fontsize=5)
    
    
plt.show()
# %%
print(team_statsdf.head())
print(team_statsdf.info())
# %%
# Pro Football Focus QB scores
# https://www.pff.com/grades
QB_stats = pd.read_html('https://www.pro-football-reference.com/years/2022/passing.htm')
# For whatever reason, it gives me the dataframe inside a list
QB_stats = QB_stats[0]
# %%
print(QB_stats.head())

# %%
# Creating a dataframe that links up the team name and team abbreviation to connect the two stats data frames
unique_teams = QB_stats.Tm.unique().tolist()
unique_teams.remove('Tm')
unique_teams.remove('2TM')
unique_teams.sort()

team_names = team_statsdf.Tm.unique().tolist()
team_names.sort()

teams_df = {'team': team_names}
teams_df = pd.DataFrame(teams_df, columns=['team', 'team_name'], )
teams_df.team_name = unique_teams
# Changing some of the values that were not correct
teams_df.loc[28,'team_name'] = 'SEA'
teams_df.loc[22,'team_name'] = 'NOR'
teams_df.loc[21,'team_name'] = 'NWE'
teams_df.loc[16,'team_name'] = 'LAR'
teams_df.loc[17,'team_name'] = 'LAC'
teams_df.loc[27,'team_name'] = 'SFO'
print(teams_df)


# %%
# Getting the game by game outcomes
start = timer()
Games_df = pd.read_html('https://www.pro-football-reference.com/years/2022/games.htm')
Games_df = Games_df[0]

Games_df.sort_values('Unnamed: 5')
away_gamesdf = Games_df[Games_df['Unnamed: 5'] == '@']
home_gamesdf = Games_df[Games_df['Unnamed: 5'] != '@']

# I want to create a dummy variabel for whether or not the home team won
# The current direction I am working towards is to eventually combine the stats and games dataframes
# In order to sue team stats to predict matchups
# Since in the original dataframe all of the winners were on the left column titled "Winners/tie" I can create
# dummy variables like this.
home_gamesdf['Home_Win'] = 1
away_gamesdf['Home_Win'] = 0
# Assigning column values to a temporary dataframe to swap the values
temp_df1 = home_gamesdf['Winner/tie']
temp_df2 = home_gamesdf['Loser/tie']
home_gamesdf['Winner/tie'] = temp_df2
home_gamesdf['Loser/tie'] = temp_df1

home_gamesdf.rename(columns={'Winner/tie':'Away','Loser/tie': 'Home'}, inplace=True)
away_gamesdf.rename(columns={'Winner/tie':'Away','Loser/tie': 'Home'}, inplace=True)
Games_df = home_gamesdf.append(away_gamesdf)
Games_df = Games_df.drop(columns=['Unnamed: 5', 'Unnamed: 7', 'PtsW', 
                                  'PtsL','YdsW','YdsL','TOW', 'TOL'])
end = timer()
Games_df
#print(Games_df[(Games_df['Winner/tie'] == 'New England Patriots') | (Games_df['Loser/tie'] == 'New England Patriots')])
# %%
# This section merges the individuasl games dataframe with the overall team stats
# This allows me to use the overall stats to predict the outcome of the games
# I am using a left join to keep everything in line with the original games dataframe
Model_DataFrame = pd.merge(left=Games_df, right=team_statsdf, how='left', left_on='Away', right_on='Tm')
Model_DataFrame.rename(columns={'W':'Away-W','L':'Away-L','T':'Away-T',
                                'W-L%':'Away-W-L%','PF':'Away-PF',
                                'PA':'Away-PA','PD':'Away-PD','MoV':'Away-MoV',
                                'SoS': 'Away-SoS','SRS':'Away-SRS',
                                'OSRS':'Away-OSRS','DSRS':'Away-DSRS'})
Model_DataFrame = pd.merge(left=Model_DataFrame, right=team_statsdf, how='left', left_on='Home', right_on='Tm')
Model_DataFrame
# %%
# Preparing the prediction Model

Model_DataFrame.dropna(inplace=True)
Model_DataFrame.info()

# %%
X = Model_DataFrame.drop(columns=['Week', 'Day', 'Date', 'Time', 'Away', 'Home','Tm_x', 'Tm_y',
                                  'PF_x', 'PA_x', 'PF_y', 'PA_y', 'Home_Win'])
y = Model_DataFrame['Home_Win']
# %%
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=100)

# %%
'''Logistic Regression'''
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
target = ['Home Team Won', 'Away Team Won']
print(classification_report(y_test, y_pred, target_names=target))
# %%
'''Classification tree'''
# Currently running with a depth of 5
from sklearn import tree
from sklearn import metrics
cif = tree.DecisionTreeClassifier(criterion="entropy", max_depth=7)

cif = cif.fit(X_train,y_train)
### This predicts the response
y_pred = cif.predict(X_test)

### This gives you the accuracy of your predictions
print(metrics.accuracy_score(y_test, y_pred))

print(classification_report(y_test, y_pred, target_names=target))
# %%
'''Support Vector Machine'''
from sklearn import svm

#Creating a svm classifier
clf = svm.SVC(kernel='rbf',probability=True)

#Training
clf.fit(X_train, y_train)

#Prediciting
y_pred = clf.predict(X_test)

# SVM Metrics
print(classification_report(y_test, y_pred, target_names=target))

y_actual = pd.Series(y_test, name='Actual')
y_predicted = pd.Series(y_pred, name='Predicted')

metrics.confusion_matrix(y_actual, y_predicted)

#%%
'''Gradient Boost Ensemble'''
# This is the method that assigns a higher weighted value to wrong predicitions.
from sklearn.ensemble import GradientBoostingClassifier
gradient = GradientBoostingClassifier()
# Learning rate is a range from 0 to 1 that signifiers how much weight the model will put on wrong predicitons

gbe = GradientBoostingClassifier(n_estimators=100, learning_rate=0.6, max_depth=6, random_state=51).fit(X_train, y_train)
y_pred = gbe.predict(X_test)
print(classification_report(y_test, y_pred, target_names=target))
# The current configuration above has the highest f1-score for both targets out of all of the models.

#%%
'''Neural Network'''
from sklearn.neural_network import MLPClassifier
#Creating a neural net classifier
nn = MLPClassifier(hidden_layer_sizes=(5,9), random_state=2, max_iter=1000)

#Training the model
nn.fit(X_train, y_train)

#Prediciting the outcomes
y_pred = nn.predict(X_test)

print(classification_report(y_test, y_pred, target_names=target))

from sklearn import metrics
y_actual = pd.Series(y_test, name='Actual')
y_predicted = pd.Series(y_pred, name='Predicted')

metrics.confusion_matrix(y_actual, y_predicted)

# %%
'''Potential way to make a function to predict an outcome
Have the inputs be the team abreviations and match them to the full names to match with the other dfs
'''
def NFL_Predicition(away_team, home_team, model_type='svm'):
    # home = teams_df[teams_df['team_name']==home_team]
    # away = teams_df[teams_df['team_name']==away_team]
    # home = home.loc[2,'team']

    # away = away['team']
    # home = teams_df.loc[teams_df['team_name'] == home_team, 'team']
    # away = teams_df.loc[teams_df['team_name'] == away_team, 'team']
    
    # Check to see if the first two inputs are strings and making them proper case to match the way its formatted
    # in the table
    if (type(away_team) == str) and (type(home_team) == str) and (type(model_type) == str):
        away_team = away_team.title()
        home_team = home_team.title()
        model_type = model_type.lower()
    else:
        raise TypeError('The inputs for home_team and away_team must be strings')

    

    df = pd.DataFrame(columns=['Away', 'Home'])
    df = df.append({'Away': away_team, 'Home': home_team}, ignore_index=True)
    
 
    df_merged = pd.merge(left=df, right=team_statsdf, how='left', left_on='Away', right_on='Tm')
    df_merged = pd.merge(left=df_merged, right=team_statsdf, how='left', left_on='Home', right_on='Tm')
    

    X_test = df_merged.drop(columns=['Away', 'Home','Tm_x', 'Tm_y',
                            'PF_x', 'PA_x', 'PF_y', 'PA_y'])
    
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


NFL_Predicition('Detroit Lions', 'Green Bay Packers', model_type='svm')

# %%
