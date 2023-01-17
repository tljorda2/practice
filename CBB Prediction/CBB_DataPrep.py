#%%
# Importing all packages that might be used
import pandas as pd
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