# practice
Contains random python files used to practice
This contains python files I created in order to just practice various things with python. There is really no large project with each of these. The main things I wanted to practice in these files was becoming more familar with the various popular python packages such as pandas, matplotlib, and numpy.

Summary of each file:
----------------------
NFLPrediction_Function(TL:DR at the end)
----------------------
  This file was used to practice data preparation and model creation for a predictive model. I used data from https://www.pro-football-reference.com, specifically the standings page, week by week schedule, and qb stats. I have not utilized the qb stats yet. I started by converting the standings page into a dataframe and getting rid of some of the special symbols used in the Tm(Team Name) column that were used to indicate their current playoff position. The next portion fo the code visualizes two of the features I am going to use in the predictive models, OSRS and DSRS, into a scatter plot. These stats are used to assess a teams overall offensive and defensive performance over the season, with 0 being the average for both stats. 
  
  I prepped the data and ran various different models on the data. After developing the models, I created a function, NFLPrediction that takes three inputs, away_team, home_team, and model_type. It will return who is predicted to win the game based on the model type as well as the prediction probability(In some of the models, it returns 100% for the prediciton probability)
  
TL:DR
A Python file that was mainly used to practice pandas and sklearn. The code prepares the data from the sources, creates a usable dataframe for the models, and runs them. Additionally, there is a function that takes a home and away team and returns who is predicted to win based on the models I ran, as well as the prediction probability.

Pokemon_Functions
---
  This was a file I used to practice functions and classes when starting up classes again in August 2022. As of January 10th, 2023, I am working on adding better comments to the file as I did not comment much at all while creating it. The functions in this file perform various things with pokemon stats and information, pulling from multiple dataframes. For the most part, the functions deal with calculating and visualizing the pokemon stats.
  
Data pulled from: https://github.com/PokeAPI/pokeapi/tree/master/data/v2/csv

CBB_Prediction
---
  This is very similar to the NFL Predicition file. In this, I focused on preparing data to be used in a function to predict the outcome of college basketball games. The data was pulled from https://www.sports-reference.com/cbb. This file has more models ran compared to the NFL one, as well as using SMOTE to compare asccuracy between using SMOTE and not adjusting the data at all(The outcome variable was about 70% one outcome, so not heavily imbalanced, but I thought it would be good practice to test it out). The function has a similar syntax of (home_team, away_team, model_type), and the names for the schools have to be entered in the proper case.
Future Additions:
- Better comments throughout the file
- Cross-Fold Validation to compare Accuracies for all of the models.
- Expand on the function with better error handling and more model types
- Having it iterate through the matchups for March Madness to predict the outcome.
