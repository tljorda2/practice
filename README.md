# practice
Contains random python files used to practice
This contains python files I created in order to just practice various things with python. There is really no large project with each of these. The main things I wanted to practice in these files was becoming more familar with the various popular python packages such as pandas, matplotlib, and numpy.

Summary of each file:
----------------------
NFLPrediction_Function(TL:DR at the end)
----------------------
  This file was used to practice data preparation and model creation for a predictive model. I used data from https://www.pro-football-reference.com, specifically the standings page, week by week schedule, and qb stats. I have not utilized the qb stats yet. I started by converting the standings page into a dataframe and getting rid of some of the special symbols used in the Tm(Team Name) column that were used to indicate their current playoff position. The next portion fo the code visualizes two of the features I am going to use in the predictive models, OSRS and DSRS, into a scatter plot. These stats are used to assess a teams overall offensive and defensive performance over the season, with 0 being the average for both stats. 

  Following that is a section that creates a new dataframe that can be used to link full team names to their abbreviations used by Pro Football Reference in their player stats pages in case there is a need to to combine the player stats to the team stats using the full team name. The next portion of the code involves the final data preparation for the model. This included rearranging the columns from "Winner/tie" and "Loser/tie" in the individual games dataframe into home and away. I then created a new variable indicating whether or not the home team won the game. That column is titled "Home_win" and is going to be the thing I want to predict.
  
  The next step is to combine the team_statsdf with the Games_df to have the features I want to use to predict the outcome on the same dataframe of the individual games. I used the pandas merge function with left joining the team_statsdf into the Games_df. After that, I dropped some of the columns I did not want to use in the models such as date, time, the team names and more. I specifically dropped the points for and points against columns for both the home and away teams since I was just going to use the point differential column.
  
  From there, I ran a logistic regression, neural network, support vector machine, gradient boost ensemble model and a classification tree, all from sklearn. With a model like this, precision and recall are not as important metrics to look at since the importance of false positives and false negatives are equal, therefore I want to look at the overall accuracy of the model. The results of each of the models were pretty okay, with accuracy hovering around 0.6 to 0.77, with the support vector machine have the higest accuracy.
  
  After developing the models, I created a function, NFLPrediction that takes three inputs, away_team, home_team, and model_type. It will return who is predicted to win the game based on the model type as well as the prediction probability(In some of the models, it returns 100% for the prediciton probability)
  
TL:DR
A Python file that was mainly used to practice pandas and sklearn. The code prepares the data from the sources, creates a usable dataframe for the models, and runs them. Additionally, there is a function that takes a home and away team and returns who is predicted to win based on the models I ran, as well as the prediction probability.

Pokemon_Functions
---
  This was a file I used to practice functions and classes when starting up classes again in August 2022. As of January 10th, 2023, I am working on adding better comments to the file as I did not comment much at all while creating it. The functions in this file perform various things with pokemon stats and information, pulling from multiple dataframes. For the most part, the functions deal with calculating and visualizing the pokemon stats.
