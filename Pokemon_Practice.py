# -*- coding: utf-8 -*-
"""
Created on Wed Aug 24 16:02:55 2022


"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import random
#%%
'''Potential Cell for getting data from the web'''
# Can only call this cell with internet, so I sectioned this off from the rest of the data
url = 'https://raw.githubusercontent.com/PokeAPI/pokeapi/master/data/v2/csv/type_efficacy.csv'
typechart = pd.read_csv(url)
url1 = 'https://raw.githubusercontent.com/PokeAPI/pokeapi/master/data/v2/csv/items.csv'
items = pd.read_csv(url1)
url2 = 'https://raw.githubusercontent.com/PokeAPI/pokeapi/master/data/v2/csv/natures.csv'
natures = pd.read_csv(url2)
url3 = 'https://raw.githubusercontent.com/PokeAPI/pokeapi/master/data/v2/csv/abilities.csv'
abilities = pd.read_csv(url3)
url4 = 'https://raw.githubusercontent.com/PokeAPI/pokeapi/master/data/v2/csv/pokemon_abilities.csv'
pkabilities = pd.read_csv(url4)
#%%
# Add nature into the stat calculator and add it as a input for the statistics method of pkstorage
stats = pd.read_csv("C:\\Users\\cacru\\OneDrive\\Documents\\pokemon_stats.csv")
species = pd.read_csv("C:\\Users\\cacru\\OneDrive\\Documents\\pokemon_species.csv")
types = pd.read_csv("C:\\Users\\cacru\\OneDrive\\Documents\\types.csv")
ptypes = pd.read_csv("C:\\Users\\cacru\\OneDrive\\Documents\\pokemon_types.csv")
movesdf = pd.read_csv("C:\\Users\\cacru\\OneDrive\\Documents\\Data Analysis Practice\\moves.csv")
movesdf.drop(columns='contest_type_id')
movesdf.drop(columns='contest_effect_id')
movesdf.drop(columns='super_contest_effect_id')
statsdf = pd.DataFrame(stats)
speciesdf = pd.DataFrame(species)
typesdf = pd.DataFrame(types)
#%%
# This cell, I want to combine species, ptypes, and stats all into one data frame.
# First, lets work with adding the stats to the species df.
# The complication with this is the way the stats df is organized vertically. 
f = statsdf.query('pokemon_id == 1')
print(f)

ptypesdf = pd.DataFrame(ptypes)

#%%


#%%
def bst(x):
    y = statsdf[statsdf['pokemon_id']==x]
    lst = list(y['base_stat'])
    bstsum = 0
    for i in lst:
        bstsum = bstsum + i
    print('BST =', bstsum)
def types(x):
        y = ptypesdf[ptypesdf['pokemon_id']==x]
        t = y['type_id'].values.tolist()
        if int(len(t)) == 1:
            k = typesdf[typesdf['id']==t[0]]
            s = k['identifier'].values.tolist()
            print('Type:', s)
        else:
            k = typesdf[typesdf['id']==t[0]]
            s = k['identifier'].values.tolist()
            print('Type 1:', s)
            k1 = typesdf[typesdf['id']==t[1]]
            s1 = k1['identifier'].values.tolist()
            print('Type 2:', s1)
def pokemoninfo(x):
    y = speciesdf[speciesdf['id']==x]
    print('Name:', y['identifier'].values.tolist())
    print('Generation:', str(y['generation_id'].values.tolist()))
    if int(y['gender_rate']) < 0:
        print('No Gender')
    else:
        print('Percentage Male:', round((int(y['gender_rate'])/8)*100,3))
    print('Capture Rate:', y['capture_rate'].values.tolist())
    bst(x)  
def pokemoninfobyname(x):
    y = speciesdf[speciesdf['identifier']==x]
    print('Pokedex No.:', y['id'].values.tolist())
    x = int(y['id'])
    types(x)
    print('Generation:', str(y['generation_id'].values.tolist()))
    if int(y['gender_rate']) < 0:
        print('No Gender')
    else:
        print('Percentage Male:', round((int(y['gender_rate'])/8)*100,3))
    print('Capture Rate:', y['capture_rate'].values.tolist())
    bst(x)
def statfinder(x):
    y = speciesdf[speciesdf['identifier']==x]
    z = int(y['id'])
    #Level 50 min stats
    df1 = statsdf[statsdf['pokemon_id']==z]
    return df1
def fiftymin(x):
    df1 = statfinder(x)
    ### This section locates all the stat values from the csv files
    hp = df1['base_stat'][df1['stat_id']==1].values.tolist()
    atk = df1['base_stat'][df1['stat_id']==2].values.tolist()
    defc = df1['base_stat'][df1['stat_id']==3].values.tolist()
    spatk = df1['base_stat'][df1['stat_id']==4].values.tolist()
    spdef = df1['base_stat'][df1['stat_id']==5].values.tolist()
    spd = df1['base_stat'][df1['stat_id']==6].values.tolist()
    nm = (2*hp[0]+0+(int(0/4)))*50
    hp1 = int(nm/100 + 50 + 10)
    ### This is solving for all of the numerators in the stat calculation equations
    nm1 = (2*atk[0]+0+(int(0/4)))*50
    nm2 = (2*defc[0]+0+(int(0/4)))*50
    nm3 = (2*spatk[0]+0+(int(0/4)))*50
    nm4 = (2*spdef[0]+0+(int(0/4)))*50
    nm5 = (2*spd[0]+0+(int(0/4)))*50
    ### This solves the final values
    atk1 = int((nm1/100+5)*.9)
    def1 = int((nm2/100+5)*.9)
    spatk1 = int((nm3/100+5)*.9)
    spdef1 = int((nm4/100+5)*.9)
    spd1 = int((nm5/100+5)*.9)
    ### This prints out the final stats
    print('HP:',hp1)
    print('ATK:', atk1)
    print('DEF:', def1)
    print('SpATK:', spatk1)
    print('SpDEF:', spdef1)
    print('SPD:', spd1)
def fiftymax(x):

    df1 = statfinder(x)
    iv = 31
    ev = 252
    ### This section locates all the stat values from the csv files
    hp = df1['base_stat'][df1['stat_id']==1].values.tolist()
    atk = df1['base_stat'][df1['stat_id']==2].values.tolist()
    defc = df1['base_stat'][df1['stat_id']==3].values.tolist()
    spatk = df1['base_stat'][df1['stat_id']==4].values.tolist()
    spdef = df1['base_stat'][df1['stat_id']==5].values.tolist()
    spd = df1['base_stat'][df1['stat_id']==6].values.tolist()
    nm = (2*hp[0]+iv+(int(ev/4)))*50
    hp1 = int(nm/100 + 50 + 10)
    ### This is solving for all of the numerators in the stat calculation equations
    nm1 = (2*atk[0]+iv+(int(ev/4)))*50
    nm2 = (2*defc[0]+iv+(int(ev/4)))*50
    nm3 = (2*spatk[0]+iv+(int(ev/4)))*50
    nm4 = (2*spdef[0]+iv+(int(ev/4)))*50
    nm5 = (2*spd[0]+iv+(int(ev/4)))*50
    ### This solves the final values
    atk1 = int((nm1/100+5)*1.1)
    def1 = int((nm2/100+5)*1.1)
    spatk1 = int((nm3/100+5)*1.1)
    spdef1 = int((nm4/100+5)*1.1)
    spd1 = int((nm5/100+5)*1.1)
    ### This prints out the final stats
    print('HP:',hp1)
    print('ATK:', atk1)
    print('DEF:', def1)
    print('SpATK:', spatk1)
    print('SpDEF:', spdef1)
    print('SPD:', spd1)
def stats(x):
    df1 = statfinder(x)
    ### This section locates all the stat values from the csv files
    hp = df1['base_stat'][df1['stat_id']==1].values.tolist()
    atk = df1['base_stat'][df1['stat_id']==2].values.tolist()
    defc = df1['base_stat'][df1['stat_id']==3].values.tolist()
    spatk = df1['base_stat'][df1['stat_id']==4].values.tolist()
    spdef = df1['base_stat'][df1['stat_id']==5].values.tolist()
    spd = df1['base_stat'][df1['stat_id']==6].values.tolist()
    statdict = {'HP': hp, 'ATK': atk, 'DEF': defc, 'SpATK': spatk, 'SpDEF': spdef, 'SPD': spd}
    return statdict
def statgraph(x):
# Need to input two word moves with a "-"
    name = x.title()
    df1 = statfinder(x)
    ### This section locates all the stat values from the csv files
    hp = df1['base_stat'][df1['stat_id']==1].values.tolist()
    atk = df1['base_stat'][df1['stat_id']==2].values.tolist()
    defc = df1['base_stat'][df1['stat_id']==3].values.tolist()
    spatk = df1['base_stat'][df1['stat_id']==4].values.tolist()
    spdef = df1['base_stat'][df1['stat_id']==5].values.tolist()
    spd = df1['base_stat'][df1['stat_id']==6].values.tolist()
    statdict = {'HP': hp, 'ATK': atk, 'DEF': defc, 'SpATK': spatk, 'SpDEF': spdef, 'SPD': spd}
    x = statdict
    labels = list(x.keys())
    values = list(x.values())
    values1 = []
    for i in values:
        it = i[0]
        values1.append(it)
    x = {labels[i]: values1[i] for i in range(len(labels))}
    bar = plt.bar(range(len(x)), values1, align='center')
    bar[0].set_color('r')
    bar[1].set_color('y')
    bar[3].set_color('purple')
    bar[4].set_color('g')
    bar[5].set_color('c')
    plt.xticks(range(len(x)), labels)
    plt.ylim(0,275)
    plt.xlabel('Base Stats')
    plt.ylabel('Value')
    plt.title(name)
    plt.show()
def moveinfo(x):
    y = movesdf[movesdf['identifier'] == x]
    power = y['power'].values.tolist()
    mtype = y['type_id'].values.tolist()
    types = typesdf[typesdf['id'] == mtype[0]]
    t = types['identifier'].values.tolist()
    sp = y['damage_class_id'].values.tolist()
    dmg = {1: 'Status', 2: 'Physical', 3: 'Special'}
    dtype = dmg[sp[0]]
    print(f'Name: {x}\nType: {t[0]}\nPower: {power[0]}\nDamage Type: {dtype}')
def stat(x, level):
    df1 = statfinder(x)
    ### This section locates all the stat values from the csv files
    hp = df1['base_stat'][df1['stat_id']==1].values.tolist()
    atk = df1['base_stat'][df1['stat_id']==2].values.tolist()
    defc = df1['base_stat'][df1['stat_id']==3].values.tolist()
    spatk = df1['base_stat'][df1['stat_id']==4].values.tolist()
    spdef = df1['base_stat'][df1['stat_id']==5].values.tolist()
    spd = df1['base_stat'][df1['stat_id']==6].values.tolist()
    nm = (2*hp[0]+0+(int(0/4)))*level
    hp1 = int(nm/100 + level + 10)
    ### This is solving for all of the numerators in the stat calculation equations
    nm1 = (2*atk[0]+0+(int(0/4)))*level
    nm2 = (2*defc[0]+0+(int(0/4)))*level
    nm3 = (2*spatk[0]+0+(int(0/4)))*level
    nm4 = (2*spdef[0]+0+(int(0/4)))*level
    nm5 = (2*spd[0]+0+(int(0/4)))*level
    ### This solves the final values
    atk1 = int((nm1/100+5)*.9)
    def1 = int((nm2/100+5)*.9)
    spatk1 = int((nm3/100+5)*.9)
    spdef1 = int((nm4/100+5)*.9)
    spd1 = int((nm5/100+5)*.9)
    sts = [hp1, atk1, def1, spatk1, spdef1, spd1]
    return sts
def varstatsHP(species, ev, level, iv=16):
    df1 = statfinder(species)
    ### This section locates all the stat values from the csv files
    hp = df1['base_stat'][df1['stat_id']==1].values.tolist()
    nm = (2*hp[0]+iv+(int(ev/4)))*level
    hp1 = int(nm/100 + level + 10)
    return hp1
def varstats(species, ev, level, stat, iv=16):
    df1 = statfinder(species)
    ### This section locates all the stat values from the csv files
    spd = df1['base_stat'][df1['stat_id']==stat].values.tolist()
    nm5 = (2*spd[0]+iv+(int(ev/4)))*level
    ### This solves the final values
    spd1 = int((nm5/100+5))
    return spd1
def MoveCalc(move):
    blk = ""
    move = move.lower()
    if isinstance(move, (str)) == True:
        for i in range(len(move)):
            if bool(re.search(r"\s", move)) == True:
                if move[i] == ' ':
                    blk = blk + '-'
                else:
                    blk = blk + move[i] 
            else:
                pass
    else:
        raise TypeError('The input must be a string')
    y = movesdf[movesdf['identifier'] == blk]
    power = y['power'].values.tolist()
    mtype = y['type_id'].values.tolist()
    types = typesdf[typesdf['id'] == mtype[0]]
    t = types['identifier'].values.tolist()
    sp = y['damage_class_id'].values.tolist()
    dmg = {1: 'Status', 2: 'Physical', 3: 'Special'}
    dtype = dmg[sp[0]]
    movedct = {'Name': blk, 'Power': power, 'Type': t, 'Damage Type': dtype}
    return movedct
def DMGCalc(stata, statb, pwr, lvl):
    stata = int(stata)
    statb = int(statb)
    pwr = int(pwr)
    lvl = int(lvl)
    num1 = (((2*lvl)/5)+2)
    num1a = ((stata/statb))
    num2 = (num1*pwr*num1a)
    rand = random.uniform(0.85, 1)
    dmg = int(((num2/50)+2)*rand)
    return dmg
#%%

class pkstorage():
    def __init__(self, name, move, level=50):
        self.name = name
        self.level = level
        # Creating a storing a moveset for the pokemon
        if isinstance(move, (list)) == True:
            if len(move) < 5:
                self.move = move
            else:
                raise ValueError('The total moves has to be 4 or less')
        else:
            raise TypeError('move must be a list')
    def moveset(self):
        # This section is going to store the moveset information for the pokemon
        move = self.move
        for i in move:
            i = i.lower()
            blk = ""
            # Checking if there is a space in the string since the function cannot use spaces
            # https://www.geeksforgeeks.org/python-check-for-spaces-in-string/
            # imported re at the top. /s looks for the spaces
            if bool(re.search(r"\s", i)) == True:
                for item in range(len(i)):
                    if i[item] == ' ':
                        blk = blk + '-'
                    else:
                        blk = blk + i[item]
                print('-----------------------------')
                moveinfo(blk)
            else:
                print('-----------------------------')
                moveinfo(i)
    # This gives me the stats of the pokemon at its level with 0 IV's and EV's
    def minstats(self):
        # The three lines below reference the information stored in the class object
        name = self.name
        name1 = name.lower()
        level = self.level
        # Calls the stat functioned defined in the functions portion.
        sts = stat(name1, level)
        print('---------')
        print(f'Level: {level}\n---------\nHP: {sts[0]}\nATK: {sts[1]}\nDEF: {sts[2]}\nSpATK: {sts[3]}\nSpDEF: {sts[4]}\nSPD: {sts[5]}')
    # This gives the pokemon information by calling the function defined in the functions section.
    def info(self):
        name = self.name
        name1 = name.lower()
        self.info = pokemoninfobyname(name1)
    def statistics(self,hp=0,atk=0,deff=0,spatk=0,spdef=0,spd=0):
        ev = [hp, atk, deff, spatk, spdef, spd]
        name = self.name
        name = name.lower()
        lvl = self.level
        for i in ev:
            if isinstance(i,(int)) == False:
                raise TypeError('Inputs for the EVs have to be an integer')
            elif i > 252:
                raise ValueError('Individual EVs cannot be higher than 252')
            elif i < 0:
                raise ValueError('Individual EVs cannot be negative')
            else:
                pass
        if sum(ev) > 510:
            raise ValueError('Total amount of EVs cannot be higher than 510')
        else:
            pass
        #HP 
        hp1 = varstatsHP(species=name, level=lvl, ev=ev[0])
        print('----------------\n',name.title(),'\nLevel:',lvl, '\n----------------')
        print(f'HP: {hp1} / EV: {ev[0]}\n----------------')
        labels = ['ATK', 'DEF', 'SpATK', 'SpDEF', 'SPD']
        ev1 = ev[1:]
        pos = 0
        statlist = [hp1]
        for i in ev1:
            pos1 = pos + 2
            st = varstats(species=name, level=lvl, ev=i, stat=pos1)
            print(f'{labels[pos]}: {st} / EV: {i}')
            pos = pos + 1
            statlist.append(st)
        return statlist
    def sgraph(self):
        name = self.name
        name = name.lower()
        statgraph(name)
    def DamageCalc(self, pkatk, pkdef, move):
        attack_move = MoveCalc(move)
        level = self.level
        if attack_move['Damage Type'] == 'Physical':
            attacking_stat = pkatk[1]
            defending_stat = pkdef[2]
        elif attack_move['Damage Type'] == 'Special':
            attacking_stat = pkatk[3]
            defending_stat = pkdef[4]
        else:
            print('That is a status move')
            exit()
        dmg = attack_move['Power'][0]
        atktype = attack_move['Type']
        damage = DMGCalc(attacking_stat, defending_stat, dmg, level)
        print(damage)
#%%
moveset1 = ['Earthquake', 'Scale Shot', 'Swords Dance', 'Outrage']
moveset2 = ['Behemoth Blade', 'Play Rough', 'Wild Charge', 'Swords Dance']
slot1 = pkstorage(name='Garchomp', move=moveset1, level=100)                  
slot2 = pkstorage(name='Zacian', move=moveset2, level=100)
#%%
'''TESTING CELL'''
stat1 = slot2.statistics(hp=4,atk=252,spd=252)
stat2 = slot1.statistics(atk=252,hp=252,deff=4)
slot2.DamageCalc(pkatk=stat1, pkdef=stat2, move='Behemoth Bash')
#%%
statsdf.info()
species.info()

#%%
def gen_analysis(x):
    print('Generation', x, 'Analysis')
    y = speciesdf.groupby('generation_id')['generation_id'].count()
    print('Number of Pokemon:', y[x])
    
gen_analysis(7)
