# Import our libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
import mlxtend.frequent_patterns
import mlxtend.preprocessing
import pandas_profiling

# import the Cw1 fm radio dataset
data_for_basket = pd.read_csv('C:/Users/msham/PycharmProjects/DataMiningCoursework/CW1_Last_FM.csv')
# Understanding the data
print(data_for_basket.info())
data_profile= data_for_basket.profile_report(title="Coursework1 part2 data statistics report")
data_profile.to_file(output_file="CW1_Data_statistics.html")
print(data_for_basket.shape)
print(data_for_basket.head())
print(data_for_basket.sex.value_counts(normalize=True))
artist_support=data_for_basket.artist.value_counts(normalize=True)
print(artist_support)
artist_support.to_csv('CW1_Secondpart_artist_support.csv', index=False)


# Understanding types of data stored in our dataset
print(data_for_basket.dtypes)

# Findout the how many unique users we have
print("User unique: ", len(data_for_basket.user.unique()))
print("Artist unique: ", len(data_for_basket.artist.unique()))
print("Country unique: ", len(data_for_basket.country.unique()))
# Understand the number of missing values in each column by the number of them and the percentage
print(data_for_basket.isna().sum())
print(data_for_basket.isna().sum() / len(data_for_basket) * 100)

# See if there is any duplicate and deleting them
print(data_for_basket[data_for_basket.duplicated(keep=False)])
data_for_basket.drop_duplicates(inplace=True)

# Change dataset to a table for analysis


basket_all_countries = (data_for_basket.groupby(['user', 'artist'])['user'].sum().unstack()
          .reset_index().fillna(0).set_index('user'))


def encode_units(z):
    if z <= 0:
        return 0
    elif z >0 :
        return 1


basket_sets_all_countries = basket_all_countries.applymap(encode_units)

print(basket_sets_all_countries.head())

# Save the changed dataframe
basket_sets_all_countries.to_csv('CW1_Secondpart_hotencoded.csv', index=False)

frequent_itemsets = apriori(basket_sets_all_countries,min_support=0.03,use_colnames=True)
print(frequent_itemsets)
print(frequent_itemsets.shape)
frequent_itemsets['length']= frequent_itemsets['itemsets'].apply(lambda x:len(x))
print(frequent_itemsets)
print(frequent_itemsets.shape)
frequent_itemsets.to_csv('CW1_Secondpart_frequebtitemsets.csv', index=False)
rule = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.4)
rule.sort_values(by='lift',inplace=True,ascending=False)
print(rule.head())
rule.to_csv('CW1_Secondpart_rule.csv', index=False)
print(rule.head())
print(rule.sort_values('lift',ascending=False))