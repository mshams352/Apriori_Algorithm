# Apriori Analysis on Last FM dataset
This repository contains Python code for analyzing the Last FM dataset using the Apriori algorithm. The Apriori algorithm is a popular algorithm for mining frequent itemsets and generating association rules from transactional data.

## Prerequisites
To run the code in this repository, you need the following libraries:

- numpy
- pandas
- matplotlib
- mlxtend
- pandas_profiling

You can install these libraries using pip.

## Dataset
The Last FM dataset used in this analysis is available in the file CW1_Last_FM.csv. The dataset contains information about the listening history of Last FM users, including the user ID, artist name, and country.

## Understanding the data
The code in apriori_analysis.py includes various data exploration techniques to understand the data better. These include:

- Checking the data types of each column
- Finding the number of unique users, artists, and countries
- Identifying missing values and their percentage
- Removing duplicate records
- Encoding the dataset for analysis
## Analysis
The code in apriori_analysis.py performs Apriori analysis on the encoded dataset. The minimum support threshold used is 0.03, and the resulting frequent itemsets are saved in the file CW1_Secondpart_frequentitemsets.csv. The code also generates association rules with a minimum confidence threshold of 0.4 and saves them in the file CW1_Secondpart_rule.csv.

## Conclusion
The Apriori algorithm is a powerful tool for finding frequent itemsets and generating association rules from transactional data. This analysis on the Last FM dataset provides insights into the listening behavior of Last FM users, which can be used to make informed decisions about music recommendations and targeted marketing campaigns.
