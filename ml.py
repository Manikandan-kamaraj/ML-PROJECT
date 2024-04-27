import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
df = pd.read_csv("/kaggle/input/road-accident-data-2020-india/df.csv")
print(df.head())
print(df.info())
print(df.describe().T)
plt.figure(figsize=(10, 6))
sns.histplot(df['Count'], kde=True)
plt.title('Distribution of Accident Count')
plt.xlabel('Accident Count')
plt.ylabel('Frequency')
plt.show()
plt.figure(figsize=(10, 6))
sns.countplot(data=df, x="Cause category")
plt.title("Distribution of Cause Categories")
plt.xlabel("Cause Category")
plt.ylabel("Count")
plt.xticks(rotation=45)
plt.show()
plt.figure(figsize=(10,6))
sns.countplot(data=df, x="Outcome of Incident")
plt.title("Distribution of Outcome of Incident")
plt.xlabel("Outcome of Incident")
plt.ylabel("Count")
plt.xticks(rotation=45)
plt.show()
plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x="Cause category", y="Count")
plt.title("Accident Count by Cause Categories")
plt.xlabel("Cause Category")
plt.ylabel("Accident Count")
plt.xticks(rotation=45)
plt.show()
print("Missing values:\n", df.isnull().sum())
# Total count of accidents, injuries, and deaths
total_accidents = df['Count'].sum()
total_injuries = df.loc[df['Outcome of Incident'] == 'Injured', 'Count'].sum()
total_deaths = df.loc[df['Outcome of Incident'] == 'Deaths', 'Count'].sum()

print("Total accidents:", total_accidents)
print("Total injuries:", total_injuries)
print("Total deaths:", total_deaths)

# Cities with the highest number of incidents
cities_with_highest_incidents = df.groupby('Million Plus Cities')['Count'].sum().nlargest(5)
print("Cities with the highest number of incidents:")
print(cities_with_highest_incidents)
# Most frequent cause categories
most_frequent_cause_categories = df['Cause category'].value_counts().nlargest(5)
print("Most frequent cause categories:")
print(most_frequent_cause_categories)

# Most frequent cause subcategories
most_frequent_cause_subcategories = df['Cause Subcategory'].value_counts().nlargest(5)
print("Most frequent cause subcategories:")
print(most_frequent_cause_subcategories)
heatmap_data = df.groupby(['Million Plus Cities', 'Cause category'])['Count'].sum().unstack().fillna(0)
if heatmap_data.empty:
    print("Insufficient data available to generate the heatmap.")
else:
    plt.figure(figsize=(10, 6))
    sns.heatmap(heatmap_data, cmap='YlOrRd', annot=True, fmt=".0f", linewidths=0.5)
    plt.title('Geographic Distribution of Accidents by Cause Category')
    plt.xlabel('Cause Category')
    plt.ylabel('Cities')
    plt.show()



