import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Import data
df = pd.read_csv('medical_examination.csv')

# Add 'overweight' column
bmi = df['weight'] / ((df['height']/100)**2)
df['overweight'] = (bmi > 25).astype(int)


# Normalize data by making 0 always good and 1 always bad. If the value of 'cholesterol' or 'gluc' is 1, make the value 0. If the value is more than 1, make the value 1.
df['cholesterol'] = (df['cholesterol'] != 1).astype(int)
df['gluc'] = (df['gluc'] != 1).astype(int)

print(df.tail())


# Draw Categorical Plot
def draw_cat_plot():
  # Create DataFrame for cat plot using `pd.melt` using just the values from 'cholesterol', 'gluc', 'smoke', 'alco', 'active', and 'overweight'.
  df_cat = pd.melt(df,
                   id_vars=['cardio'],
                   value_vars=[
                     'cholesterol', 'gluc', 'smoke', 'alco', 'active',
                     'overweight'
                   ])

  # Group and reformat the data to split it by 'cardio'. Show the counts of each feature. You will have to rename one of the columns for the catplot to work correctly.
  df_cat = df_cat.groupby(['cardio', 'variable', 'value']) \
               .size() \
               .reset_index(name='total')
  df_cat.head()

  # Draw the catplot with 'sns.catplot()'
  g = sns.catplot(x='variable', y='total', hue='value', col='cardio', data=df_cat, kind='bar')

  # Get the figure for the output
  fig = g.fig

  # Do not modify the next two lines
  fig.savefig('catplot.png')
  return fig


# Draw Heat Map
def draw_heat_map():
  # Clean the data
  # diastolic pressure is higher than systolic (Keep the correct data with (df['ap_lo'] <= df['ap_hi']))
  # height is less than the 2.5th percentile (Keep the correct data with (df['height'] >= df['height'].quantile(0.025)))
  # height is more than the 97.5th percentile
  # weight is less than the 2.5th percentile
  # weight is more than the 97.5th percentile
  df_heat = df[
  (df['ap_lo'] <= df['ap_hi'])
  & (df['height'] >= df['height'].quantile(0.025))
  & (df['height'] <= df['height'].quantile(0.975))
  & (df['weight'] >= df['weight'].quantile(0.025))
  & (df['weight'] <= df['weight'].quantile(0.975))
  ]
  print(df_heat.head())

  # Calculate the correlation matrix
  corr = df_heat.corr()

  # Generate a mask for the upper triangle
  mask = np.triu(corr)

  # Set up the matplotlib figure
  fig, ax = plt.subplots(figsize=(12, 8))

  # Draw the heatmap with 'sns.heatmap()'
  sns.heatmap(corr, annot=True, mask=mask, fmt='.1f', vmin=-0.16, vmax=0.32)

  # Do not modify the next two lines
  fig.savefig('heatmap.png')
  return fig
