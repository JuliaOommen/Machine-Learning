# -*- coding: utf-8 -*-
"""
Created on Sun Oct 29 12:09:17 2023

@author: jonah
"""

import os
os.getcwd()
os.chdir("C:/Users/jonah/OneDrive/Documents/Julia/Sem 5/SKproject") 

# To enable plotting graphs in Jupyter notebook
%matplotlib inline

# Importing libraries
import pandas as pd
from sklearn.linear_model import LogisticRegression

# importing ploting libraries
import matplotlib.pyplot as plt   

#importing seaborn for statistical plots
import seaborn as sns

#Let us break the X and y dataframes into training set and test set. For this we will use
#Sklearn package's data splitting function which is based on random function

from sklearn.model_selection import train_test_split

import numpy as np
#import os,sys
from scipy import stats

# calculate accuracy measures and confusion matrix
from sklearn import metrics

from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import confusion_matrix

# Importing necessary libraries
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("Newdataset.csv")
df.isnull().sum()
df.info()

df.columns.value_counts().sum()

#Working with 1st quarter
bin_edges = [0, 3, 6, float("inf")]  # Example bin edges: 0-100, 101-200, 201 and above

# Create labels for the bins
bin_labels = ['Low', 'Average', 'High']

# Cut the column into bins and assign labels
df['RATING_Q1'] = pd.cut(df['RATING_Q1'], bins=bin_edges, labels=bin_labels)
df['RATING_Q2'] = pd.cut(df['RATING_Q2'], bins=bin_edges, labels=bin_labels)
df['RATING_Q3'] = pd.cut(df['RATING_Q3'], bins=bin_edges, labels=bin_labels)
df['RATING_Q4'] = pd.cut(df['RATING_Q4'], bins=bin_edges, labels=bin_labels)

bin_counts = df['RATING_Q2'].value_counts()
print(bin_counts)
plt.bar(bin_counts.index, bin_counts.values)
plt.xlabel('Bins')
plt.ylabel('Count')
plt.title('Count of Values in Each Bin')
plt.show()

grouped_data = df.groupby(['RATING_Q1', 'Y/N']).size().reset_index(name='Count')
grouped_data = df.groupby(['RATING_Q2', 'Y/N']).size().reset_index(name='Count')
grouped_data = df.groupby(['RATING_Q3', 'Y/N']).size().reset_index(name='Count')
grouped_data = df.groupby(['RATING_Q4', 'Y/N']).size().reset_index(name='Count')

# 'grouped_data' now contains the count of occurrences for each bin and 'Another_Column' combination
print(grouped_data)

# Set up the figure and axis
fig, ax = plt.subplots()

# Define unique values for 'RATING_Q1' and 'Y/N'
bins = np.unique(grouped_data['RATING_Q3'])
categories = np.unique(grouped_data['Y/N'])

# Set the width of the bars
bar_width = 0.35

# Create bar positions for each category and bin
x = np.arange(len(bins))
for i, category in enumerate(categories):
    counts = grouped_data[grouped_data['Y/N'] == category]['Count']
    ax.bar(x + i * bar_width, counts, bar_width, label=f'Y/N: {category}')

# Set labels, title, and legend
ax.set_xlabel('RATING_Q3')
ax.set_ylabel('Count')
ax.set_title('Counts of Occurrences by Bins and Y/N')
ax.set_xticks(x + bar_width * (len(categories) - 1) / 2)
ax.set_xticklabels(bins)
ax.legend()

# Show the plot
plt.tight_layout()
plt.show()

#######
#difference between q1 and q4
df['DIFF_Q1_Q4'] = df['RATING_Q4'] - df['RATING_Q1']

# Plotting the histogram based on DIFF_Q1_Q4
plt.hist(df['DIFF_Q1_Q4'], bins=30, color='pink', alpha=0.7)
plt.xlabel('Difference Value')
plt.ylabel('Frequency')
plt.title('Histogram for RATING_Q1 - RATING_Q4 Difference')
plt.show()


plt.hist('AGE','I_LM')
plt.hist2d(df['AGE'], df['I_LM'], bins=(2100, 1000), cmap='Blues')
plt.hexbin(df['AGE'], df['I_LM'], gridsize=50, cmap='Blues')
plt.xlabel('AGE')
plt.ylabel('Income')
plt.title('Hexbin Plot between Column1 and Column2')
plt.colorbar(label='Frequency')
plt.show()

sns.kdeplot(data=df, x='AGE', y='I_LM', cmap='Blues', fill=True)
plt.xlabel('AGE')
plt.ylabel('Income')
plt.title('2D Kernel Density Estimation Plot between Column1 and Column2')
plt.show()

##########################################
df['A_O_DT'] = pd.to_datetime(df['A_O_DT'])

# Define the bin edges (years from 2005 to 2016)
bin_edges = pd.date_range(start='2005-01-01', end='2017-01-01', freq='YS')

# Define the bin labels (years)
bin_labels = [str(year) for year in range(2005, 2017)]

# Cut the date column into bins and assign labels
df['A_O_DT'] = pd.cut(df['A_O_DT'], bins=bin_edges, labels=bin_labels)
bin_counts = df['A_O_DT'].value_counts()
bin_counts 
grouped_data = df.groupby(['A_O_DT', 'Y/N']).size().reset_index(name='Count')
print(grouped_data)

fig, ax = plt.subplots()

# Define unique values for 'RATING_Q1' and 'Y/N'
bins = np.unique(grouped_data['A_O_DT'])
categories = np.unique(grouped_data['Y/N'])

# Set the width of the bars
bar_width = 0.35

# Create bar positions for each category and bin
x = np.arange(len(bins))
for i, category in enumerate(categories):
    counts = grouped_data[grouped_data['Y/N'] == category]['Count']
    ax.bar(x + i * bar_width, counts, bar_width, label=f'Y/N: {category}')

# Set labels, title, and legend
ax.set_xlabel('Years')
ax.set_ylabel('Count')
ax.set_title('Counts of Occurrences by Years and Y/N')
ax.set_xticks(x + bar_width * (len(categories) - 1) / 2)
ax.set_xticklabels(bins)
ax.legend()

################################################################

#################################################################
from scipy.stats import chi2_contingency

# Assuming df is your DataFrame containing the 'Y/N' and 'F_9' columns
# Create a contingency table
contingency_table = pd.crosstab(df['Y/N'], df['F_9'])
contingency_table = pd.crosstab(df['Y/N'], df['RATING_Q4'])

# Perform chi-square test
chi2, p, dof, expected = chi2_contingency(contingency_table)

# Print the results
print("Chi-square statistic:", chi2)
print("P-value:", p)
print("Degrees of freedom:", dof)
print("Expected frequencies table:")
print(expected)

from scipy.stats import pointbiserialr
df['Y/N'] = df['Y/N'].map({'Y': 1, 'N': 0})

# Now, you can convert the column to numeric
df['Y/N'] = pd.to_numeric(df['Y/N'])

df.isnull().sum()
# Assuming 'continuous_var' is the continuous variable and 'binary_var' is the binary categorical variable
correlation_coefficient, p_value = pointbiserialr(df['AGE'], df['Y/N'])
correlation_coefficient, p_value = pointbiserialr(df['EDUCATION_LEVEL'], df['Y/N'])
correlation_coefficient, p_value = pointbiserialr(df['I_LM'], df['Y/N'])
correlation_coefficient, p_value = pointbiserialr(df['A1_LM'], df['Y/N'])
correlation_coefficient, p_value = pointbiserialr(df['A2_LM'], df['Y/N'])
correlation_coefficient, p_value = pointbiserialr(df['A3_LM'], df['Y/N'])

# Print the correlation coefficient and p-value
print("Point-biserial Correlation Coefficient:", correlation_coefficient)
print("P-value:", p_value)


######################################################################

#Randomly undersampling
class_counts = df['Y/N'].value_counts()
class_counts

plt.figure(figsize=(8, 6))  # Optional: set the figure size
class_counts.plot(kind='bar', color='maroon')
plt.xlabel('Class')
plt.ylabel('Count')
plt.title('Class Counts of Y/N Column')
plt.xticks(rotation=0)  # Optional: rotate x-axis labels if needed
plt.show()

minority_class = class_counts.idxmin()
majority_class = class_counts.idxmax()

# Count of samples in minority class
minority_count = class_counts.min()

# Randomly sample the majority class to match the minority class count
majority_samples = df[df['Y/N'] == majority_class].sample(n=minority_count, random_state=42)

# Concatenate minority class and sampled majority class to create a balanced dataset
balanced_df = pd.concat([df[df['Y/N'] == minority_class], majority_samples])

# 'balanced_df' now contains balanced samples of both classes


###################################################################
features = balanced_df.drop(['Y/N', 'A_O_DT','F_9', 'AGE', 'EDUCATION_LEVEL', 'I_LM'],1)  # Add more features as needed
features = balanced_df[['RATING_Q1','RATING_Q2','RATING_Q3','RATING_Q4']]
features = balanced_df[['RATING_Q1','RATING_Q2','RATING_Q3','RATING_Q4','AVG_RATING','DIFF_Q2_Q1','DIFF_Q3_Q2','DIFF_Q4_Q3']]
features=df[['RATING_Q1','RATING_Q2','RATING_Q3','RATING_Q4','AVG_RATING','DIFF_Q2_Q1','DIFF_Q3_Q2','DIFF_Q4_Q3']]
target = df['Y/N']
target = balanced_df['Y/N']

# Evaluate the model
accuracy = accuracy_score(y_test, predictions)
report = classifica
X = features
y = target


#Logistic Regression

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Logistic Regression model
model = LogisticRegression()

# Train the model on the training data
model.fit(X_train, y_train)

# Make predictions on the test data
predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)
report = classification_report(y_test, predictions)

print("Accuracy:", accuracy)
print("Classification Report:")
print(report)
# Calculate the confusion matrix
conf_matrix = confusion_matrix(y_test, predictions)

print("Confusion Matrix:")
print(conf_matrix)

plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='g', cmap='Blues', 
            xticklabels=['Class 0', 'Class 1'], 
            yticklabels=['Class 0', 'Class 1'])

plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix Decision Tree')
plt.show()


# Initialize the Decision Tree classifier
model = DecisionTreeClassifier(random_state=42)

# Train the model using the training data
model.fit(X_train, y_train)

# Make predictions on the test data
predictions = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, predictions)
print(f"Accuracy: {accuracy:.2f}")

# Print classification report for detailed evaluation
print("Classification Report:")
print(classification_report(y_test, predictions))

#Random forest

# Create Random Forest Classifier
random_forest = RandomForestClassifier(random_state=42)

# Train the classifier
random_forest.fit(X_train, y_train)

# Make predictions
predictions = random_forest.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, predictions)
print("Accuracy:", accuracy)
print("Classification Report:")
print(classification_report(y_test, predictions))


#Support Vector Machine (SVM) Classifier:

# Standardize features (SVM is sensitive to feature scales)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create SVM Classifier
svm_classifier = SVC(kernel='linear', random_state=42)

# Train the classifier
svm_classifier.fit(X_train_scaled, y_train)

# Make predictions
predictions = svm_classifier.predict(X_test_scaled)

# Evaluate the model
accuracy = accuracy_score(y_test, predictions)
print("Accuracy:", accuracy)
print("Classification Report:")
print(classification_report(y_test, predictions))

#################################################################
#Feature engineering

features = balanced_df[['AVG_RATING','DIFF_Q2_Q1','DIFF_Q3_Q2','DIFF_Q4_Q3']]
df['AVG_RATING'] = df[['RATING_Q1', 'RATING_Q2', 'RATING_Q3', 'RATING_Q4']].mean(axis=1)
import matplotlib.pyplot as plt
plt.hist(df['AVG_RATING'], bins=30, color='skyblue', alpha=0.7)
plt.xlabel('Average Rating Across Quarters')
plt.ylabel('Frequency')
plt.title('Distribution of Average Ratings')
plt.show()


df['DIFF_Q2_Q1'] = df['RATING_Q2'] - df['RATING_Q1']
df['DIFF_Q3_Q2'] = df['RATING_Q3'] - df['RATING_Q2']
df['DIFF_Q4_Q3'] = df['RATING_Q4'] - df['RATING_Q3']

X = df[['DIFF_Q2_Q1', 'DIFF_Q3_Q2', 'DIFF_Q4_Q3']]  # Features
y = df['Y/N']  # Target variable
# Split the data, train the model, and evaluate its performance

correlation_matrix = df[['DIFF_Q2_Q1', 'DIFF_Q3_Q2', 'DIFF_Q4_Q3']].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap for Difference Between Quarters')
plt.show()

sns.pairplot(df[['DIFF_Q2_Q1', 'DIFF_Q3_Q2', 'DIFF_Q4_Q3']])
plt.suptitle('Pairplot for Difference Between Quarters', y=1.02)
plt.show()

plt.boxplot([balanced_df['DIFF_Q2_Q1'], balanced_df['DIFF_Q3_Q2'], balanced_df['DIFF_Q4_Q3']],
            labels=['Q2-Q1 Difference', 'Q3-Q2 Difference', 'Q4-Q3 Difference'])
plt.ylabel('Difference Value')
plt.title('Box Plot for Difference Between Quarters')
plt.show()

plt.hist(df['DIFF_Q2_Q1'], bins=30, color='skyblue', alpha=0.7, label='Q2-Q1 Difference')
plt.xlabel('Difference Value')
plt.ylabel('Frequency')
plt.title('Histogram for Q2-Q1 Difference')
plt.legend()
plt.show()
plt.hist(df['DIFF_Q3_Q2'], bins=30, color='skyblue', alpha=0.7, label='Q2-Q1 Difference')
plt.xlabel('Difference Value')
plt.ylabel('Frequency')
plt.title('Histogram for Q3-Q2 Difference')
plt.legend()
plt.show()
plt.hist(df['DIFF_Q4_Q3'], bins=30, color='skyblue', alpha=0.7, label='Q2-Q1 Difference')
plt.xlabel('Difference Value')
plt.ylabel('Frequency')
plt.title('Histogram for Q4-Q3 Difference')
plt.legend()
plt.show()

plt.hist(df['DIFF_Q2_Q1'], bins=30, color='skyblue', alpha=0.7, label='Q2-Q1 Difference')
plt.hist(df['DIFF_Q4_Q3'], bins=30, color='orange', alpha=0.7, label='Q4-Q3 Difference')

plt.xlabel('Difference Value')
plt.ylabel('Frequency')
plt.title('Histogram Comparison: Q2-Q1 vs Q4-Q3 Difference')
plt.legend()
plt.show()

#################################################################
balanced_df['I_LM']
#income
highy=(balanced_df.loc[(balanced_df['I_LM']>900000) & (balanced_df['Y/N']=='Y')]).value_counts().sum()
highn=(balanced_df.loc[(balanced_df['I_LM']>900000) & (balanced_df['Y/N']=='N')]).value_counts().sum()
lowy=(balanced_df.loc[(balanced_df['I_LM']<200000) & (balanced_df['Y/N']=='Y')]).value_counts().sum()
lown=(balanced_df.loc[(balanced_df['I_LM']<200000) & (balanced_df['Y/N']=='N')]).value_counts().sum()
labels=['Yes','No']
income=[highy, highn, lowy, lown]
plt.bar(label, income)

in_yes = [lowy, highy]
in_no = [lown, highn]
labels = ["Low Income", "High Income"]

x = np.arange(len(labels))  # the label locations
width = 0.35  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, in_yes, width, label='Yes')
rects2 = ax.bar(x + width/2, in_no, width, label='No')
ax.set_xlabel('Income Levels')
ax.set_ylabel('Counts')
ax.set_title('Income Level Counts by Yes/No')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

#Education

edlowy=(balanced_df.loc[(balanced_df['EDUCATION_LEVEL']<=50)&(balanced_df['Y/N']=='Y')]).value_counts().sum()
edlown = (balanced_df.loc[(balanced_df['EDUCATION_LEVEL']<= 50) & (balanced_df['Y/N'] == 'N')]).value_counts().sum()
edhighy=(balanced_df.loc[(balanced_df['EDUCATION_LEVEL']>55)&(balanced_df['Y/N']=='Y')]).value_counts().sum()
edhighn = (balanced_df.loc[(balanced_df['EDUCATION_LEVEL']> 55) & (balanced_df['Y/N'] == 'N')]).value_counts().sum()
ed=[edlowy,edlown,edhighy,edhighn]
labels=["Less educated (Y)", "Less educated (N)", "Highly educated (Y)", 'Highly educated (N)']
plt.scatter(label,ed)


# Values and labels
ed = [edlowy, edlown, edhighy, edhighn]
labels = ["Less educated (Y)", "Less educated (N)", "Highly educated (Y)", 'Highly educated (N)']

# Plotting the bar chart
plt.bar(labels, ed, color=['green', 'red', 'green', 'red'])
plt.xlabel('Education Level and Y/N')
plt.ylabel('Counts')
plt.title('Education Level Counts based on Y/N')
plt.xticks(rotation=45)  # Rotate x-axis labels for better visibility
plt.show()