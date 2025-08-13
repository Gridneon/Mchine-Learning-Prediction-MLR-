import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns

# identify the folder that contain the data 

import os
os.getcwd()  # get the current working directory
os.listdir('c:\\Users\\HP\\Downloads\\datascience.py')  # identify files in the directory 
os.chdir('c:\\Users\\HP\\Downloads\\datascience.py\\data_files')  # change directory 
os.listdir('c:\\Users\\HP\\Downloads\\datascience.py\\data_files')

link_to_Iris_data = "c:\\Users\\HP\\Downloads\\datascience.py\\data_files\\irisdata.csv"


Iris_data = pd.read_csv(link_to_Iris_data)

# step 1 : Data Understanding.getting to know the data 

# head  and tail of the data
Iris_data.head(6)  # head 

Iris_data.tail(6)  #tail 

#shape of the data 
Iris_data.shape

# dtypes 
Iris_data.dtypes 

#info of the data 
Iris_data.info()

# describe
Iris_data.describe()

# Step 2 : Data Preparation. cleaning the data 
# Dropping Irrelevant columns and rows

Iris_data.columns  # all column are relevant 
 
# Identifying duplicates columns and missing values
Iris_data.isna().sum()
Iris_data.duplicated()

Iris_data.loc[Iris_data.duplicated()].head(4)
# or 
Iris_data.loc[Iris_data.duplicated()]

Iris_data.query('sepal_width == 2.7 & sepal_length == 5.8 & petal_length == 5.1')

#dropping duplicates 

Iris_data1 = Iris_data.loc[~Iris_data.duplicated(subset =['sepal_length','sepal_width','petal_length','petal_width','flower_type'])]
Iris_data1.loc[Iris_data.duplicated()]
Iris_data1.head(6)
Iris_data1.shape

# Renaming Columns and Feature Creation 
Iris_data = Iris_data.rename(columns = {'species' : 'flower_type'})


# Feature Understanding 
#histogram for each and every numeric variable 

ax = Iris_data1['flower_type'].value_counts() \
    .head(6) \
    .plot(kind='bar',title=" flower type")
    
ax.set_xlabel(" species")
ax.set_ylabel("counts of each species")

Iris_data1.columns
Iris_data1['sepal_length'].plot(kind='hist',title= "Sepal length distribution",)
Iris_data1['sepal_width'].plot(kind='hist',title= "Sepal width distribution" )
Iris_data1['petal_length'].plot(kind='hist',title= "Petal length distribution" )
Iris_data1['petal_width'].plot(kind='hist',title= "Petal width  distribution" )

iris_setosa = Iris_data1.query('flower_type == "Iris-setosa" ')
iris_setosa.describe()

iris_versicolor = Iris_data1.query('flower_type== "Iris-versicolor"')
iris_versicolor.describe()

iris_virginica = Iris_data1.query('flower_type == "Iris-virginica"')
iris_virginica.describe()

sns.barplot(x= 'flower_type', y= "sepal_length",data = Iris_data1)
sns.boxplot(x= 'flower_type',y= "sepal_length",data = Iris_data1)

sns.barplot(x= 'flower_type', y= "sepal_width",data = Iris_data1)
sns.boxplot(x= 'flower_type',y= "sepal_width",data = Iris_data1)


# Feature Relationship 

Iris_data1.plot(kind= 'scatter',
                               x="sepal_length",y="sepal_width")

sns.scatterplot(x='sepal_length',y="sepal_width",hue='flower_type',data = Iris_data1,style='flower_type')

sns.relplot(x='sepal_length',y="sepal_width",hue='flower_type',col='flower_type',data = Iris_data1,style='flower_type')

sns.pairplot(Iris_data1,vars= ['sepal_length', 'sepal_width', 'petal_length', 'petal_width'],hue="flower_type")

# get the plot correlation matrix of the variables 
corr_df = Iris_data1[['sepal_length', 'sepal_width', 
                       'petal_length', 'petal_width']].dropna().corr()


sns.heatmap(corr_df,annot=True)