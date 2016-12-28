
# coding: utf-8

# Below is the code for my best submission in the Santander Product Recommendation Kaggle Competition (a link to the competition: https://www.kaggle.com/c/santander-product-recommendation). This competition interested me because I had never created a recommendation model before, and it seemed like a good challenge. The goal of this competition is to recommend new products to customers.
# 
# The section of the analysis are:
# -Inspecting the data
# -Cleaning a sample of the data
# -Feature Engineering a sample of the data
# -Cleaning and Feature Engineering the data to be used to train the model and make predictions with
# -Building the Model
# -A summary of my findings

# In[1]:

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from datetime import datetime
from sklearn.metrics import accuracy_score
import xgboost as xgb


# Load the datasets

# In[2]:

train = pd.read_csv("/Users/Dave/Desktop/Programming/Personal Projects/Santander-Kaggle/train.csv")
test = pd.read_csv("/Users/Dave/Desktop/Programming/Personal Projects/Santander-Kaggle/test.csv")


# Convert all the dates to datetime

# In[3]:

train.fecha_dato = pd.to_datetime(train.fecha_dato, format="%Y-%m-%d")
train.fecha_alta = pd.to_datetime(train.fecha_alta, format="%Y-%m-%d")
test.fecha_dato = pd.to_datetime(test.fecha_dato, format="%m/%d/%y")
test.fecha_alta = pd.to_datetime(test.fecha_alta, format="%m/%d/%y")


# # Inspect the Data

# Let's take a preview at the data we are working with

# In[4]:

pd.set_option('display.max_columns', 50)
train.head()


# In[5]:

train.describe()


# We can already see that we have some missing values.

# In[6]:

train.describe(include = ['object'])


# In[7]:

test.head()


# In[8]:

print train.shape
print test.shape


# In[9]:

train.isnull().sum()


# There are numerous features with the same number of missing values, I expect those relate to the same rows. ult_fec_cli_1t and conyuemp are missing most of their data, hopefully we can extract some use from those features.

# Combine the train and test sets to explore and clean the data.

# In[4]:

df = pd.concat([train,test], axis = 0, ignore_index = True)


# Let's drop all of the rows that we think have numerous missing features.

# In[5]:

badRows = train[train.ind_empleado.isnull()].index


# In[6]:

df = df.drop(badRows, axis = 0)


# In[99]:

df.isnull().sum()


# Although we have many new missing values bacause the test dataset does not include the target features, we have removed many of the nulls values that we had before.

# To explore and clean the data quicker, let's take a sample of the dataset.

# In[7]:

smallDF = df.sample(frac = 0.15, random_state = 2)


# In[8]:

len(smallDF)


# Let's double check to make sure everything is alright before we start cleaning and exploring the data.

# In[174]:

smallDF.isnull().sum()


# Everything looks good, let the cleaning begin!

# # Clean the Data

# In[17]:

smallDF.canal_entrada.value_counts()


# I think it would be too difficult to make a fair guess, so let's just set the missing values to Unknown.

# In[9]:

smallDF.canal_entrada = smallDF.canal_entrada.fillna("Unknown")


# Since cod_prov is the province's code, we can drop this variable because we already have nomprov, the names of the provinces, which we can change to a categorical feature. You might be thinking, why not just keep cod_prov and save a step, but I think it would be easier to ensure that the data is correct and easier to feature engineer if we use the provinces' names instead.

# In[10]:

smallDF = smallDF.drop("cod_prov", 1)


# In[20]:

smallDF.conyuemp.value_counts()


# To avoid making any grand assumptions, I will set the null values to Unknown. This avoids greatly altering the ratio of 'N' to 'S'.

# In[11]:

smallDF.conyuemp = smallDF.conyuemp.fillna("Unknown")


# In[12]:

smallDF.indrel_1mes.value_counts()


# Since this feature describes types of customer, it would make more sense for it to be a categorical feature. Let's change the numeric features to string, then use one-hot-encoding later.

# In[13]:

smallDF.loc[smallDF.indrel_1mes == '1', 'indrel_1mes'] = 'Primary'
smallDF.loc[smallDF.indrel_1mes == '1.0', 'indrel_1mes'] = 'Primary'
smallDF.loc[smallDF.indrel_1mes == 1, 'indrel_1mes'] = 'Primary'
smallDF.loc[smallDF.indrel_1mes == 1.0, 'indrel_1mes'] = 'Primary'

smallDF.loc[smallDF.indrel_1mes == '2', 'indrel_1mes'] = 'Co-owner'
smallDF.loc[smallDF.indrel_1mes == '2.0', 'indrel_1mes'] = 'Co-owner'
smallDF.loc[smallDF.indrel_1mes == 2, 'indrel_1mes'] = 'Co-owner'
smallDF.loc[smallDF.indrel_1mes == 2.0, 'indrel_1mes'] = 'Co-owner'

smallDF.loc[smallDF.indrel_1mes == '3', 'indrel_1mes'] = 'Former Primary'
smallDF.loc[smallDF.indrel_1mes == '3.0', 'indrel_1mes'] = 'Former Primary'
smallDF.loc[smallDF.indrel_1mes == 3, 'indrel_1mes'] = 'Former Primary'
smallDF.loc[smallDF.indrel_1mes == 3.0, 'indrel_1mes'] = 'Former Primary'

smallDF.loc[smallDF.indrel_1mes == '4', 'indrel_1mes'] = 'Former Co-owner'
smallDF.loc[smallDF.indrel_1mes == '4.0', 'indrel_1mes'] = 'Former Co-owner'
smallDF.loc[smallDF.indrel_1mes == 4, 'indrel_1mes'] = 'Former Co-owner'
smallDF.loc[smallDF.indrel_1mes == 4.0, 'indrel_1mes'] = 'Former Co-owner'


# Let's see if everything worked out okay.

# In[172]:

smallDF.indrel_1mes.value_counts()


# Good, now let's set the missing values to the most common, Primary.

# In[14]:

smallDF.indrel_1mes = smallDF.indrel_1mes.fillna('Primary')


# In[90]:

smallDF.nomprov.value_counts()


# Let's change all the null values to the most common, MADRID.

# In[15]:

smallDF.nomprov = smallDF.nomprov.fillna("MADRID")


# In[136]:

smallDF.renta.value_counts()


# Maybe we can fill in the null values with the median renta for each province.

# In[16]:

smallDF.loc[smallDF.renta == '         NA',"renta"] = 0
smallDF.renta = smallDF.renta.astype(float)


# In[156]:

smallDF.renta.isnull().sum()


# In[17]:

smallDF.loc[smallDF.renta == 0, 'renta'] = smallDF[smallDF.renta > 0].groupby('nomprov').renta.transform('median')
smallDF.loc[smallDF.renta.isnull(), "renta"] = smallDF.groupby('nomprov').renta.transform('median')


# In[165]:

smallDF.renta.isnull().sum()


# Yup, that got everything cleared up.

# In[175]:

smallDF.segmento.value_counts()


# Rather than setting all the null values to the most common, let's see if there's a relationship between renta and segmento that we could use.

# In[176]:

smallDF.renta.groupby([smallDF.segmento]).median()


# Good, there is! Let's use these values to help assign new values to the null values.

# In[18]:

smallDF.segmento = smallDF[smallDF.renta <= 96000].segmento.fillna("03 - UNIVERSITARIO")
smallDF.segmento = smallDF[smallDF.renta <= 119500].segmento.fillna("02 - PARTICULARES")
smallDF.segmento = smallDF.segmento.fillna("01 - TOP")


# In[178]:

smallDF.sexo.value_counts()


# There are not many missing values in sexo, so we can just set them to the majority, V.

# In[19]:

smallDF.sexo = smallDF.sexo.fillna("V")


# In[181]:

smallDF.tiprel_1mes.value_counts()


# Perhaps there is also a relationship between renta and tiprel_1mes to help us better assign values to the null values.

# In[185]:

smallDF.renta.groupby(smallDF.tiprel_1mes).median()


# In[191]:

smallDF[smallDF.renta >= 99500].tiprel_1mes.value_counts()


# I am not too convinced that there is a strong enough of a relationship between renta and tiprel_1mes to be useful. To keep things simple, I going to assign the null values to the most popular, I.

# In[20]:

smallDF.tiprel_1mes = smallDF.tiprel_1mes.fillna('I')


# In[193]:

smallDF.ult_fec_cli_1t.value_counts()


# Given the number of missing values, I am not going to try and find values for them, but I am going to keep this feature, which will be used in the feature engineering section.

# In[194]:

smallDF.ind_nomina_ult1.value_counts()


# Assign nulls to the most popular, 0.0

# In[21]:

smallDF.ind_nomina_ult1 = smallDF.ind_nomina_ult1.fillna(0.0)


# In[196]:

smallDF.ind_nom_pens_ult1.value_counts()


# Assign nulls to the most popular, 0.0

# In[22]:

smallDF.ind_nom_pens_ult1 = smallDF.ind_nom_pens_ult1.fillna(0.0)


# In[202]:

smallDF.antiguedad.value_counts()


# The values equalling -999999 don't look right. Let's substitute them for the median value.

# In[23]:

smallDF.loc[smallDF.antiguedad == -999999, 'antiguedad'] = smallDF[smallDF.antiguedad >= 0].antiguedad.median()


# In[226]:

smallDF.indrel.value_counts()


# Change 99.0 to 0.0 so that it is already scaled for gradient descent. 

# In[33]:

smallDF.loc[smallDF.indrel == 99.0, "indrel"] = 0.0


# Let's see if we have everything clean.

# In[199]:

smallDF.isnull().sum()


# Yes we do! On to the feature engineering stage!

# # Feature Engineering

# Let's start off by taking a look at ages. 

# In[203]:

smallDF.age.value_counts().sort_index()


# In[24]:

smallDF.age = smallDF.age.astype(int)


# In[207]:

plt.hist(smallDF.age, bins = 50)
plt.show()


# Some of these values are probably incorrect. I doubt most of the oldest people in the world are all banking with Santander. There are also quite a few, very young people, i.e. younger than 10. Because there is quite a jump in frequency between 19 and 20, I'm going to group everyone younger than 20 together. I'll also group people older than 90 together.

# In[25]:

smallDF.loc[smallDF.age < 20,"age"] = 19
smallDF.loc[smallDF.age > 90,"age"] = 91


# Group people in sets of 10 years. Subtract 1 so that the values start at 0.

# In[26]:

smallDF['ageGroup'] = (smallDF.age // 10) - 1


# In[213]:

plt.hist(smallDF.ageGroup, bins = 9)
plt.show()


# In[214]:

smallDF.ageGroup.value_counts().sort_index()


# The top two customer age groups are 20s and 40s. It will be interesting to see if there is any difference between these two groups.

# If a customer lives in Spain, for simplicity, they are considered Spanish.

# In[27]:

smallDF['isSpanish'] = smallDF.pais_residencia.map(lambda x: 1 if x == "ES" else 0)


# In[216]:

smallDF.isSpanish.value_counts()


# Most customers live in Spain, over 99%.

# If a customer lives in a Barcelona or Madrid, they are considered from a major city.

# In[28]:

smallDF['majorCity'] = smallDF.nomprov.map(lambda x: 1 if x == "MADRID" or x == "BARCELONA" else 0)


# In[218]:

smallDF.majorCity.value_counts()


# The majority of customers live in Barcelona or Madrid, about 58%.

# Extract the year and month from the date features, and set the lowest value to 0.

# In[29]:

smallDF['fecha_alta_year'] = pd.DatetimeIndex(smallDF.fecha_alta).year - 1995
smallDF['fecha_dato_year'] = pd.DatetimeIndex(smallDF.fecha_dato).year - 2015


# In[30]:

smallDF['fecha_alta_month'] = pd.DatetimeIndex(smallDF.fecha_alta).month - 1
smallDF['fecha_dato_month'] = pd.DatetimeIndex(smallDF.fecha_dato).month - 1


# In[31]:

smallDF.antiguedad = smallDF.antiguedad.astype(int)


# In[223]:

plt.hist(smallDF.antiguedad)
plt.xlabel("antiguedad (months)")
plt.show()


# In[224]:

print min(smallDF.antiguedad)
print max(smallDF.antiguedad)


# Group antiguedad by years.

# In[32]:

smallDF['antiguedad_years'] = smallDF.antiguedad // 12


# In[228]:

smallDF.ult_fec_cli_1t.value_counts()


# Given the number of missing values, I'm going to create a new feature 'HAS_ult_fec_cli_1t' based on if we have a value for this feature or not.

# In[34]:

smallDF['HAS_ult_fec_cli_1t'] = smallDF.ult_fec_cli_1t.map(lambda x: 1 if x > 0 else 0)


# In[230]:

smallDF.HAS_ult_fec_cli_1t.value_counts()


# This looks good. Now we can drop ult_fec_cli_1t because we won't need it anymore.

# In[35]:

smallDF = smallDF.drop('ult_fec_cli_1t', 1)


# In[232]:

smallDF.renta.describe()


# Group customers by renta in $50,000 increments.

# In[36]:

smallDF['rentaGroup'] = smallDF.renta // 50000


# Since renta values become less dense after 1,000,000, we will group people with values between 1M and > 10M together, as well as those with values greater than or equal to $10M.

# In[37]:

smallDF.loc[smallDF.renta >= 1000000, "rentaGroup"] = 20
smallDF.loc[smallDF.renta >= 10000000, "rentaGroup"] = 21


# In[235]:

plt.hist(smallDF.rentaGroup, bins = 21)
plt.show()


# In[236]:

smallDF.rentaGroup.value_counts().sort_index()


# In[237]:

list(smallDF.columns.values)


# Okay, now that we have everything cleaned and new features have been created. We will repeat this process on the data that we want to model. 
# You might be thinking, why not just clean and create the features on the data you want to model. It took a few iterations to choose the data I wanted to model and seperating these two steps simplified things. 

# # Transform the Training Data

# The data we will be modeling with are the rows in the dataset where a customer has added a new product since the previous month. For example, if the customer did not have a credit card in March 2015, but received one in April 2015, we will use the customer's April 2015 data as the training features, and the credit card (ind_tjcr_fin_ult1) as the target feature.

# months is all of the months in the dataset.

# In[38]:

months = train.fecha_dato.unique()


# In[39]:

train_final = pd.DataFrame()


# In[40]:

#Start with the second month because we need a previous month to compare data with.
i = 1
while i < len(months):
    #Subset all of the data of the new month, which will be compared to the previous month.
    train_new_month = train[train.fecha_dato == months[i]]
    train_previous_month = train[train.fecha_dato == months[i-1]]
    
    print("Original length of train1: ", len(train_new_month))
    print("Original length of train2: ", len(train_previous_month))
    print
    
    #Only select the customers who have data in each month.
    train_new_month = train_new_month.loc[train_new_month['ncodpers'].isin(train_previous_month.ncodpers)]
    train_previous_month = train_previous_month.loc[train_previous_month['ncodpers'].isin(train_new_month.ncodpers)]
    
    print("New length of train_new_month: ", len(train_new_month))
    print("New length of train_previous_month: ", len(train_previous_month))
    print
    
    #Sort by ncodpers (Customer code) to allow for easy subtraction between dataframes later.
    train_new_month.sort_values(by = 'ncodpers', inplace = True)
    train_previous_month.sort_values(by = 'ncodpers', inplace = True)
    
    #These are all of the target features.
    target_cols_all = ['ind_ahor_fin_ult1','ind_aval_fin_ult1','ind_cco_fin_ult1','ind_cder_fin_ult1',
                'ind_cno_fin_ult1','ind_ctju_fin_ult1','ind_ctma_fin_ult1','ind_ctop_fin_ult1',
                'ind_ctpp_fin_ult1','ind_deco_fin_ult1','ind_dela_fin_ult1','ind_deme_fin_ult1',
                'ind_ecue_fin_ult1','ind_fond_fin_ult1','ind_hip_fin_ult1','ind_nom_pens_ult1',
                'ind_nomina_ult1','ind_plan_fin_ult1','ind_pres_fin_ult1','ind_reca_fin_ult1',
                'ind_recibo_ult1','ind_tjcr_fin_ult1','ind_valo_fin_ult1','ind_viv_fin_ult1']
    
    #Select only the target columns.
    train_new_month_targets = train_new_month[target_cols_all]
    #Add ncodpers to the dataframe.
    train_new_month_targets['ncodpers'] = train_new_month.ncodpers
    #Remove the index.
    train_new_month_targets.reset_index(drop = True, inplace = True)

    #Select only the target columns.
    train_previous_month_targets = train_previous_month[target_cols_all]
    #Add ncodpers to the dataframe.
    train_previous_month_targets['ncodpers'] = train_previous_month.ncodpers
    #Set ncodpers' values to 0, so that there is no effect to this feature when this dataframe is 
    #subtracted from train_new_month_targets.
    train_previous_month_targets.ncodpers = 0
    #Remove the index.
    train_previous_month_targets.reset_index(drop = True, inplace = True)
    
    #Subtract the previous month from the current to find which new products the customers have.
    train_new_products = train_new_month_targets.subtract(train_previous_month_targets)
    #Values will be negative if the customer no longer has a product that they once did. 
    #Set these negative values to 0.
    train_new_products[train_new_products < 0] = 0
    print("Quantity of features to use:")
    #Sum columns to learn about the quantity of the types of new products.
    print train_new_products.sum(axis = 0)
    
    train_new_products = train_new_products.fillna(0)
    
    #Merge the target features with the data we will use to train the model.
    train_new_products = train_new_products.merge(train_new_month.ix[:,0:24], on = 'ncodpers')
    
    #Add each month's data to the same dataframe.
    train_final = pd.concat([train_final,train_new_products], axis = 0)
    
    print("Length of new dataframe:", len(train_final))
    print
    percent_finished = float(i/len(months))
    print("Percent finished:", percent_finished)
    
    i += 1


# In[241]:

train_final.head()


# Only select the rows in the dataframe where there is a new product, i.e. where at least one target feature has a value of 1.

# In[41]:

train_final = train_final.loc[(train_final.ix[:,0:24] != 0).any(axis=1)]


# In[42]:

len(train_final)


# We need the data from May 2016 because we are only interested in building a model about which new products customers will have in June 2016. Therefore, we need to compare the model's prediction of reccommended products, versus the products the customer already has. 

# In[43]:

final_month = train[train.fecha_dato == '2016-05-28']


# In[44]:

len(final_month)


# Join the training and testing data to reduce repetitive code.

# In[45]:

df = pd.concat([train_final,test], axis = 0, ignore_index = True)


# In[46]:

df.isnull().sum()


# In[47]:

#Clean the data - we will follow the same steps as the 'smallDF'.
print("Step 1/13")
badRows = df[df.ind_empleado.isnull()].index

print("Step 2/13")
df = df.drop(badRows, axis = 0)

print("Step 3/13")
df.canal_entrada = df.canal_entrada.fillna("Unknown")

print("Step 4/13")
df = df.drop("cod_prov", 1)

print("Step 5/13")
df.conyuemp = df.conyuemp.fillna("Unknown")

print("Step 6/13")
df.loc[df.indrel_1mes == '1', 'indrel_1mes'] = 'Primary'
df.loc[df.indrel_1mes == '1.0', 'indrel_1mes'] = 'Primary'
df.loc[df.indrel_1mes == 1, 'indrel_1mes'] = 'Primary'
df.loc[df.indrel_1mes == 1.0, 'indrel_1mes'] = 'Primary'
df.loc[df.indrel_1mes == '2', 'indrel_1mes'] = 'Co-owner'
df.loc[df.indrel_1mes == '2.0', 'indrel_1mes'] = 'Co-owner'
df.loc[df.indrel_1mes == 2, 'indrel_1mes'] = 'Co-owner'
df.loc[df.indrel_1mes == 2.0, 'indrel_1mes'] = 'Co-owner'
df.loc[df.indrel_1mes == '3', 'indrel_1mes'] = 'Former Primary'
df.loc[df.indrel_1mes == '3.0', 'indrel_1mes'] = 'Former Primary'
df.loc[df.indrel_1mes == 3, 'indrel_1mes'] = 'Former Primary'
df.loc[df.indrel_1mes == 3.0, 'indrel_1mes'] = 'Former Primary'
df.loc[df.indrel_1mes == '4', 'indrel_1mes'] = 'Former Co-owner'
df.loc[df.indrel_1mes == '4.0', 'indrel_1mes'] = 'Former Co-owner'
df.loc[df.indrel_1mes == 4, 'indrel_1mes'] = 'Former Co-owner'
df.loc[df.indrel_1mes == 4.0, 'indrel_1mes'] = 'Former Co-owner'

df.indrel_1mes = df.indrel_1mes.fillna('Primary')

print("Step 7/13")
df.nomprov = df.nomprov.fillna("MADRID")

print("Step 8/13")
df.loc[df.renta == '         NA',"renta"] = 0
df.renta = df.renta.astype(float)
df.loc[df.renta == 0, 'renta'] = df[df.renta > 0].groupby('nomprov').renta.transform('median')
df.loc[df.renta.isnull(), "renta"] = df.groupby('nomprov').renta.transform('median')

print("Step 9/13")
df.segmento = df[df.renta <= 98000].segmento.fillna("03 - UNIVERSITARIO")
df.segmento = df[df.renta <= 125500].segmento.fillna("02 - PARTICULARES")
df.segmento = df.segmento.fillna("01 - TOP")

print("Step 10/13")
df.sexo = df.sexo.fillna("V")

print("Step 11/13")
df.tiprel_1mes = df.tiprel_1mes.fillna('I')

print("Step 12/13")
df.ind_nomina_ult1 = df.ind_nomina_ult1.fillna(0.0)
df.ind_nom_pens_ult1 = df.ind_nom_pens_ult1.fillna(0.0)

print("Step 13/13")
df.loc[df.antiguedad == -999999, 'antiguedad'] = df[df.antiguedad >= 0].antiguedad.median()


# In[48]:

#Feature Engineering - follow the same steps as 'smallDF'.
print("Step 1/10")
df.age = df.age.astype(int)
df.loc[df.age < 20,"age"] = 19
df.loc[df.age > 90,"age"] = 91

print("Step 2/10")
df['ageGroup'] = (df.age // 10) - 1

print("Step 3/10")
df['isSpanish'] = df.pais_residencia.map(lambda x: 1 if x == "ES" else 0)

print("Step 4/10")
df['majorCity'] = df.nomprov.map(lambda x: 1 if x == "MADRID" or x == "BARCELONA" else 0)

print("Step 5/10")
df['fecha_alta_year'] = pd.DatetimeIndex(df.fecha_alta).year - 1995
df['fecha_dato_year'] = pd.DatetimeIndex(df.fecha_dato).year - 2015
df['fecha_alta_month'] = pd.DatetimeIndex(df.fecha_alta).month - 1
df['fecha_dato_month'] = pd.DatetimeIndex(df.fecha_dato).month - 1

print("Step 6/10")
df.antiguedad = df.antiguedad.astype(int)
df['antiguedad_years'] = df.antiguedad // 12

print("Step 7/10")
df.loc[df.indrel == 99.0, "indrel"] = 0.0

print("Step 8/10")
df['HAS_ult_fec_cli_1t'] = df.ult_fec_cli_1t.map(lambda x: 1 if x > 0 else 0)

print("Step 9/10")
df = df.drop('ult_fec_cli_1t', 1)

print("Step 10/10")
df['rentaGroup'] = df.renta.astype(float) // 50000
df.loc[df.renta >= 1000000, "rentaGroup"] = 20
df.loc[df.renta >= 10000000, "rentaGroup"] = 21


# Since we removed the 'badRows' from the data, we need to find the new length of the dataframe.

# In[49]:

train_final_length = len(train_final) - len(badRows)


# In[50]:

train_final, test = df[:train_final_length], df[train_final_length:] 


# In[51]:

print len(train_final)
print len(test)


# The length looks good there.

# Seperate the training columns from the dataframe as different transformations will be performed on these features, compared to the target columns.

# In[52]:

train_final_training_cols = train_final
train_final_training_cols = train_final_training_cols.drop(target_cols_all, axis=1)
test = test.drop(target_cols_all, axis=1)


# In[53]:

df = pd.concat([train_final_training_cols, test], axis = 0)


# Some features need to be converted to integers with cat.codes. Some of these will then have dummy variables created from them, however features such as pais_residencia will not, because too many features would be created (there are over 100 countries in this dataset).

# In[54]:

print("Step 1/6")
df.pais_residencia = df.pais_residencia.astype('category').cat.codes
print("Step 2/6")
df.canal_entrada = df.canal_entrada.astype('category').cat.codes
print("Step 3/6")
df.nomprov = df.nomprov.astype('category').cat.codes
print("Step 4/6")
final_month.nomprov = final_month.indrel_1mes.astype('category').cat.codes
print("Step 5/6")
df = pd.get_dummies(df, columns = ['ind_empleado','sexo','tiprel_1mes','indresi',
                                   'indext','conyuemp','indfall','segmento','indrel_1mes'])
print("Step 6/6")
#Drop the date features because we can't use them to train the model.
df = df.drop(['fecha_dato', 'fecha_alta'], axis = 1)


# In[55]:

train_final_training_cols, test = df[:train_final_length], df[train_final_length:] 


# In[56]:

print("Step 1/11")
#Get the target columns
labels = train_final[target_cols_all]

print("Step 2/11")
#Add ncodpers to the dataframe
labels['ncodpers'] = train_final.ncodpers

print("Step 3/11")
labels = labels.set_index("ncodpers")

print("Step 4/11")
stacked_labels = labels.stack()

print("Step 5/11")
filtered_labels = stacked_labels.reset_index()

print("Step 6/11")
filtered_labels.columns = ["ncodpers", "product", "newly_added"]

print("Step 7/11")
#Only select the rows where there are a new product.
filtered_labels = filtered_labels[filtered_labels["newly_added"] == 1]

print("Step 8/11")
#Merge with the training features.
multiclass_train = filtered_labels.merge(train_final_training_cols, on="ncodpers", how="left")

print("Step 9/11")
train_final = multiclass_train.drop_duplicates(multiclass_train, keep='last')

print("Step 10/11")
labels_final = train_final['product']

print("Step 11/11")
train_final_ncodpers = train_final.ncodpers
#Remove the columns that are not needed to train the model.
train_final = train_final.drop(['ncodpers','newly_added','product'], axis = 1)


# Below we will perform similar, but not identical, tranformations to the 'final_month' dataframe.

# In[57]:

#Clean the data
print("Step 1/13")
badRows = final_month[final_month.ind_empleado.isnull()].index

print("Step 2/13")
final_month = final_month.drop(badRows, axis = 0)

print("Step 3/13")
final_month.canal_entrada = final_month.canal_entrada.fillna("Unknown")

print("Step 4/13")
final_month = final_month.drop("cod_prov", 1)

print("Step 5/13")
final_month.conyuemp = final_month.conyuemp.fillna("Unknown")

print("Step 6/13")
final_month.loc[final_month.indrel_1mes == '1', 'indrel_1mes'] = 'Primary'
final_month.loc[final_month.indrel_1mes == '1.0', 'indrel_1mes'] = 'Primary'
final_month.loc[final_month.indrel_1mes == 1, 'indrel_1mes'] = 'Primary'
final_month.loc[final_month.indrel_1mes == 1.0, 'indrel_1mes'] = 'Primary'
final_month.loc[final_month.indrel_1mes == '2', 'indrel_1mes'] = 'Co-owner'
final_month.loc[final_month.indrel_1mes == '2.0', 'indrel_1mes'] = 'Co-owner'
final_month.loc[final_month.indrel_1mes == 2, 'indrel_1mes'] = 'Co-owner'
final_month.loc[final_month.indrel_1mes == 2.0, 'indrel_1mes'] = 'Co-owner'
final_month.loc[final_month.indrel_1mes == '3', 'indrel_1mes'] = 'Former Primary'
final_month.loc[final_month.indrel_1mes == '3.0', 'indrel_1mes'] = 'Former Primary'
final_month.loc[final_month.indrel_1mes == 3, 'indrel_1mes'] = 'Former Primary'
final_month.loc[final_month.indrel_1mes == 3.0, 'indrel_1mes'] = 'Former Primary'
final_month.loc[final_month.indrel_1mes == '4', 'indrel_1mes'] = 'Former Co-owner'
final_month.loc[final_month.indrel_1mes == '4.0', 'indrel_1mes'] = 'Former Co-owner'
final_month.loc[final_month.indrel_1mes == 4, 'indrel_1mes'] = 'Former Co-owner'
final_month.loc[final_month.indrel_1mes == 4.0, 'indrel_1mes'] = 'Former Co-owner'

final_month.indrel_1mes = final_month.indrel_1mes.fillna('Primary')

print("Step 7/13")
final_month.nomprov = final_month.nomprov.fillna("MADRID")

print("Step 8/13")
final_month.renta = final_month.renta.astype(float)
final_month.loc[final_month.renta.isnull(), "renta"] = final_month.groupby('nomprov').renta.transform('median')

print("Step 9/13")
final_month.segmento = final_month[final_month.renta <= 98000].segmento.fillna("03 - UNIVERSITARIO")
final_month.segmento = final_month[final_month.renta <= 125500].segmento.fillna("02 - PARTICULARES")
final_month.segmento = final_month.segmento.fillna("01 - TOP")

print("Step 10/13")
final_month.sexo = final_month.sexo.fillna("V")

print("Step 11/13")
final_month.tiprel_1mes = final_month.tiprel_1mes.fillna('I')

print("Step 12/13")
final_month.ind_nomina_ult1 = final_month.ind_nomina_ult1.fillna(0.0)
final_month.ind_nom_pens_ult1 = final_month.ind_nom_pens_ult1.fillna(0.0)

print("Step 13/13")
final_month.loc[final_month.antiguedad == -999999, 'antiguedad'] = final_month[final_month.antiguedad >= 0].antiguedad.median()

#Feature Engineering

print("Step 1/10")
final_month.age = final_month.age.astype(int)
final_month.loc[final_month.age < 20,"age"] = 19
final_month.loc[final_month.age > 90,"age"] = 91

print("Step 2/10")
final_month['ageGroup'] = (final_month.age // 10) - 1

print("Step 3/10")
final_month['isSpanish'] = final_month.pais_residencia.map(lambda x: 1 if x == "ES" else 0)

print("Step 4/10")
final_month['majorCity'] = final_month.nomprov.map(lambda x: 1 if x == "MADRID" or x == "BARCELONA" else 0)

print("Step 5/10")
final_month['fecha_alta_year'] = pd.DatetimeIndex(final_month.fecha_alta).year - 1995
final_month['fecha_dato_year'] = pd.DatetimeIndex(final_month.fecha_dato).year - 2015
final_month['fecha_alta_month'] = pd.DatetimeIndex(final_month.fecha_alta).month - 1
final_month['fecha_dato_month'] = pd.DatetimeIndex(final_month.fecha_dato).month - 1

print("Step 6/10")
final_month.antiguedad = final_month.antiguedad.astype(int)
final_month['antiguedad_years'] = final_month.antiguedad // 12

print("Step 7/10")
final_month.loc[final_month.indrel == 99.0, "indrel"] = 0.0

print("Step 8/10")
final_month['HAS_ult_fec_cli_1t'] = final_month.ult_fec_cli_1t.map(lambda x: 1 if x > 0 else 0)

print("Step 9/10")
final_month = final_month.drop('ult_fec_cli_1t', 1)

print("Step 10/10")
final_month['rentaGroup'] = final_month.renta.astype(float) // 50000
final_month.loc[final_month.renta >= 1000000, "rentaGroup"] = 20
final_month.loc[final_month.renta >= 10000000, "rentaGroup"] = 21


final_month_training_cols = final_month
final_month_training_cols = final_month_training_cols.drop(target_cols_all, axis=1)


print("Step 1/6")
final_month.pais_residencia = final_month.pais_residencia.astype('category').cat.codes
print("Step 2/6")
final_month.canal_entrada = final_month.canal_entrada.astype('category').cat.codes
print("Step 3/6")
final_month.nomprov = final_month.nomprov.astype('category').cat.codes
print("Step 4/6")
final_month.nomprov = final_month.indrel_1mes.astype('category').cat.codes
print("Step 5/6")
final_month = pd.get_dummies(final_month, columns = ['ind_empleado','sexo','tiprel_1mes','indresi',
                                   'indext','conyuemp','indfall','segmento','indrel_1mes'])
print("Step 6/6")
final_month = final_month.drop(['fecha_dato', 'fecha_alta'], axis = 1)



print("Step 1/11")
#Get the target columns
labels_final_month = final_month[target_cols_all]

print("Step 2/11")
#Add ncodpers to the dataframe
labels_final_month['ncodpers'] = final_month.ncodpers

print("Step 3/11")
labels_final_month = labels_final_month.set_index("ncodpers")

print("Step 4/11")
stacked_labels_final_month = labels_final_month.stack()

print("Step 5/11")
filtered_labels_final_month = stacked_labels_final_month.reset_index()

print("Step 6/11")
filtered_labels_final_month.columns = ["ncodpers", "product", "newly_added"]

print("Step 7/11")
#Only select the rows where there is a new product.
filtered_labels_final_month = filtered_labels_final_month[filtered_labels_final_month["newly_added"] == 1]

print("Step 8/11")
#Merge with the training features.
multiclass_final_month = filtered_labels_final_month.merge(final_month_training_cols, on="ncodpers", how="left")

print("Step 9/11")
final_month = multiclass_final_month.drop_duplicates(multiclass_final_month, keep='last')

print("Step 10/11")
labels_final_month_final = final_month['product']

print("Step 11/11")
final_month_ncodpers = final_month.ncodpers
#Remove the columns that are not needed to train the model.
final_month = final_month.drop(['ncodpers','newly_added','product'], axis = 1)


# Let's take a look at the length of our dataframes to help ensure that everything is still in order.

# In[58]:

print len(train_final)
print len(labels_final)
print len(final_month)
print len(labels_final_month_final)


# Yup, everything is still looking good!

# Let's see what labels we are looking to train with.

# In[274]:

labels_final.value_counts()


# # Build the Model

# Convert the values of labels_final to integers so that it can be used by xgboost.

# In[59]:

labels_final = labels_final.astype('category').cat.codes


# Check to make sure we have the same features in train_final and test.

# In[60]:

print len(train_final.columns)
print len(test.columns)


# Nope, we have an extra one, let's find it!

# In[61]:

print train_final.columns
print
print test.columns


# It's 'ncodpers', we'll have to drop it.

# In[62]:

test = test.drop('ncodpers', axis = 1)


# The features below are dropped because I have already ran the model once and learned that these features are not useful.

# In[63]:

train_final = train_final.drop(['tipodom','HAS_ult_fec_cli_1t','ind_empleado_S','indresi_N','indresi_S', 
                                'conyuemp_S','conyuemp_Unknown','indfall_S','indrel_1mes_Co-owner', 
                                'indrel_1mes_Former Primary','indrel_1mes_Primary'],
                               axis = 1)

test = test.drop(['tipodom','HAS_ult_fec_cli_1t','ind_empleado_S','indresi_N','indresi_S', 
                  'conyuemp_S','conyuemp_Unknown','indfall_S','indrel_1mes_Co-owner', 
                  'indrel_1mes_Former Primary','indrel_1mes_Primary'],
                  axis = 1)


# Although I am splitting the data here, I used all of the train_final data for my submission in the Kaggle competition.

# In[64]:

X_train, X_test, y_train, y_test = train_test_split(train_final, labels_final, test_size=0.2, random_state=2)


# Convert the data into matrices so that they can be used by xgboost.

# In[65]:

import warnings
warnings.filterwarnings("ignore")

xgtrain = xgb.DMatrix(X_train, label = y_train)
xgtest = xgb.DMatrix(X_test, label = y_test)
watchlist = [(xgtrain, 'train'), (xgtest, 'eval')] 


# In[66]:

random_state = 4
params = {
        'eta': 0.05,
        'max_depth': 6,
        'min_child_weight': 4,
        'subsample': 0.9,
        'colsample_bytree': 0.9,
        'gamma': 0,
        'alpha': 0,
        'lambda': 1,
        'verbose_eval': True,
        'seed': random_state,
        'num_class': 24,
        'objective': "multi:softprob",
        'eval_metric': 'mlogloss'
    }

''' 
BEST PARAMETERS
params = {
        'eta': 0.05,
        'max_depth': 6,
        'min_child_weight': 4,
        'subsample': 0.9,
        'colsample_bytree': 0.9,
        'gamma': 0,
        'alpha': 0,
        'lambda': 1,
        'verbose_eval': True,
        'seed': random_state,
        'num_class': 16,
        'objective': "multi:softprob",
        'eval_metric': 'mlogloss'
    }
lowest mlogloss: 1.81136, iterations: 143
'''


# Train the model!

# In[88]:

iterations = 40
printN = 1
#early_stopping_rounds = 10

xgbModel = xgb.train(params, 
                      xgtrain, 
                      iterations, 
                      watchlist,
                      verbose_eval = printN
                      #early_stopping_rounds=early_stopping_rounds
                      )


# Use f-score to find the most/least important features. This allowed us to know which features we could remove.

# In[90]:

import operator
importance = xgbModel.get_fscore()
importance = sorted(importance.items(), key=operator.itemgetter(1))
print importance
print len(importance)


# Convert labels to integers so that we can use them to find which products customers already have, and thus, do not need to be included in the final prediction.

# In[91]:

labels_final_month_final_cat = labels_final_month_final.astype('category').cat.codes


# used_products are the products that the customers have already used...duh.

# In[74]:

used_products = pd.DataFrame()
used_products['product'] = labels_final_month_final_cat
used_products['ncodpers'] = final_month_ncodpers
used_products = used_products.drop_duplicates(keep = 'last')


# In[75]:

#create a dictionary to store each product a customer already has
used_recommendation_products = {}
target_cols_all = np.array(target_cols_all)
#iterate through used_products and add each one to used_recommendation_products
for idx,row_val in used_products.iterrows():
    used_recommendation_products.setdefault(row_val['ncodpers'],[]).append(target_cols_all[row_val['product']])
    if len(used_recommendation_products) % 100000 == 0:
        print len(used_recommendation_products)


# In[77]:

len(used_recommendation_products)


# Let's take a look at a customer's used_recommendation_products to see if everything looks alright.

# In[78]:

used_recommendation_products[15889]


# Looks good!

# Use our model to make the predictions

# In[92]:

xgbtest = xgb.DMatrix(test)
XGBpreds = xgbModel.predict(xgbtest)


# Let's take a peek at these predictions

# In[93]:

XGBpreds


# Sort the predictions, then reverse the order so that they are ranked from most likely to least.

# In[94]:

pred = np.argsort(XGBpreds, axis=1)
pred = np.fliplr(pred) 


# In[95]:

pred[0]


# Things look good here because the highest reccommended products are the most common in the training labels.

# In[96]:

#test_ids are the customer codes for the testing data.
test_ids = np.array(pd.read_csv("/Users/Dave/Desktop/Programming/Personal Projects/Santander-Kaggle/test.csv",usecols=['ncodpers'])['ncodpers'])
target_cols_all = np.array(target_cols_all)
final_preds = []
#iterate through our model's predictions (pred) and add the 7 most recommended products that the customer does not have.
for idx,predicted in enumerate(pred):
    ids = test_ids[idx]
    top_product = target_cols_all[predicted]
    used_products = used_recommendation_products.get(ids,[])
    new_top_product = []
    for product in top_product:
        if product not in used_products:
            new_top_product.append(product)
        if len(new_top_product) == 7:
            break
    final_preds.append(' '.join(new_top_product))
    if len(final_preds) % 100000 == 0:
        print len(final_preds)


# Let's take a peek at the final predictions

# In[97]:

final_preds


# final_preds is looking good because the most common products in the training labels are appearing often.

# Let's make our submission to Kaggle by combining the customer codes (ncodpers) with our predictions

# In[98]:

submission = pd.DataFrame({'ncodpers':test_ids,'added_products':final_preds})
submission.to_csv('/Users/Dave/Desktop/Programming/Personal Projects/Santander-Kaggle/submission.csv',index=False)


# Let's check our submission to see if everything looks alright.

# In[511]:

submission.head(10)


# In[102]:

print len(submission)
print len(test)


# # Summary Report

# The goal of this model was to recommend new products to customers of Santander Bank. This was done by comparing the products that a customer had in each month with the previous month. If a customer had a new product in a given month, that month's data would be used to train the model. In order to avoid recommending products that a customer already had, the month prior to the prediction, May 2016, would be compared to the prediction month, June 2016. Any products that a customer had in May 2016 would be removed from the prediction set.
# 
# As part of the requirements for the Kaggle competition, only seven products were needed to be recommended to each customer. The seven products that were most frequently recommended were (English translations are in paraeneses): ind_recibo_ult1 (Direct Debit), ind_nom_pens_ult1 (Pensions), ind_nomina_ult1 (Payroll), ind_cco_fin_ult1 (Current Accounts), ind_tjcr_fin_ult1 (Credit Card), ind_cno_fin_ult1 (Payroll Account), and ind_ecue_fin_ult1 (e-account).
# 
# The seven most common products that customers already had were: ind_cco_fin_ult1(Current Accounts), ind_recibo_ult1 (Direct Debit), ind_ctop_fin_ult1 (Particular Account), ind_ecue_fin_ult1 (e-account), ind_cno_fin_ult1 (Payroll Account), ind_nom_pens_ult1 (Pensions), and ind_nomina_ult1 (Payroll).
# 
# Although I did not perform as well as I would have liked to, and normally do, in this Kaggle Competition, I believe that I still have a useful model. The 'Sample Submission Benchmark' score was 0.004211, my score was 0.0220795, and the winning score was 0.031409 (a like to how the score was caluclated: https://www.kaggle.com/c/santander-product-recommendation/details/evaluation). Although I scored far closer to the winning score than the Sample Submission Benchmark, there is still room for improvement. This could have been done with greater feature engineering, using cross-validation (I chose not to do this given the amount of time it would have taken to train the model), or using an ensemble of models. I joined this competition days before it finished, which gave me some time to be creative and find a useful solution, but more time would have likely been beneficial to improving my final score.

# In[ ]:



