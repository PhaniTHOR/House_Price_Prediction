#!/usr/bin/env python
# coding: utf-8

# **Session-8 31/07/2023**

# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import io
import os


# In[ ]:


get_ipython().run_line_magic('cd', '"D:\\Imarticus\\stat\\house-prices-advanced-regression-techniques"')


# In[ ]:


trainhp=pd.read_csv('train.csv')
trainhp


# In[ ]:


testhp=pd.read_csv('test.csv')
testhp


# In[ ]:


#Temperory add dependent variable into test data
testhp['SalePrice']='test'


# In[ ]:


#concatenate both dataframes for preprocessing
combinedf=pd.concat([trainhp,testhp],axis=0)#row wise concationation


# In[ ]:


#split the data into numeric and object columns
numcols=combinedf.select_dtypes(include=np.number)
objcols=combinedf.select_dtypes(include=['object'])


# In[ ]:


print(numcols.shape)
print(objcols.shape)


# In[ ]:


objcols.columns


# In[ ]:


notavailable=['Alley','BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2','FireplaceQu',
              'GarageType','GarageFinish','GarageQual','GarageCond','PoolQC','Fence','MiscFeature']


# In[ ]:


len(notavailable)


# In[ ]:


for col in notavailable:
    objcols[col]=objcols[col].fillna('NotAvailable')


# In[ ]:


objcols.info()


# In[ ]:


objcols.isnull().sum().sort_values(ascending=False)


# In[ ]:


for col in objcols.columns:
    objcols[col]=objcols[col].fillna(objcols[col].value_counts().idxmax())
#idxmax() identifies the class or index of maximum frequency in value_counts().


# In[ ]:


objcols.isnull().sum().sort_values(ascending=False)


# In[ ]:


objcols.info()


# In[ ]:


#checking null values and null_values count
numcols.isnull().sum().sort_values(ascending=False)


# In[ ]:


numcols.columns


# In[ ]:


categeorycols=numcols[['OverallQual','OverallCond','YearBuilt','YearRemodAdd','GarageYrBlt','MoSold','YrSold']]


# In[ ]:


numcols=numcols.drop(['OverallQual','OverallCond','YearBuilt','YearRemodAdd','GarageYrBlt','MoSold','YrSold'],axis=1)


# In[ ]:


numcols.isnull().sum().sort_values(ascending=False)


# In[ ]:


#missing values impetation
for col in numcols.columns:
    numcols[col]=numcols[col].fillna(numcols[col].median())


# In[ ]:


numcols.info()


# In[ ]:


for col in categeorycols.columns:
    categeorycols[col]=categeorycols[col].fillna(categeorycols[col].median())


# In[ ]:


categeorycols.info()


# In[ ]:


#correlation analysis of numeric data
plt.figure(figsize=(30,15))
sns.heatmap(numcols.corr(),annot=True)


# In[ ]:


plt.figure(figsize=(30,15))
sns.heatmap(numcols.corr(),annot=True,cmap='plasma')


# ![image.png](attachment:image.png)

# ![image.png](attachment:image.png)

# In[ ]:


numcols['SalePrice']=objcols.SalePrice


# In[ ]:


objcols=objcols.drop('SalePrice',axis=1)


# In[ ]:


objcols.columns


# In[ ]:


#trainhp=pd.drop(trainhp.Alley,axis=1)


# In[ ]:


#trainhp=pd.drop(trainhp.FireplaceQu,axis=1)


# trainhp=pd.drop(trainhp.PoolQC,axis=1)
# trainhp=pd.drop(trainhp.Fence,axis=1)
# trainhp=pd.drop(trainhp.MiscFeature,axis=1)

# **Session2 ML 01-08-2023**

# In[ ]:


objcols.shape


# In[ ]:


numcols.shape


# # 28-08-2023

# # dummy encode object cols and categorycols
# objcols_dummy=pd.get_dummies(objcols)

# objcols_dummy.shape

# In[ ]:


from sklearn.preprocessing import LabelEncoder


# In[ ]:


object_dummy=objcols.apply(LabelEncoder().fit_transform)


# In[ ]:


category_dummy=categeorycols.apply(LabelEncoder().fit_transform)


# In[ ]:


combinedf_clean=pd.concat([numcols,object_dummy,category_dummy],axis=1)


# In[ ]:


combinedf_clean.shape


# In[ ]:


# split data back to train to test
housetrain_df=combinedf_clean[combinedf_clean.SalePrice!='test']
housetest_df=combinedf_clean[combinedf_clean.SalePrice=='test']


# In[ ]:


# drop the dependent variable from test data
housetest_df=housetest_df.drop('SalePrice',axis=1)


# In[ ]:


housetest_df.info()


# In[ ]:


# split data into dependent variable(y) and independent variable(X)
y=housetrain_df.SalePrice
x=housetrain_df.drop(['Id','SalePrice'],axis=1)


# In[ ]:


y=y.astype('int64')


# In[ ]:


#histogram,boxplot,density curve of y
plt.figure(figsize=(30,15))
fig,ax=plt.subplots(3,1)
sns.histplot(y,ax=ax[0])
sns.boxplot(y,orient='h',ax=ax[1])
sns.kdeplot(y,ax=ax[2])


# In[ ]:


#histogram,boxplot,density curve of y
plt.figure(figsize=(30,15))
fig,ax=plt.subplots(3,1)
sns.histplot(np.log1p(y),ax=ax[0])
sns.boxplot(np.log1p(y),orient='h',ax=ax[1])
sns.kdeplot(np.log1p(y),ax=ax[2])


# In[ ]:


from sklearn.linear_model import LinearRegression


# In[ ]:


reg=LinearRegression()


# In[ ]:


regmodel=reg.fit(x,np.log(y))


# In[ ]:


regmodel.score(x,np.log(y)) # r_square


# In[ ]:


housetest_df=housetest_df.drop('Id',axis=1)


# In[ ]:


regpredict=regmodel.predict(housetest_df)


# In[ ]:


pd.DataFrame(np.exp(regpredict)).to_csv('reg.csv')


# In[ ]:





# In[ ]:


from sklearn.tree import DecisionTreeRegressor


# In[ ]:


tree=DecisionTreeRegressor(max_depth=6)


# In[ ]:


treemodel=tree.fit(x,y)


# In[ ]:


treemodel.score(x,y)


# In[ ]:


from sklearn.model_selection import cross_val_score


# In[ ]:


cross_val_score(tree,x,y)


# In[ ]:


np.mean([0.71561352, 0.75268719, 0.80528335, 0.74688138, 0.57884563])


# In[ ]:


treepredict=treemodel.predict(housetest_df)


# In[ ]:


pd.DataFrame(treepredict).to_csv('tree.csv')


# In[ ]:


pd.DataFrame(treemodel.feature_importances_,x.columns).sort_values(by=0,ascending=False)


# In[ ]:


from sklearn.linear_model import Lasso


# In[ ]:


las=Lasso()


# In[ ]:


lasmodel=las.fit(x,y)


# In[ ]:


lasmodel.score(x,y)


# In[ ]:


lasso_params={'alpha':[0.5,0.25,1,2,3,4,5,10,15]}


# In[ ]:


from sklearn.model_selection import GridSearchCV


# In[ ]:


lasso_grid=GridSearchCV(estimator=Lasso(max_iter=3000,selection='random'),param_grid=lasso_params).fit(x,y)


# In[ ]:


lasso_grid.best_estimator_


# In[ ]:


lasso_grid.best_score_


# In[ ]:


lasso_predict=lasso_grid.predict(housetest_df)


# In[ ]:


pd.DataFrame(lasso_predict).to_csv('lasso.csv')


# In[ ]:





# In[ ]:


from sklearn.linear_model import Ridge


# In[ ]:


rid=Ridge()


# In[ ]:


ridmodel=rid.fit(x,y)


# In[ ]:


ridmodel.score(x,y)


# In[ ]:


rid_predict=ridmodel.predict(housetest_df)


# In[ ]:


pd.DataFrame(rid_predict).to_csv('rid.csv')


# In[ ]:





# In[ ]:


from sklearn.ensemble import RandomForestRegressor


# In[ ]:


RF=RandomForestRegressor(n_estimators=3000)


# In[ ]:


rfmodel=RF.fit(x,y)


# In[ ]:


rfmodel.score(x,y)


# In[ ]:


cross_val_score(RF,x,y)


# In[ ]:


np.mean([0.87590798, 0.8462566 , 0.87457573, 0.88311702, 0.81761072])


# In[ ]:


rfpredict=rfmodel.predict(housetest_df)


# In[ ]:


pd.DataFrame(rfpredict).to_csv('rf.csv')


# In[ ]:





# In[ ]:


from sklearn.ensemble import GradientBoostingRegressor


# In[ ]:


gbm=GradientBoostingRegressor(n_estimators=3000)


# In[ ]:


gmmodel=gbm.fit(x,y)


# In[ ]:


gbm.score(x,y)


# In[ ]:


cross_val_score(gbm,x,y)


# In[ ]:


np.mean([0.91103841, 0.84305525, 0.90537834, 0.89691904, 0.90093657])


# In[ ]:


gbmpredict=gmmodel.predict(housetest_df)


# In[ ]:


pd.DataFrame(gbmpredict).to_csv('gbm.csv')


# In[ ]:





# In[ ]:


from sklearn.linear_model import ElasticNet


# In[ ]:


el=ElasticNet(alpha=90,max_iter=8000)


# In[ ]:


elmodel=el.fit(x,y)


# In[ ]:


elmodel.score(x,y)


# In[ ]:


rfpredict=elmodel.predict(housetest_df)


# In[ ]:


pd.DataFrame(rfpredict).to_csv('el.csv')


# In[ ]:


cross_val_score(el,x,y)


# In[ ]:


np.mean([0.79559838, 0.75264743, 0.80879966, 0.76896685, 0.54853002])


# from sklearn.preprocessing import StandardScaler

# In[ ]:




