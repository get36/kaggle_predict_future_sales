# -*- coding: utf-8 -*-
"""
Created on Sat Jul 20 18:54:27 2019

@author: PC
"""

import pandas as pd
import numpy as np
########################load data############################
data_train=pd.read_csv('/home/guoqiang/recommend/data/sales_train.csv/sales_train_v2.csv',engine='python')
data_test=pd.read_csv('/home/guoqiang/recommend/data/test.csv/test.csv',engine='python')
X_train=data_train[['item_price']]
X_label=data_train[['item_cnt_day']]
test_t=pd.read_csv('/home/guoqiang/recommend/test_t.csv',engine='python')
########################add item_price#######################
#Y_train=data_test[['shop_id','item_id']]
Y_train=test_t[['item_price']]
#Y_train['item_price']=None
#for i in range(len(Y_train)):
#    a=Y_train.loc[i,['shop_id']]
#    b=Y_train.loc[i,['item_id']]
#    shop=X_train.shop_id
#    shop_index=shop.isin(a)
#    item=X_train.item_id
#    item_index=item.isin(b)
#    price=X_train.loc[shop_index & item_index]
#    panbie=price.empty
#    if panbie==False:
 #       price=price.reset_index(drop=True)
       # print(price)
#        Y_train.loc[i,['item_price']]=price.loc[len(price)-1,'item_price']
#Y_train['item_price'].fillna(Y_train['item_price'].mean(),inplace=True)

#X_label=X_label[['item_cnt_day']][(X_train.date_block_num>=30) &( X_train.date_block_num<=33)]
#X_train=X_train[['shop_id','item_id','item_price']][(X_train.date_block_num>=30) &( X_train.date_block_num<=33)]


###################################feature_extraction###############################


#from sklearn.feature_extraction import DictVectorizer
#dict_vec=DictVectorizer(sparse=False)
#X_train[['shop_id']]=dict_vec.fit_transform(X_train[['shop_id']].to_dict(orient='record'))
#Y_train[['shop_id']]=dict_vec.fit_transform(Y_train[['shop_id']].to_dict(orient='record'))
#X_train[['item_id']]=dict_vec.fit_transform(X_train[['item_id']].to_dict(orient='record'))
#Y_train[['item_id']]=dict_vec.fit_transform(Y_train[['item_id']].to_dict(orient='record'))
#print(X_train)

#shop_dummies_train=pd.get_dummies(X_train.shop_id,prefix='shop_id')
#item_dummies_train=pd.get_dummies(X_train.item_id,prefix='item_id')
#shop_dummies_test=pd.get_dummies(Y_train.shop_id,prefix='shop_id')
#item_dummies_test=pd.get_dummies(Y_train.item_id,prefix='item_id')
#X_train=pd.concat([X_train,shop_dummies_train],axis=1)
#Y_train=pd.concat([Y_train,shop_dummies_test],axis=1)
#X_train.drop(['shop_id'],axis=1,inplace=True)
#Y_train.drop(['shop_id'],axis=1,inplace=True)
#for i in X_train.columns:
#    if i not in Y_train.columns:
 #       Y_train[i]=0

#Y_train=Y_train[X_train.columns]
#print(X_train.info())
#print(Y_train.info())

#Y_train['item_price']=None


######################Normalize###########################
from sklearn.preprocessing import MinMaxScaler
minmax=MinMaxScaler()

x=np.array(X_train['item_price'])
x_std=minmax.fit_transform(x.reshape(-1,1))
X_train.loc[:,['item_price']]=x_std


x=np.array(Y_train['item_price'])
x_std=minmax.fit_transform(x.reshape(-1,1))
Y_train.loc[:,['item_price']]=x_std
##########################################################





from sklearn.linear_model import LogisticRegression
gbr=LogisticRegression()
gbr.fit(X_train,X_label)
gbr_predict=gbr.predict(Y_train)
print(gbr.coef_)
gbr_submission=pd.DataFrame({'ID':data_test['ID'],'item_cnt_month':gbr_predict})
gbr_submission.to_csv('submission.csv',index=False)

