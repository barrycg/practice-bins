# -*- coding: utf-8 -*-
"""
Created on Sun Jul 21 16:51:05 2019

@author: hewei
"""

'''
一个完整机器学习项目的实例
1.项目概述
2.获取数据
3.发现并可视化数据，发现规律
4.为机器学习算法准备数据
5.选择模型，进行训练
6.微调模型
7.给出解决方案
8.部署、监控、维护系统

'''


# 项目概览， 模型要利用这个数据进行学习，然后根据其它指标，预测任何街区的房价中位数。
## 划定问题， 如何划定问题，选择什么算法，评估模型性能的指标是什么，要花多少精力进行微调。
### 一系列的数据处理组件被称之为数据流水线。 流水线在机器学习系统中很常见，因为有很多数据要处理和转换。

#### 老板会告诉你你的模型输出，会传给另一个机器学习系统，也会有其他的信号传入后面的系统。
#### 下一个问题是现在的解决方案效果如何。 
##### 进行问题的划定： 监督或非监督， 还是强化学习？  分类还是回归，还是其它？要使用批量学习还是线上学习？
##### 很明显，这是一个典型的监督学习任务，因为你要使用的是有标签的训练样本，并且是典型的回归任务，预测一个值。这是一个多变量回归问题。

## 选择性能指标 
### 回归问题的典型指标是均方根误差（RMSE）。均方根误差测i昂的是系统预测误差的标准差。
### 很多情况下，你需要另外的函数。如你可能需要使用平均绝对误差（Mean Absolute Error, 平均绝对误差）

# 获取数据

import os
import tarfile
from six.moves import urllib

DOWNLOAD_ROOT	=	"https://raw.githubusercontent.com/ageron/handson-ml/master/" 
HOUSING_PATH	=	"datasets/housing" 
HOUSING_URL	=	DOWNLOAD_ROOT	+	HOUSING_PATH	+	"/housing.tgz"

def fetch_housing_data( housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    if not os.path.isdir(housing_path):
        os.makedirs(housing_path)
    tgz_path  = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()
    
import pandas as pd

def load_housing_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)

# 创建数据集， 测试集一般是数据集的20%

import numpy as np

def split_train_test(data, test_ratio):
    np.random.seed(42) ### 设置不同的随机数生成器的种子
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data)*test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]

## 用哈希值来检测数据集
import hashlib
def test_set_check(identifier, test_ratio, hash):
    return hash(np.int64(identifier)).digest()[-1] < 256 * test_ratio

def split_train_test_by_id(data, test_ratio, id_column, hash=hashlib.md5):
    ids = data[id_column]
    in_test_set = ids.apply(lambda id_: test_set_check(id_, test_ratio, hash))
    return data.loc[~in_test_set], data.loc[in_test_set]

import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedShuffleSplit

from pandas.tools.plotting import scatter_matrix

from sklearn.preprocessing import Imputer

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import LabelEncoder

if __name__ == "__main__":
   # fetch_housing_data()
    housing = load_housing_data()
    housing.head()
    housing.info()
    housing["ocean_proximity"].value_counts()
    housing.hist(bins=50,figsize=(20,15))
    plt.show()
    
    train_set, test_set = split_train_test(housing, 0.2)
    print(len(train_set), "train +", len(test_set), "test")

#### 合并经纬度进行作为 train_id
    housing_with_id = housing.reset_index()
    
    train_set,	test_set= split_train_test_by_id( housing_with_id, 0.2, "index" )
    
#   housing_with_id["id"] = housing["longitude"] * 1000 + housing["latitude"]
#   train_set, test_set = split_train_test_by_id(housing_with_id, 0.2, "id")
    
    housing["income_cat"] = np.ceil(housing["median_income"]/1.5)
    housing["income_cat"].where(housing["income_cat"] < 5, 5.0, inplace=True)
    
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_index, test_index in split.split(housing, housing["income_cat"]):
        strat_train_set = housing.loc[train_index]
        strat_test_set = housing.loc[test_index]
        
    housing["income_cat"].value_counts() / len(housing)
    
    for set in (strat_train_set, strat_test_set):
        set.drop(["income_cat"], axis=1, inplace=True)
        
# 数据探索和可视化、发现规律
    housing = strat_train_set.copy()

# 地理数据可视化
    housing.plot(kind="scatter", x="longitude", y="latitude")
    housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.1)
    housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4,
                 s=housing["population"]/100, label="population",
                 c="median_house_value", cmap=plt.get_cmap("jet"),colorbar=True)
    plt.legend()
    
    
#   standard correlation coefficient
    corr_matrix = housing.corr()
    corr_matrix["median_house_value"].sort_values(ascending=False)
    
    attributes = ["median_house_value", "median_income", "total_rooms", 
                  "housing_median_age"]
    scatter_matrix(housing[attributes], figsize=(12, 8))
    housing_plot(kind="scatter", x="median_income", 
                         y="median_house_value",alpha=0.1)
    
    housing["rooms_per_household"]=housing["total_rooms"]/housing["households"]
    housing["bedrooms_per_room"]=housing["total_bedrooms"]/housing["total_rooms"] 
    housing["population_per_household"]=housing["population"]/housing["households"]

    corr_matrix = housing.corr()
    corr_matrix["median_house_value"].sort_values(ascending=False)
    
#  为机器学习算法准备数据
    housing = strat_train_set.drop("median_house_value", axis=1)
    housing_labels = strat_train_set["median_house_value"].copy()

##  数据清洗
###1.去掉对应的街区；2.去掉整个属性；3.进行赋值（0、平均值、中位数等等）
    housing.dropna(subset=["total_bedrooms"])   #选项1
    housing.drop("total_bedrooms", axis=1)      #选项2
    median=housing["total_bedrooms"].median()
    housing["total_bedrooms"].fillna(median)  #选项3
    
### Scikit-Learn  提供Imputer, 创建Imputer实例，指定某属性的中位数来替代该属性所有的缺失值
    imputer = Imputer(strategy="median")
    housing_num = housing.drop("ocean_proximity", axis=1)
    imputer.fit(housing_num)
    
    imputer.statistics_
    housing_num.median().values
    
    X = imputer.transform(housing_num)
    housing_tr = pd.DataFrame(X, columns=housing_num.columns)
    
## 处理文本和类别属性
### sciki-Learn 提供了一个转换器LabelEncoder
    encoder = LabelEncoder()
    housing_cat = housing["ocean_proximity"]
    housing_cat_encoded = encoder.fit_transform(housing_cat)
    print(housing_cat_encoded)
    
### 独热向量，独热编码转换器
    
    


### 特征缩放 两种常见的方法可以让所有的属性具有相同的度量：线性函数归一化（Min-Max scaling）和标准化（standardization）
#### 线性函数归一化（normalization） Scikit-Learn 提供了一个转换器MinMaxScaler来实现这个功能。
#### 标准化很不一样。Scikit-learn提供了一个转换器StandardScaker来进行标准化。
    
##  转换流水线  使得很多数据转换步骤，按照一定的顺序执行。
    
    
# 选择并训练模型
    
# 模型调优

    print( 'debugging  ------------- ')
    print( corr_matrix["median_house_value"] )






















