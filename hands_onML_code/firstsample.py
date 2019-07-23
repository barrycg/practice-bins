# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 10:19:13 2019

@author: hewei
"""

import os
import tarfile
from six.moves import urllib

import pandas as pd

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

import	matplotlib.pyplot as plt
from pandas.plotting	import	scatter_matrix
from sklearn.model_selection import	StratifiedShuffleSplit
from sklearn.preprocessing import Imputer
from sklearn.linear_model import LinearRegression

from sklearn.pipeline import FeatureUnion
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import CategoricalEncoder

class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names=attribute_names
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return X[self.attribute_names].values
        
rooms_ix, bedrooms_ix, population_ix, household_ix = 3, 4, 5, 6

class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room = True): # no *args or **kargs
        self.add_bedrooms_per_room=add_bedrooms_per_room

    def fit(self, X, y=None):
        return self # nothing else to do

    def transform(self, X, y=None):
        rooms_per_household=X[:,rooms_ix]/X[:,household_ix]
        population_per_household=X[:,population_ix]/X[:,household_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:,bedrooms_ix]/X[:,rooms_ix]
            return np.c_[X, rooms_per_household, population_per_household, bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]

from sklearn.base import BaseEstimator,	TransformerMixin
from sklearn.utils import check_array
from sklearn.preprocessing import LabelEncoder
from scipy import sparse

class CategoricalEncoder(BaseEstimator, TransformerMixin):
    """Encode categorical features as a numeric array.
    The input to this transformer should be an array-like of integers or
    strings, denoting the values taken on by categorical (discrete) features.
    The features can be encoded using a one-hot (aka one-of-K or dummy)
    encoding scheme (``encoding='onehot'``, the default) or converted
    to ordinal integers (``encoding='ordinal'``).
    This encoding is needed for feeding categorical data to many scikit-learn
    estimators, notably linear models and SVMs with the standard kernels.
    Read more in the :ref:`User Guide <preprocessing_categorical_features>`.
    Parameters
    ----------
    encoding : str, 'onehot', 'onehot-dense' or 'ordinal'
        The type of encoding to use (default is 'onehot'):
        - 'onehot': encode the features using a one-hot aka one-of-K scheme
          (or also called 'dummy' encoding). This creates a binary column for
          each category and returns a sparse matrix.
        - 'onehot-dense': the same as 'onehot' but returns a dense array
          instead of a sparse matrix.
        - 'ordinal': encode the features as ordinal integers. This results in
          a single column of integers (0 to n_categories - 1) per feature.
    categories : 'auto' or a list of lists/arrays of values.
        Categories (unique values) per feature:
        - 'auto' : Determine categories automatically from the training data.
        - list : ``categories[i]`` holds the categories expected in the ith
          column. The passed categories must be sorted and should not mix
          strings and numeric values.
        The used categories can be found in the ``categories_`` attribute.
    dtype : number type, default np.float64
        Desired dtype of output.
    handle_unknown : 'error' (default) or 'ignore'
        Whether to raise an error or ignore if a unknown categorical feature is
        present during transform (default is to raise). When this parameter
        is set to 'ignore' and an unknown category is encountered during
        transform, the resulting one-hot encoded columns for this feature
        will be all zeros. In the inverse transform, an unknown category
        will be denoted as None.
        Ignoring unknown categories is not supported for
        ``encoding='ordinal'``.
    Attributes
    ----------
    categories_ : list of arrays
        The categories of each feature determined during fitting
        (in order corresponding with output of ``transform``).
    Examples
    --------
    Given a dataset with two features, we let the encoder find the unique
    values per feature and transform the data to a binary one-hot encoding.
    >>> from sklearn.preprocessing import CategoricalEncoder
    >>> enc = CategoricalEncoder(handle_unknown='ignore')
    >>> X = [['Male', 1], ['Female', 3], ['Female', 2]]
    >>> enc.fit(X)
    ... # doctest: +ELLIPSIS
    CategoricalEncoder(categories='auto', dtype=<... 'numpy.float64'>,
              encoding='onehot', handle_unknown='ignore')
    >>> enc.categories_
    [array(['Female', 'Male'], dtype=object), array([1, 2, 3], dtype=object)]
    >>> enc.transform([['Female', 1], ['Male', 4]]).toarray()
    array([[ 1.,  0.,  1.,  0.,  0.],
           [ 0.,  1.,  0.,  0.,  0.]])
    >>> enc.inverse_transform([[0, 1, 1, 0, 0], [0, 0, 0, 1, 0]])
    array([['Male', 1],
           [None, 2]], dtype=object)
    See also
    --------
    sklearn.preprocessing.OneHotEncoder : performs a one-hot encoding of
      integer ordinal features. The ``OneHotEncoder assumes`` that input
      features take on values in the range ``[0, max(feature)]`` instead of
      using the unique values.
    sklearn.feature_extraction.DictVectorizer : performs a one-hot encoding of
      dictionary items (also handles string-valued features).
    sklearn.feature_extraction.FeatureHasher : performs an approximate one-hot
      encoding of dictionary items or strings.
    """

    def __init__(self, encoding='onehot', categories='auto', dtype=np.float64,
                 handle_unknown='error'):
        self.encoding = encoding
        self.categories = categories
        self.dtype = dtype
        self.handle_unknown = handle_unknown

    def fit(self, X, y=None):
        """Fit the CategoricalEncoder to X.
        Parameters
        ----------
        X : array-like, shape [n_samples, n_features]
            The data to determine the categories of each feature.
        Returns
        -------
        self
        """
        if self.encoding not in ['onehot', 'onehot-dense', 'ordinal']:
            template = ("encoding should be either 'onehot', 'onehot-dense' "
                        "or 'ordinal', got %s")
            raise ValueError(template % self.handle_unknown)

        if self.handle_unknown not in ['error', 'ignore']:
            template = ("handle_unknown should be either 'error' or "
                        "'ignore', got %s")
            raise ValueError(template % self.handle_unknown)

        if self.encoding == 'ordinal' and self.handle_unknown == 'ignore':
            raise ValueError("handle_unknown='ignore' is not supported for"
                             " encoding='ordinal'")

        if self.categories != 'auto':
            for cats in self.categories:
                if not np.all(np.sort(cats) == np.array(cats)):
                    raise ValueError("Unsorted categories are not yet "
                                     "supported")

        X_temp = check_array(X, dtype=None)
        if not hasattr(X, 'dtype') and np.issubdtype(X_temp.dtype, str):
            X = check_array(X, dtype=np.object)
        else:
            X = X_temp

        n_samples, n_features = X.shape

        self._label_encoders_ = [LabelEncoder() for _ in range(n_features)]

        for i in range(n_features):
            le = self._label_encoders_[i]
            Xi = X[:, i]
            if self.categories == 'auto':
                le.fit(Xi)
            else:
                if self.handle_unknown == 'error':
                    valid_mask = np.in1d(Xi, self.categories[i])
                    if not np.all(valid_mask):
                        diff = np.unique(Xi[~valid_mask])
                        msg = ("Found unknown categories {0} in column {1}"
                               " during fit".format(diff, i))
                        raise ValueError(msg)
                le.classes_ = np.array(self.categories[i])

        self.categories_ = [le.classes_ for le in self._label_encoders_]

        return self

    def transform(self, X):
        """Transform X using specified encoding scheme.
        Parameters
        ----------
        X : array-like, shape [n_samples, n_features]
            The data to encode.
        Returns
        -------
        X_out : sparse matrix or a 2-d array
            Transformed input.
        """
        X_temp = check_array(X, dtype=None)
        if not hasattr(X, 'dtype') and np.issubdtype(X_temp.dtype, str):
            X = check_array(X, dtype=np.object)
        else:
            X = X_temp

        n_samples, n_features = X.shape
        X_int = np.zeros_like(X, dtype=np.int)
        X_mask = np.ones_like(X, dtype=np.bool)

        for i in range(n_features):
            Xi = X[:, i]
            valid_mask = np.in1d(Xi, self.categories_[i])

            if not np.all(valid_mask):
                if self.handle_unknown == 'error':
                    diff = np.unique(X[~valid_mask, i])
                    msg = ("Found unknown categories {0} in column {1}"
                           " during transform".format(diff, i))
                    raise ValueError(msg)
                else:
                    # Set the problematic rows to an acceptable value and
                    # continue `The rows are marked `X_mask` and will be
                    # removed later.
                    X_mask[:, i] = valid_mask
                    Xi = Xi.copy()
                    Xi[~valid_mask] = self.categories_[i][0]
            X_int[:, i] = self._label_encoders_[i].transform(Xi)

        if self.encoding == 'ordinal':
            return X_int.astype(self.dtype, copy=False)

        mask = X_mask.ravel()
        n_values = [cats.shape[0] for cats in self.categories_]
        n_values = np.array([0] + n_values)
        feature_indices = np.cumsum(n_values)

        indices = (X_int + feature_indices[:-1]).ravel()[mask]
        indptr = X_mask.sum(axis=1).cumsum()
        indptr = np.insert(indptr, 0, 0)
        data = np.ones(n_samples * n_features)[mask]

        out = sparse.csr_matrix((data, indices, indptr),
                                shape=(n_samples, feature_indices[-1]),
                                dtype=self.dtype)
        if self.encoding == 'onehot-dense':
            return out.toarray()
        else:
            return out

    def inverse_transform(self, X):
        """Convert back the data to the original representation.
        In case unknown categories are encountered (all zero's in the
        one-hot encoding), ``None`` is used to represent this category.
        Parameters
        ----------
        X : array-like or sparse matrix, shape [n_samples, n_encoded_features]
            The transformed data.
        Returns
        -------
        X_tr : array-like, shape [n_samples, n_features]
            Inverse transformed array.
        """
        check_is_fitted(self, 'categories_')
        X = check_array(X, accept_sparse='csr')

        n_samples, _ = X.shape
        n_features = len(self.categories_)
        n_transformed_features = sum([len(cats) for cats in self.categories_])

        # validate shape of passed X
        msg = ("Shape of the passed X data is not correct. Expected {0} "
               "columns, got {1}.")
        if self.encoding == 'ordinal' and X.shape[1] != n_features:
            raise ValueError(msg.format(n_features, X.shape[1]))
        elif (self.encoding.startswith('onehot')
                and X.shape[1] != n_transformed_features):
            raise ValueError(msg.format(n_transformed_features, X.shape[1]))

        # create resulting array of appropriate dtype
        dt = np.find_common_type([cat.dtype for cat in self.categories_], [])
        X_tr = np.empty((n_samples, n_features), dtype=dt)

        if self.encoding == 'ordinal':
            for i in range(n_features):
                labels = X[:, i].astype('int64')
                X_tr[:, i] = self.categories_[i][labels]

        else:  # encoding == 'onehot' / 'onehot-dense'
            j = 0
            found_unknown = {}

            for i in range(n_features):
                n_categories = len(self.categories_[i])
                sub = X[:, j:j + n_categories]

                # for sparse X argmax returns 2D matrix, ensure 1D array
                labels = np.asarray(_argmax(sub, axis=1)).flatten()
                X_tr[:, i] = self.categories_[i][labels]

                if self.handle_unknown == 'ignore':
                    # ignored unknown categories: we have a row of all zero's
                    unknown = np.asarray(sub.sum(axis=1) == 0).flatten()
                    if unknown.any():
                        found_unknown[i] = unknown

                j += n_categories

            # if ignored are found: potentially need to upcast result to
            # insert None values
            if found_unknown:
                if X_tr.dtype != object:
                    X_tr = X_tr.astype(object)

                for idx, mask in found_unknown.items():
                    X_tr[mask, idx] = None

        return X_tr

def display_scores(scores):
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard deviation:", scores.std())

from sklearn.metrics import	mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import	cross_val_score
from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import GridSearchCV

if __name__ == "__main__":

#### 获取是数据
    housing = load_housing_data()

    housing.hist(bins=50,figsize=(20,15))
    plt.show()
    
#### 创建数据集  随机数据集合
    train_set, test_set = split_train_test(housing, 0.2)
    print(len(train_set), "train +", len(test_set), "test")
    
#### 创建 中位数数据集合
    housing["income_cat"] =	np.ceil(housing["median_income"]/1.5)
    housing["income_cat"].where(housing["income_cat"]<5, 5.0,inplace=True)
    
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_index, test_index in split.split(housing, housing["income_cat"]):
        strat_train_set = housing.loc[train_index]
        strat_test_set  = housing.loc[test_index]
    
    for set in (strat_train_set, strat_test_set):
        set.drop(["income_cat"],  axis=1, inplace=True)
        
#### 数据探索，可视化，发现规律
    #pass

#### 查找关联
 
    #### 标准误差方式
 #   corr_matrix	= housing.corr()
 #   corr_matrix["median_house_value"].sort_values(ascending=False)
    
    #### 另一种关联检测 
    
 #   attributes=	["median_house_value",	"median_income", "total_rooms",
 #                     "housing_median_age"]
#    scatter_matrix(housing[attributes],	figsize=(12, 8))
    
  #  housing["rooms_per_household"]	=	housing["total_rooms"]/housing["households"] 
  #  housing["bedrooms_per_room"] = housing["total_bedrooms"]/housing["total_rooms"]
  #  housing["population_per_household"] = housing["population"]/housing["households"]
    
  #  corr_matrix	= housing.corr()
#    corr_matrix["median_house_value"].sort_values(ascending=False)

## 数据探索结束，开始为机器学习准备数据
    housing=strat_train_set.drop("median_house_value", axis=1)
    housing_labels=strat_train_set["median_house_value"].copy()
    
### 数据清洗
#### 手动清洗
    housing.dropna(subset=["total_bedrooms"])#	选项1 
    housing.drop("total_bedrooms",	axis=1) #	选项2 
    median = housing["total_bedrooms"].median()
    housing["total_bedrooms"].fillna(median) #	选项3
####标准库来 imputer 处理缺失值
    
    imputer	= Imputer(strategy="median")
    housing_num	= housing.drop("ocean_proximity", axis=1)
    imputer.fit(housing_num)
    
    X = imputer.transform(housing_num)
    housing_tr = pd.DataFrame(X, columns=housing_num.columns)
    
### 处理文本和类别属性   文本向量到整数向量
    housing_cat	= housing["ocean_proximity"]
    housing_cat_encoded, housing_categories	= housing_cat.factorize()
    housing_cat_encoded[:10]
    
    ###独热向量独热编码便是解决这个问题，
    ###其方法是使用N位状态寄存器来对N个状态进行编码，每个状态都由他独立的寄存器位，并且在任意时候，其中只有一位有效。
    ####文本编码和独热编码
    
### 流水线处理数据
    
    num_attribs=list(housing_num) 
    cat_attribs=["ocean_proximity"]
    num_pipeline=Pipeline([('selector', DataFrameSelector(num_attribs)),
                           ('imputer', Imputer(strategy="median")),
                           ('attribs_adder',CombinedAttributesAdder()),
                           ('std_scaler',StandardScaler()),])

#    cat_pipeline=Pipeline([('selector', DataFrameSelector(cat_attribs)),
#                           ('label_binarizer', LabelBinarizer()),])
    
    cat_pipeline=Pipeline([('selector', DataFrameSelector(cat_attribs)),
                           ('cat_encoder', CategoricalEncoder(encoding="onehot-dense")),])
    
    
    full_pipeline = FeatureUnion(transformer_list=[("num_pipeline",num_pipeline),
                                                 ("cat_pipeline",	cat_pipeline),])
    housing_prepared = full_pipeline.fit_transform(housing)
    
    
### 选择并训练模型
  #### 在训练集上训练和评估
  
    lin_reg	=	LinearRegression()
    lin_reg.fit(housing_prepared,	housing_labels)

    some_data = housing.iloc[:5]
    some_labels = housing_labels.iloc[:5]
    some_data_prepared = full_pipeline.transform(some_data)
    print("Predictions:\t", lin_reg.predict(some_data_prepared))
    print("Labels:\t\t", list(some_labels))
    
    housing_predictions = lin_reg.predict(housing_prepared)
    lin_mse	= mean_squared_error(housing_labels, housing_predictions)
    lin_rmse = np.sqrt(lin_mse)
    print(lin_rmse)
    
#### 过拟合的训练结果
    tree_reg = DecisionTreeRegressor()
    tree_reg.fit(housing_prepared, housing_labels)

    housing_predictions = tree_reg.predict(housing_prepared)
    tree_mse = mean_squared_error(housing_labels, housing_predictions)
    tree_rmse = np.sqrt(tree_mse)
    print(tree_rmse)

#### 交叉验证    
    tree_scores = cross_val_score(tree_reg, housing_prepared, housing_labels,
				scoring="neg_mean_squared_error", cv=10)
    tree_rmse_scores	= np.sqrt(-tree_scores)
    display_scores(tree_rmse_scores)

######线性回归的验证
    lin_scores=cross_val_score(lin_reg, housing_prepared, housing_labels,
                               scoring="neg_mean_squared_error", cv=10)
    lin_rmse_scores=np.sqrt(-lin_scores)
    display_scores(lin_rmse_scores)
##### 随机森林的验证
    forest_reg=RandomForestRegressor()
    forest_reg.fit(housing_prepared, housing_labels)
    housing_predictions = forest_reg.predict(housing_prepared)
    forest_mse = mean_squared_error(housing_labels, housing_predictions)
    forest_rmse = np.sqrt(tree_mse)
    print(forest_rmse)
    
    forest_scores=cross_val_score(forest_reg, housing_prepared, housing_labels,
                               scoring="neg_mean_squared_error", cv=10)
    forest_rmse_scores=np.sqrt(-forest_scores)
    display_scores(forest_rmse_scores)
## 模型微调
#### 手工调整超参数，直到找到一个好的超参数组合。 
##### 网格搜索
    param_grid = [{'n_estimators': [3, 10, 30],'max_features': [2, 4, 6, 8]},
        {'bootstrap': [False],
         'n_estimators':  [3, 10],
         'max_features':  [2, 3,  4]},]
    forest_reg = RandomForestRegressor()
    grid_search = GridSearchCV(forest_reg,  param_grid,	cv=5,
                        scoring='neg_mean_squared_error')
    grid_search.fit(housing_prepared, housing_labels)
#   print(grid_search)
#   print(grid_search.best_params_ )
    
    cvres=grid_search.cv_results_ 
    for	mean_score,	params in zip(cvres["mean_test_score"],	cvres["params"]):
        print(np.sqrt(-mean_score),	params)
#### 随机搜索 RandomizedSearchCV
        
#### 集成方法
        
####  分析最佳模型和它们的误差
        
#### 用测试集评估系统

    print("_________end__________________")
    
