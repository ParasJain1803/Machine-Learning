Python 3.10.11 (tags/v3.10.11:7d4cc5a, Apr  5 2023, 00:38:17) [MSC v.1929 64 bit (AMD64)] on win32
Type "help", "copyright", "credits" or "license()" for more information.
import pandas as pd
import numpy as np
import matplotlib as mlt
df = pd.read_csv('D:\\Paras\\college\\Machine \Learning\\Projects\\01_housing\\housing.csv')
Traceback (most recent call last):
  File "<pyshell#3>", line 1, in <module>
    df = pd.read_csv('D:\\Paras\\college\\Machine \Learning\\Projects\\01_housing\\housing.csv')
  File "C:\Users\lenovo12\AppData\Local\Programs\Python\Python310\lib\site-packages\pandas\io\parsers\readers.py", line 1026, in read_csv
    return _read(filepath_or_buffer, kwds)
  File "C:\Users\lenovo12\AppData\Local\Programs\Python\Python310\lib\site-packages\pandas\io\parsers\readers.py", line 620, in _read
    parser = TextFileReader(filepath_or_buffer, **kwds)
  File "C:\Users\lenovo12\AppData\Local\Programs\Python\Python310\lib\site-packages\pandas\io\parsers\readers.py", line 1620, in __init__
    self._engine = self._make_engine(f, self.engine)
  File "C:\Users\lenovo12\AppData\Local\Programs\Python\Python310\lib\site-packages\pandas\io\parsers\readers.py", line 1880, in _make_engine
    self.handles = get_handle(
  File "C:\Users\lenovo12\AppData\Local\Programs\Python\Python310\lib\site-packages\pandas\io\common.py", line 873, in get_handle
    handle = open(
FileNotFoundError: [Errno 2] No such file or directory: 'D:\\Paras\\college\\Machine \\Learning\\Projects\\01_housing\\housing.csv'
df = pd.read_csv('D:\\Paras\\college\\Machine \\Learning\\Projects\\01_housing\\housing.csv')
Traceback (most recent call last):
  File "<pyshell#4>", line 1, in <module>
    df = pd.read_csv('D:\\Paras\\college\\Machine \\Learning\\Projects\\01_housing\\housing.csv')
  File "C:\Users\lenovo12\AppData\Local\Programs\Python\Python310\lib\site-packages\pandas\io\parsers\readers.py", line 1026, in read_csv
    return _read(filepath_or_buffer, kwds)
  File "C:\Users\lenovo12\AppData\Local\Programs\Python\Python310\lib\site-packages\pandas\io\parsers\readers.py", line 620, in _read
    parser = TextFileReader(filepath_or_buffer, **kwds)
  File "C:\Users\lenovo12\AppData\Local\Programs\Python\Python310\lib\site-packages\pandas\io\parsers\readers.py", line 1620, in __init__
    self._engine = self._make_engine(f, self.engine)
  File "C:\Users\lenovo12\AppData\Local\Programs\Python\Python310\lib\site-packages\pandas\io\parsers\readers.py", line 1880, in _make_engine
    self.handles = get_handle(
  File "C:\Users\lenovo12\AppData\Local\Programs\Python\Python310\lib\site-packages\pandas\io\common.py", line 873, in get_handle
    handle = open(
FileNotFoundError: [Errno 2] No such file or directory: 'D:\\Paras\\college\\Machine \\Learning\\Projects\\01_housing\\housing.csv'
df = pd.read_csv('D:\\Paras\\college\\Machine Learning\\Projects\\01_housing\\housing.csv')
clear
Traceback (most recent call last):
  File "<pyshell#6>", line 1, in <module>
    clear
NameError: name 'clear' is not defined
df.columns
Index(['longitude', 'latitude', 'housing_median_age', 'total_rooms',
       'total_bedrooms', 'population', 'households', 'median_income',
       'median_house_value', 'ocean_proximity'],
      dtype='object')
x = df[['longitude', 'latitude']]
x.shape
(20640, 2)
type(x)
<class 'pandas.core.frame.DataFrame'>
y = df[['median_house_value']]
x.info()
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 20640 entries, 0 to 20639
Data columns (total 2 columns):
 #   Column     Non-Null Count  Dtype  
---  ------     --------------  -----  
 0   longitude  20640 non-null  float64
 1   latitude   20640 non-null  float64
dtypes: float64(2)
memory usage: 322.6 KB
from sklearn.linear_model import LinearRegression
model_1 = LinearRegression()
model_1.fit(x,y)
LinearRegression()
pred_1 = model_1.predict(x)
from sklearn.metrics import root_mean_squared_error
error_model_1 = root_mean_squared_error(pred_1, y)
error_model_1
100436.86257963683
df.columns
Index(['longitude', 'latitude', 'housing_median_age', 'total_rooms',
       'total_bedrooms', 'population', 'households', 'median_income',
       'median_house_value', 'ocean_proximity'],
      dtype='object')
model_2 = LinearRegression()
x_age = df[['housing_median_age']]
model_2.fit(x_age,y)
LinearRegression()
pred_2 = model_2.predict(x_age)
error_model_2 = root_mean_squared_error(pred_2, y)
error_model_2
114747.3362852539
model_3 = LinearRegression()
x_income = df[['median_income']]
model_3.fit(x_income,y)
LinearRegression()
pred_3 = model_3.predict(x_income)
error_model_3 = root_mean_squared_error(pred_3, y)
error_model_3
83733.57452616918
model_4 = LinearRegression()
x4 = df[['longitude', 'latitude,median_income']]
Traceback (most recent call last):
  File "<pyshell#34>", line 1, in <module>
    x4 = df[['longitude', 'latitude,median_income']]
  File "C:\Users\lenovo12\AppData\Local\Programs\Python\Python310\lib\site-packages\pandas\core\frame.py", line 4096, in __getitem__
    indexer = self.columns._get_indexer_strict(key, "columns")[1]
  File "C:\Users\lenovo12\AppData\Local\Programs\Python\Python310\lib\site-packages\pandas\core\indexes\base.py", line 6200, in _get_indexer_strict
    self._raise_if_missing(keyarr, indexer, axis_name)
  File "C:\Users\lenovo12\AppData\Local\Programs\Python\Python310\lib\site-packages\pandas\core\indexes\base.py", line 6252, in _raise_if_missing
    raise KeyError(f"{not_found} not in index")
KeyError: "['latitude,median_income'] not in index"
x4 = df[['longitude', 'latitude','median_income']]
model_4.fit(x4,y)
LinearRegression()
pred_4 = model_4.predict(x4)
model_3.fit(x_income,y))
SyntaxError: unmatched ')'
error_model_4 = root_mean_squared_error(pred_4, y)
error_model_4
74402.93701662727
cor_rel = x.corr()
x_num = df.drop('ocean_proximity', axis = 1)
cor_rel = x_num.corr()
cor_rel['median_house_value'].sort_values(ascending=False)
median_house_value    1.000000
median_income         0.688075
total_rooms           0.134153
housing_median_age    0.105623
households            0.065843
total_bedrooms        0.049686
population           -0.024650
longitude            -0.045967
latitude             -0.144160
Name: median_house_value, dtype: float64
cor_rel['median_house_value'].sort_values(ascending=False)
median_house_value    1.000000
median_income         0.688075
total_rooms           0.134153
housing_median_age    0.105623
households            0.065843
total_bedrooms        0.049686
population           -0.024650
longitude            -0.045967
latitude             -0.144160
Name: median_house_value, dtype: float64
cor_rel['median_house_value'].sort_values(ascending=False)
median_house_value    1.000000
median_income         0.688075
total_rooms           0.134153
housing_median_age    0.105623
households            0.065843
total_bedrooms        0.049686
population           -0.024650
longitude            -0.045967
latitude             -0.144160
Name: median_house_value, dtype: float64
# NaN values
# categorical value # NaN values imputation
x_num.isna()
       longitude  latitude  ...  median_income  median_house_value
0          False     False  ...          False               False
1          False     False  ...          False               False
2          False     False  ...          False               False
3          False     False  ...          False               False
4          False     False  ...          False               False
...          ...       ...  ...            ...                 ...
20635      False     False  ...          False               False
20636      False     False  ...          False               False
20637      False     False  ...          False               False
20638      False     False  ...          False               False
20639      False     False  ...          False               False

[20640 rows x 9 columns]
x_num.isna().sum()
longitude               0
latitude                0
housing_median_age      0
total_rooms             0
total_bedrooms        207
population              0
households              0
median_income           0
median_house_value      0
dtype: int64
>>> df['ocean-proximity'].value_counts()
Traceback (most recent call last):
  File "C:\Users\lenovo12\AppData\Local\Programs\Python\Python310\lib\site-packages\pandas\core\indexes\base.py", line 3805, in get_loc
    return self._engine.get_loc(casted_key)
  File "index.pyx", line 167, in pandas._libs.index.IndexEngine.get_loc
  File "index.pyx", line 196, in pandas._libs.index.IndexEngine.get_loc
  File "pandas\\_libs\\hashtable_class_helper.pxi", line 7081, in pandas._libs.hashtable.PyObjectHashTable.get_item
  File "pandas\\_libs\\hashtable_class_helper.pxi", line 7089, in pandas._libs.hashtable.PyObjectHashTable.get_item
KeyError: 'ocean-proximity'

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "<pyshell#51>", line 1, in <module>
    df['ocean-proximity'].value_counts()
  File "C:\Users\lenovo12\AppData\Local\Programs\Python\Python310\lib\site-packages\pandas\core\frame.py", line 4090, in __getitem__
    indexer = self.columns.get_loc(key)
  File "C:\Users\lenovo12\AppData\Local\Programs\Python\Python310\lib\site-packages\pandas\core\indexes\base.py", line 3812, in get_loc
    raise KeyError(key) from err
KeyError: 'ocean-proximity'
>>> df['ocean_proximity'].value_counts()
ocean_proximity
<1H OCEAN     9136
INLAND        6551
NEAR OCEAN    2658
NEAR BAY      2290
ISLAND           5
Name: count, dtype: int64
