import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.linear_model import Ridge
from sklearn.datasets import load_boston
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

#2 数据载入
df = pd.read_csv('reg_data_final_ridge.csv')
#df = pd.read_csv('pca_word_pb3.csv')
df.columns = ['index', 'word', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'y1', 'y2']
cols = ['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8']
#cols = ['x1', 'x2', 'x4', 'x6', 'x7', 'x8'] # After pruning

# 划分80%的数据为训练集
train_df = df.sample(frac=0.8)

# 剩下20%作为测试集
test_df = df[~df.index.isin(train_df.index)]

# Ridge

#2 数据集的划分
x_train = train_df[cols]
y_train = train_df.iloc[:,11]
x_test = test_df[cols]
y_test = test_df.iloc[:,11]
y_test_class = test_df.iloc[:,10]
y_test_basic = test_df.iloc[:,0:2]

#3 特征工程 - 标准化
transfer = StandardScaler()
#x_train = transfer.fit_transform(x_train)
#x_test = transfer.fit_transform(x_test)

x_train = sm.add_constant(train_df[cols])#生成自变量
x_test = sm.add_constant(test_df[cols])

#4 岭回归
model = Ridge(alpha=1) # 法1
# estimator = RidgeCV(alphas=(0.1, 1, 10))  # 法2
model.fit(x_train, y_train)

# 5 模型评估
# 5.1 获取系数等值
y_predict = model.predict(x_test)
print("y_predict:\n", y_predict)
print("coef:\n", model.coef_)
print("intercept:\n", model.intercept_)

# 5.2 评价
# 均方误差
MSE = mean_squared_error(y_test, y_predict)
print("MSE:\n", MSE)

y_predict = pd.DataFrame(y_predict) 
y_predict.columns = ['class_predict']
#boundry_2_3 = y_predict['class_predict'].nlargest(8).iloc[-1]
#boundry_1_2 = y_predict['class_predict'].nlargest(35).iloc[-1]
#boundry_0_1 = y_predict['class_predict'].nlargest(55).iloc[-1]
boundry_2_3 = 3.19
boundry_1_2 = 2.755
boundry_0_1 = 2.365

for word in range(72):
    if y_predict.iloc[word][0] < boundry_0_1:
        y_predict.iat[word,0] = 0
    elif y_predict.iloc[word][0] < boundry_1_2:
        y_predict.iat[word,0] = 1
    elif y_predict.iloc[word][0] < boundry_2_3:
        y_predict.iat[word,0] = 2
    else:
        y_predict.iat[word,0] = 3
           
class_result = pd.concat([y_test_basic, y_test_class], join="outer", axis=1)
y_predict.index = class_result.index
class_result = pd.concat([class_result, y_predict], join="outer", axis=1)
class_result.columns = ['index','word','class_true','class_predict']

accu_count = 0
for i in range(72):
    if class_result.iloc[i]['class_true'] == class_result.iloc[i]['class_predict']:
        accu_count = accu_count+1
        
print(accu_count)

# 10 Tests:
# 36 35 37 42 32 33 33 31 40 33
# mean accuracy: 35.2/72 = 48.9%

# 10 Tests（After pruning）:
# 33 31 31 32 29 37 33 41 30 36
# mean accuracy: 33.3/72 = 46.3%    

# OLS

model_ols = sm.OLS(y_train, x_train) #生成模型
result = model_ols.fit() #模型拟合
#print(result1.summary()) #模型描述


def looper(limit,label):
    cols = ['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8']
    for i in range(len(cols)):
        data1 = train_df[cols]
        x = sm.add_constant(data1) #生成自变量
        y = train_df[label] #生成因变量
        model = sm.OLS(y, x) #生成模型
        result = model.fit() #模型拟合
        pvalues = result.pvalues #得到结果中所有P值
        pvalues.drop('const',inplace=True) #把const取得
        pmax = max(pvalues) #选出最大的P值
        if pmax>limit:
            ind = pvalues.idxmax() #找出最大P值的index
            cols.remove(ind) #把这个index从cols中删除
        else:
            return result,cols
 
result,cols = looper(0.05,'y1')
print(result.summary())   
