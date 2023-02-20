import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.linear_model import Lasso,LassoCV
from sklearn.metrics import mean_squared_error
 
df = pd.read_csv('pca_word.csv')
df.columns = ['index', 'word', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'y1', 'y2']
cols = ['x1', 'x2', 'x3', 'x4', 'x5', 'x6']
pbC_data = pd.read_excel('Problem_C_Data_Wordle.xlsx')
percentage_true = pbC_data.iloc[1:,6:]
percentage_true.index = percentage_true.index-1

# 划分80%的数据为训练集
train_df = df.sample(frac=0.8)
train_percentage_true = percentage_true[percentage_true.index.isin(train_df.index)]

# 剩下20%作为测试集
test_df = df[~df.index.isin(train_df.index)]
test_percentage_true = percentage_true[~percentage_true.index.isin(train_df.index)]


# LASSO

test_y1_true = test_df['y1']
test_y2_true = test_df['y2']
test_word = test_df['word']

x_train = sm.add_constant(train_df[cols])#生成自变量
x_test = sm.add_constant(test_df[cols])

y1_train = train_df['y1'] #生成因变量
y2_train = train_df['y2'] #生成因变量

#构造不同的lambda值
Lambdas = np.logspace(-5,2,200)
#设置交叉验证的参数，使用均方误差评估
lasso_cv1 = LassoCV(alphas=Lambdas,normalize=True,cv=10,max_iter=10000)
lasso_cv1.fit(x_train,y1_train)
print(lasso_cv1.alpha_)
 
#基于最佳lambda值建模
model1_lasso = Lasso(alpha=lasso_cv1.alpha_,normalize=True,max_iter=10000)
model1_lasso.fit(x_train,y1_train)
#打印回归系数
print(pd.Series(index=['Intercept']+x_train.columns.tolist(),
                data=[model1_lasso.intercept_]+model1_lasso.coef_.tolist()))
 
#模型评估
test_y1_predict_lasso = model1_lasso.predict(x_test)
#均方误差
MSE1_LASSO = mean_squared_error(test_y1_true,test_y1_predict_lasso)

#构造不同的lambda值
Lambdas = np.logspace(-5,2,200)
#设置交叉验证的参数，使用均方误差评估
lasso_cv2 = LassoCV(alphas=Lambdas,normalize=True,cv=10,max_iter=10000)
lasso_cv2.fit(x_train,y2_train)
print(lasso_cv2.alpha_)
 
#基于最佳lambda值建模
model2_lasso = Lasso(alpha=lasso_cv2.alpha_,normalize=True,max_iter=10000)
model2_lasso.fit(x_train,y2_train)
#打印回归系数
print(pd.Series(index=['Intercept']+x_train.columns.tolist(),
                data=[model2_lasso.intercept_]+model2_lasso.coef_.tolist()))
 
#模型评估
test_y2_predict_lasso = model2_lasso.predict(x_test)
#均方误差
MSE2_LASSO = mean_squared_error(test_y2_true,test_y2_predict_lasso)

test_y1_predict_lasso = pd.DataFrame(test_y1_predict_lasso)
test_y2_predict_lasso = pd.DataFrame(test_y2_predict_lasso)

test_y1_predict_lasso.index = test_y1_true.index
test_y2_predict_lasso.index = test_y2_true.index

test_y1 = pd.concat([test_y1_true, test_y1_predict_lasso], join="outer", axis=1)
test_y2 = pd.concat([test_y2_true, test_y2_predict_lasso], join="outer", axis=1)


# OLS

model1_ols = sm.OLS(y1_train, x_train) #生成模型
result1 = model1_ols.fit() #模型拟合
#print(result1.summary()) #模型描述

model2_ols = sm.OLS(y2_train, x_train) #生成模型
result2 = model2_ols.fit() #模型拟合
#print(result2.summary()) #模型描述


def looper(limit,label):
    cols = ['x1', 'x2', 'x3', 'x4', 'x5', 'x6']
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
 
result1,cols1 = looper(0.05,'y1')
print(result1.summary())

result2,cols2 = looper(0.05,'y2')
print(result2.summary())

print(result1.params)
print(result2.params)

print(cols1,cols2)

test_data1 = test_df[cols1]
test_x1 = sm.add_constant(test_data1) #生成自变量

test_data2 = test_df[cols2]
test_x2 = sm.add_constant(test_data2) #生成自变量

test_y1_predict_ols = np.dot(test_x1,result1.params)
test_y2_predict_ols = np.dot(test_x2,result2.params)

test_y1_predict_ols = pd.DataFrame(test_y1_predict_ols)
test_y2_predict_ols = pd.DataFrame(test_y2_predict_ols)

test_y1_predict_ols.index = test_y1_true.index
test_y2_predict_ols.index = test_y2_true.index

test_y1 = pd.concat([test_y1, test_y1_predict_ols], join="outer", axis=1)
test_y2 = pd.concat([test_y2, test_y2_predict_ols], join="outer", axis=1)

MSE1_OLS = mean_squared_error(test_y1_true,test_y1_predict_ols)
MSE2_OLS = mean_squared_error(test_y2_true,test_y2_predict_ols)

print('MSE1(LASSO):',MSE1_LASSO)
print('MSE2(LASSO):',MSE2_LASSO)
print('MSE1(OLS):',MSE1_OLS)
print('MSE2(OLS):',MSE2_OLS)

test_index = pd.DataFrame(test_y2_true.index)

percentage = np.empty([72,9], dtype = int) 
percentage = pd.DataFrame(percentage)
percentage = pd.concat([test_index, percentage], join="outer", axis=1)
# 8:SUM  9:SSE
percentage.columns = ['Number',1,2,3,4,5,6,7,8,9]


for index in range(0,72):
    percentage.iat[index,8] = 0
    for try_time in range(1,8):
        percentage.iat[index,try_time] = 100*np.exp(-(try_time - test_y1_predict_lasso.iloc[index][0]) ** 2 / (2 * test_y2_predict_lasso.iloc[index][0] ** 2)) / (np.sqrt(2 * np.pi) * test_y2_predict_lasso.iloc[index][0])
        percentage.iat[index,8] = percentage.iloc[index][8]+percentage.iloc[index][try_time]
   
for index in range(0,72):
    for try_time in range(1,8):
        percentage.iat[index,try_time] = 100/percentage.iat[index,8]*percentage.iloc[index][try_time]
   
for index in range(0,72):
    percentage.iat[index,8] = 0
    for try_time in range(1,8):
        percentage.iat[index,8] = percentage.iloc[index][8]+percentage.iloc[index][try_time]

test_percentage_true.index = percentage.index
test_percentage_true = pd.concat([test_index, test_percentage_true], join="outer", axis=1)
test_percentage_true.columns = ['Number',1,2,3,4,5,6,7]

for index in range(0,72):
    percentage.iat[index,9] = 0
    for try_time in range(1,8):
        percentage.iat[index,9] = percentage.iat[index,9]+(percentage.iloc[index][try_time]-test_percentage_true.iloc[index][try_time])**2

test_word.index = percentage.index
percentage = pd.concat([test_word, percentage], join="outer", axis=1)
percentage.columns = ['Word','Number','1 try','2 tries','3 tries','4 tries','5 tries','6 tries','7 or more tries(X)','Sum','SSE']
print('SSE mean:',percentage['SSE'].mean())
print('SSE median:',percentage['SSE'].median())


# predict the distribution of the word EERIE on 2023/03/01

EERIE_pca_vector = [-0.04917196,  0.20248792, -0.66087095,  0.66808991,  0.654808  ,0.27586237]
EERIE_pca_vector = pd.DataFrame(EERIE_pca_vector).T
EERIE_pca_vector.columns = ['x1', 'x2', 'x3', 'x4', 'x5', 'x6']
bias = np.empty([1,1], dtype = int) 
bias = pd.DataFrame(bias)
bias.iat[0,0] = 1
EERIE_x = pd.concat([bias, EERIE_pca_vector], join="outer", axis=1)
    
EERIE_mean_predict_lasso = model1_lasso.predict(EERIE_x)
EERIE_variance_predict_lasso = model2_lasso.predict(EERIE_x)

EERIE_per = np.empty([1,8], dtype = int) 
EERIE_per = pd.DataFrame(EERIE_per)
EERIE_number = np.empty([1,1], dtype = int) 
EERIE_number = pd.DataFrame(EERIE_number)
EERIE_number.iat[0,0] = 420
EERIE_per = pd.concat([EERIE_number, EERIE_per], join="outer", axis=1)
# 8:SUM  
EERIE_per.columns = ['Number',1,2,3,4,5,6,7,8]

EERIE_per.iat[0,8] = 0
for try_time in range(1,8):
    EERIE_per.iat[0,try_time] = 100*np.exp(-(try_time - EERIE_mean_predict_lasso) ** 2 / (2 * EERIE_variance_predict_lasso ** 2)) / (np.sqrt(2 * np.pi) * EERIE_variance_predict_lasso)
    EERIE_per.iat[0,8] = EERIE_per.iloc[0][8]+EERIE_per.iloc[0][try_time]
   
for try_time in range(1,8):
    EERIE_per.iat[0,try_time] = 100/EERIE_per.iat[0,8]*EERIE_per.iloc[0][try_time]
   
EERIE_per.iat[0,8] = 0
for try_time in range(1,8):
    EERIE_per.iat[0,8] = EERIE_per.iloc[0][8]+EERIE_per.iloc[0][try_time]
    
EERIE_word = np.empty([1,1], dtype = int) 
EERIE_word = pd.DataFrame(EERIE_word)
EERIE_word.iat[0,0] = 'eerie'
EERIE_per = pd.concat([EERIE_word, EERIE_per], join="outer", axis=1)
EERIE_per.columns = ['Word','Number','1 try','2 tries','3 tries','4 tries','5 tries','6 tries','7 or more tries(X)','Sum']

print(EERIE_per)

print('SSE mean:',percentage['SSE'].mean())
print('SSE median:',percentage['SSE'].median())
