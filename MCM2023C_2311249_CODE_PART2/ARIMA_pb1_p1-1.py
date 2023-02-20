import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
import seaborn as sns
import datetime

# 读取数据
data = pd.read_excel('Problem_C_Data_Wordle.xlsx')
index = data.iloc[1:,2]
results_number = data.iloc[1:,4]

problem = pd.concat([index, results_number], join="outer", axis=1)
problem.columns = list(['index','results_number'])
problem.index = index - 202
problem = problem.iloc[::-1]

# data clean
for i in range(1,357):
    reasonable_average = (problem.iloc[i-1][1]+problem.iloc[i+1][1])/2
    if problem.iloc[i][1] > reasonable_average*2 or problem.iloc[i][1] < reasonable_average*0.5:
        print(i)
        problem.iloc[i][1] = reasonable_average

problem['results_number'].plot(figsize=(10,6))
plt.xticks(rotation=30)
plt.show()

from statsmodels.tsa.stattools import adfuller

# 单位根检验
result = adfuller(problem['results_number'])
if result[1] > 0.05:
    print('概率p值为：%s，原始序列不平稳' % (result[1]))
else:
    print('概率p值为：%s，原始序列平稳' % (result[1]))
    
problem_diff_0 = pd.DataFrame(problem['results_number'])

# 一阶差分
problem_diff_1 = problem['results_number'].diff(1).dropna()
problem_diff_1.plot(figsize=(14,6))

# 一阶差分序列单位根检验
result = adfuller(problem_diff_1)
if result[1] > 0.05:
    print('概率p值为：%s，一阶差分序列不平稳' % (result[1]))
else:
    print('概率p值为：%s，一阶差分序列平稳' % (result[1]))
    
# 二阶差分
problem_diff_2 = problem['results_number'].diff(2).dropna()
problem_diff_2.plot(figsize=(14,6))


# 一阶差分序列单位根检验
result = adfuller(problem_diff_2)
if result[1] > 0.05:
    print('概率p值为：%s，二阶差分序列不平稳' % (result[1]))
else:
    print('概率p值为：%s，二阶差分序列平稳' % (result[1]))

from statsmodels.stats.diagnostic import acorr_ljungbox as lb_test

# LB检验
p = pd.DataFrame(lb_test(problem_diff_1, lags=1)).iloc[1, 0]
                 
if p < 0.05:
    print(u'一阶差分序列为非白噪声序列，p值为:%s' % p)
else:
    print(u'一阶差分序列为白噪声序列，p值为:%s' % p)
    
    
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

fig=plt.figure(figsize=(12,5))
ax1=fig.add_subplot(1,2,1)
ax2=fig.add_subplot(1,2,2)
plot_acf(problem_diff_1,ax=ax1,lags=13)
plot_pacf(problem_diff_1,ax=ax2,lags=13)
plt.tight_layout()
plt.show()

from statsmodels.tsa.api import ARIMA

#对模型进行定阶
pmax = int(len(problem_diff_1) / 150)    #一般阶数不超过 length /10
qmax = int(len(problem_diff_1) / 150)
aic_matrix = []
for p in range(pmax +1):
    temp= []
    for q in range(qmax+1):
        try:
            temp.append(ARIMA(problem['results_number'].tolist(), (p, 1, q)).fit().aic)
        except:
            temp.append(None)
    aic_matrix.append(temp)

aic_matrix = pd.DataFrame(aic_matrix)   #将其转换成Dataframe 数据结构
p,q = aic_matrix.stack().idxmin()   #先使用stack 展平， 然后使用 idxmin 找出最小值的位置
print(u'BIC 最小的p值 和 q 值：%s,%s' %(p,q))  #  BIC 最小的p值 和 q 值：0,1
#所以可以建立ARIMA 模型，ARIMA(1,1,0)

fig, ax = plt.subplots(figsize=(10, 8))
ax = sns.heatmap(aic_matrix,
                 mask=aic_matrix.isnull(),
                 ax=ax,
                 annot=True,
                 fmt='.2f')
ax.set_title('AIC')


model1 = ARIMA(problem['results_number'].tolist(),order=(1, 1, 0)).fit()
print(model1.summary2())
