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

#验证
problem['results_number'].plot(figsize=(10,6))
plt.xticks(rotation=30)
plt.show()

problem_diff_1 = problem['results_number'].diff(1).dropna()

output1 = []
output1.append(problem_diff_1[1])
for i in range(2,358):
    output1.append(-168.296-0.362*problem_diff_1[i-1])

output = []
output.append(problem['results_number'][0])
for i in range(1,358):
    output.append(problem['results_number'][i-1]+output1[i-1])
 
output = pd.DataFrame(output)

predict1 = pd.concat([problem['results_number'], output],
                          axis=1,
                          keys=['original', 'predicted'])

fig=plt.figure(figsize=(16,5))
ax1=fig.add_subplot(1,2,2)
predict1.plot(ax=ax1,title='model1: Original vs predicted')
for xtick in ax1.get_xticklabels():
    xtick.set_rotation(30)
plt.xticks(rotation=30)
plt.show()

#预测
problem['results_number'].plot(figsize=(10,6))
plt.xticks(rotation=30)
plt.show()

problem_diff_1 = problem['results_number'].diff(1).dropna()

output1 = []
output1.append(problem_diff_1[1])
for i in range(2,358):
    output1.append(-168.296-0.362*problem_diff_1[i-1])

output = []
output.append(problem['results_number'][0])
for i in range(1,358):
    output.append(problem['results_number'][i-1]+output1[i-1])

for i in range(358,419):
    output1.append(-168.296-0.362*output1[i-2])

for i in range(358,419):
    output.append(output[i-2]+output1[i-2])
 
output = pd.DataFrame(output)

para1_max = (-168.296+747.07)/2
para2_max = (-0.362-0.266)/2

output1_max = []
output1_max.append(problem_diff_1[1])
for i in range(2,358):
    output1_max.append(para1_max+para2_max*problem_diff_1[i-1])

output_max = []
output_max.append(problem['results_number'][0])
for i in range(1,358):
    output_max.append(problem['results_number'][i-1]+output1_max[i-1])

for i in range(358,419):
    output1_max.append(para1_max+para2_max*output1_max[i-2])

for i in range(358,419):
    output_max.append(output_max[i-2]+output1_max[i-2])
 
output_max = pd.DataFrame(output_max)

para1_min = (-168.296-1083.66)/2
para2_min = (-0.362-0.459)/2

output1_min = []
output1_min.append(problem_diff_1[1])
for i in range(2,358):
    output1_min.append(para1_min+para2_min*problem_diff_1[i-1])

output_min = []
output_min.append(problem['results_number'][0])
for i in range(1,358):
    output_min.append(problem['results_number'][i-1]+output1_min[i-1])

for i in range(358,419):
    output1_min.append(para1_min+para2_min*output1_min[i-2])

for i in range(358,419):
    output_min.append(output_min[i-2]+output1_min[i-2])
 
output_min = pd.DataFrame(output_min)

predict1 = pd.concat([problem['results_number'], output, output_max, output_min],
                          axis=1,
                          keys=['original', 'predicted', 'predicted_max', 'predicted_min'])

fig=plt.figure(figsize=(16,5))
ax1=fig.add_subplot(1,2,2)
predict1.plot(ax=ax1,title='model1: Original vs predicted')
for xtick in ax1.get_xticklabels():
    xtick.set_rotation(30)
plt.xticks(rotation=30)
plt.show()
