import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC

# 0 vs others
df0 = pd.read_csv('reg_data_final_0.csv',delimiter=',')

train_df0 = df0.sample(frac=0.8)
# Create a SVM classifier using a soft margin

test_df0 = df0[~df0.index.isin(train_df0.index)]
test_df0_x = test_df0.iloc[:,2:10]
test_df0_y = test_df0.iloc[:,10]

train_df0_target_1 = train_df0[train_df0.difficultyLevel==1]
#train_df0_target_0 = train_df0[train_df0.difficultyLevel==0]

#print(len(train_df_target_1))
train_df0_target_1.index = np.linspace(len(df0),len(df0)+len(train_df0_target_1)-1,len(train_df0_target_1))
train_df0 = train_df0.append(train_df0_target_1)

train_df0_target_1.index = np.linspace(len(df0)+len(train_df0_target_1),len(df0)+2*len(train_df0_target_1)-1,len(train_df0_target_1))
train_df0 = train_df0.append(train_df0_target_1)

train_df0_target_1.index = np.linspace(len(df0)+len(train_df0_target_1)*2,len(df0)+3*len(train_df0_target_1)-1,len(train_df0_target_1))
train_df0 = train_df0.append(train_df0_target_1)

#print(train_df)

train_df0_x = train_df0.iloc[:,2:10]
train_df0_y = train_df0.iloc[:,10]

model_df0= SVC(kernel='linear', C=1.0, decision_function_shape='ovo')
# Train the model using the training sets
model_df0.fit(train_df0_x, train_df0_y)
#print(clf1.score(train_df0_x, train_df0_y))
#print(clf1.score(test_df0_x, test_df0_y))
y_predict_scores_df0 = model_df0.decision_function(test_df0_x)
y_predict_scores_df0 = pd.DataFrame(y_predict_scores_df0)


# 1 vs others
df1 = pd.read_csv('reg_data_final_1.csv',delimiter=',')

train_df1 = df1[df0.index.isin(train_df0.index)]
# Create a SVM classifier using a soft margin

test_df1 = df1[~df1.index.isin(train_df1.index)]
test_df1_x = test_df1.iloc[:,2:10]
test_df1_y = test_df1.iloc[:,10]

train_df1_target_1 = train_df1[train_df1.difficultyLevel==1]

#print(len(train_df_target_1))
#train_df1_target_1.index = np.linspace(len(df1),len(df1)+len(train_df1_target_1)-1,len(train_df1_target_1))
#train_df1 = train_df1.append(train_df1_target_1)

#train_df1_target_1.index = np.linspace(len(df1)+len(train_df1_target_1),len(df1)+2*len(train_df1_target_1)-1,len(train_df1_target_1))
#train_df1 = train_df1.append(train_df1_target_1)

#train_df1_target_1.index = np.linspace(len(df1)+len(train_df1_target_1)*2,len(df1)+3*len(train_df1_target_1)-1,len(train_df1_target_1))
#train_df1 = train_df1.append(train_df1_target_1)

#print(train_df)

train_df1_x = train_df1.iloc[:,2:10]
train_df1_y = train_df1.iloc[:,10]

model_df1= SVC(kernel='linear', C=1.0, decision_function_shape='ovo')
# Train the model using the training sets
model_df1.fit(train_df1_x, train_df1_y)
#print(clf1.score(train_df1_x, train_df1_y))
#print(clf1.score(test_df1_x, test_df1_y))
y_predict_scores_df1 = model_df1.decision_function(test_df1_x)
y_predict_scores_df1 = pd.DataFrame(y_predict_scores_df1)

# Predict the response for test dataset
#test_df_y_pred = clf.predict(test_df_x)
#test_df_y_pred = pd.DataFrame(test_df_y_pred)


# 2 vs others
df2 = pd.read_csv('reg_data_final_2.csv',delimiter=',')

train_df2 = df2[df0.index.isin(train_df0.index)]
# Create a SVM classifier using a soft margin

test_df2 = df2[~df2.index.isin(train_df2.index)]
test_df2_x = test_df2.iloc[:,2:10]
test_df2_y = test_df2.iloc[:,10]

train_df2_target_1 = train_df2[train_df2.difficultyLevel==1]

#print(len(train_df_target_1))
train_df2_target_1.index = np.linspace(len(df2),len(df2)+len(train_df2_target_1)-1,len(train_df2_target_1))
train_df2 = train_df2.append(train_df2_target_1)

train_df2_target_1.index = np.linspace(len(df2)+len(train_df2_target_1),len(df2)+2*len(train_df2_target_1)-1,len(train_df2_target_1))
train_df2 = train_df2.append(train_df2_target_1)

#train_df2_target_1.index = np.linspace(len(df2)+len(train_df2_target_1)*2,len(df2)+3*len(train_df2_target_1)-1,len(train_df2_target_1))
#train_df2 = train_df2.append(train_df2_target_1)

#print(train_df)

train_df2_x = train_df2.iloc[:,2:10]
train_df2_y = train_df2.iloc[:,10]

model_df2= SVC(kernel='linear', C=1.0, decision_function_shape='ovo')
# Train the model using the training sets
model_df2.fit(train_df2_x, train_df2_y)
#print(clf1.score(train_df2_x, train_df2_y))
#print(clf1.score(test_df2_x, test_df2_y))
y_predict_scores_df2 = model_df2.decision_function(test_df2_x)
y_predict_scores_df2 = pd.DataFrame(y_predict_scores_df2)


# 3 vs others
df3= pd.read_csv('reg_data_final_3.csv',delimiter=',')

train_df3= df3[df0.index.isin(train_df0.index)]
# Create a SVM classifier using a soft margin

test_df3= df3[~df3.index.isin(train_df3.index)]
test_df3_x = test_df3.iloc[:,2:10]
test_df3_y = test_df3.iloc[:,10]

train_df3_target_1 = train_df3[train_df3.difficultyLevel==1]

#print(len(train_df_target_1))
train_df3_target_1.index = np.linspace(len(df3),len(df3)+len(train_df3_target_1)-1,len(train_df3_target_1))
train_df3= train_df3.append(train_df3_target_1)

train_df3_target_1.index = np.linspace(len(df3)+len(train_df3_target_1),len(df3)+2*len(train_df3_target_1)-1,len(train_df3_target_1))
train_df3= train_df3.append(train_df3_target_1)

train_df3_target_1.index = np.linspace(len(df3)+len(train_df3_target_1)*2,len(df3)+3*len(train_df3_target_1)-1,len(train_df3_target_1))
train_df3= train_df3.append(train_df3_target_1)

train_df3_target_1.index = np.linspace(len(df3)+len(train_df3_target_1)*3,len(df3)+4*len(train_df3_target_1)-1,len(train_df3_target_1))
train_df3= train_df3.append(train_df3_target_1)

train_df3_target_1.index = np.linspace(len(df3)+len(train_df3_target_1)*4,len(df3)+5*len(train_df3_target_1)-1,len(train_df3_target_1))
train_df3= train_df3.append(train_df3_target_1)

train_df3_target_1.index = np.linspace(len(df3)+len(train_df3_target_1)*5,len(df3)+6*len(train_df3_target_1)-1,len(train_df3_target_1))
train_df3= train_df3.append(train_df3_target_1)

train_df3_target_1.index = np.linspace(len(df3)+len(train_df3_target_1)*6,len(df3)+7*len(train_df3_target_1)-1,len(train_df3_target_1))
train_df3= train_df3.append(train_df3_target_1)

#train_df3_target_1.index = np.linspace(len(df3)+len(train_df3_target_1)*7,len(df3)+8*len(train_df3_target_1)-1,len(train_df3_target_1))
#train_df3= train_df3.append(train_df3_target_1)

#print(train_df)

train_df3_x = train_df3.iloc[:,2:10]
train_df3_y = train_df3.iloc[:,10]

model_df3= SVC(kernel='linear', C=1.0, decision_function_shape='ovo')
# Train the model using the training sets
model_df3.fit(train_df3_x, train_df3_y)
#print(clf1.score(train_df3_x, train_df3_y))
#print(clf1.score(test_df3_x, test_df3_y))
y_predict_scores_df3= model_df3.decision_function(test_df3_x)
y_predict_scores_df3= pd.DataFrame(y_predict_scores_df3)

y_predict_scores = pd.concat([y_predict_scores_df0, y_predict_scores_df1], join="outer", axis=1)
y_predict_scores = pd.concat([y_predict_scores, y_predict_scores_df2], join="outer", axis=1)
y_predict_scores = pd.concat([y_predict_scores, y_predict_scores_df3], join="outer", axis=1)
y_predict_scores.columns = [0,1,2,3]

y_predict_scores['predict_class'] = y_predict_scores.idxmax(axis=1) #求一行的最大值对应的索引

df = pd.read_csv('reg_data_final.csv',delimiter=',')
A= ~df0.index.isin(train_df0.index)
B= ~df.index.isin(train_df0.index)

test_df = df[~df0.index.isin(train_df0.index)]
test_df_y = test_df.iloc[:,10]
test_df_y.index = y_predict_scores.index

y_predict_scores['true_class'] = test_df_y

accu_count = 0
for i in range(72):
    if y_predict_scores.iloc[i]['predict_class'] == y_predict_scores.iloc[i]['true_class']:
        accu_count = accu_count+1
        
print(accu_count)
        
# 10 Tests:
# 28 29 29 23 31 37 25 23 27 30
# mean accuracy: 28.2/72 = 39.2%

        

