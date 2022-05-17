import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# 读取文件
df = pd.read_csv(
    'D:\\客户流失\\WA_Fn-UseC_-Telco-Customer-Churn.csv',
    encoding='utf8', engine='python'
)
df.shape    #查看数据规格
df.head()   #查看前几行数据
df.dtypes   #查看表中各个特征的数据类型
# print(df.isnull().any()) 判断出哪些列一列有缺失值
print(df['TotalCharges'].isnull().sum())   # 统计TotalCharges（总费用）这一列缺失值的个数
filter = df.TotalCharges.str.match('^[0-9]')# 利用正则表达式匹配所有数字
df[~filter]  #输出TotalCharges（总费用）为空格字符所在行的数据
df[~filter]['TotalCharges']
        #输出结果正如我们所看到的，只显示了索引号，但没有找到数值，而只是一个空格字符
df.iloc[488]['TotalCharges'] # 输出为' ' ,

## 接着我们用中位数填充所有空格字符
temp_df = df.drop(index=[488,753,936,1082,1340,3331,3826,4380,5218,6670,6754 ])
temp_df['TotalCharges'] = temp_df['TotalCharges'].astype('float64')

print(temp_df['TotalCharges'].median()) #得到中位数1397.475
#print(df['TotalCharges'].unique())#unique()方法返回的是去重之后的不同值，而nunique()方法则直接放回不同值的个数
df['TotalCharges'][ df['TotalCharges'] == ' ' ] = 1397.475
df['TotalCharges'] = df['TotalCharges'].astype('float64')
print(df.dtypes)

#从数据集中删除 ID 特征
df = df.drop(['customerID'], axis=1)
#将流失功能的值yes/no，标记为 1/0
df['Churn'].value_counts()
df['target'] = np.where(df['Churn']=='Yes', 1, 0)
print(df)
#删除 Churn保留Target
df=df.drop(['Churn'], axis=1)
print(df)

##定义目标和独立特征（x和y）
Y = df[['target']]
X = df.drop(['target'], axis=1)
Y.mean() #获取流失率

##将特征拆分为数值和分类
num = X.select_dtypes(include = 'number')
char = X.select_dtypes(include = 'object')
print(num.describe())
# num.describe()输出结果中发现SeniorCitizen比较奇怪， 检查SeniorCitizen特征是否是一个指标
#Check whether SeniorCitizen feature is an indicator
num.SeniorCitizen.value_counts()
ind=num[['SeniorCitizen']]
num=num.drop(['SeniorCitizen'],axis=1)

#数值特征的异常值分析
num.describe(percentiles=[0.01,0.05,0.10,0.25,0.50,0.75,0.85,0.9,0.99])
print(num.describe(percentiles=[0.01,0.05,0.10,0.25,0.50,0.75,0.85,0.9,0.99]))

#异常值的上限和下限

#我们进行封顶只是为了确保特征的标准差及其方差不会发生很大变化，因为方差决定了该特征的预测能力
def outlier_cap(x):
    x = x.clip(lower = x.quantile(0.01),upper = x.quantile(0.99))
   
    return x
num = num.apply(lambda x: outlier_cap(x))
num.describe(percentiles=[0.01,0.05,0.10,0.25,0.50,0.75,0.85,0.9,0.99])
print(num.describe(percentiles=[0.01,0.05,0.10,0.25,0.50,0.75,0.85,0.9,0.99]))


##缺失值分析
print(num.isnull().sum())#由于数据不包含任何缺失值，因此不需要插补过程

##特征选择 - 数值float特征
        #第 1 部分：删除具有 0 方差的特征 
#获取要保留的列并仅使用这些列创建新的数据框 
from sklearn.feature_selection import VarianceThreshold

varselector = VarianceThreshold(threshold = 0) #默认删除方差为0的特征
varselector.fit_transform(num)
#Get columns to keep and create new dataframe with those only
cols = varselector.get_support(indices=True)
print(cols )
num_1 = num.iloc[:,cols]

       #第 2 部分 - 双变量分析（特征离散化）

#将连续型变量排序后按顺序分箱后编码 
from sklearn.preprocessing import KBinsDiscretizer
discrete = KBinsDiscretizer(n_bins = 10, encode = 'ordinal', strategy = 'quantile')
num_binned = pd.DataFrame(discrete.fit_transform(num_1), index = num_1.index, columns=num_1.columns ).add_suffix('Rank')
#add_suffix--列标签后面添加后缀

X_bin_combined=pd.concat([Y,num_binned],axis=1,join='inner')
print(X_bin_combined)
from numpy import mean
for col in (num_binned.columns):
    plt.figure()
    sns.lineplot(x=col, y=X_bin_combined['target'].mean(), data=X_bin_combined, color='red')
    sns.barplot(x=col, y="target",data=X_bin_combined , estimator=mean)
plt.show()
select_features_df_num=num_1

##特征选择 - 分类特征
char.dtypes
  
        #第 1 部分 - 双变量分析
X_char_merged=pd.concat([Y,char],axis=1,join='inner')
print(X_char_merged)
from numpy import mean
#for col in (char.columns):
    #plt.figure()
    #sns.lineplot(x=col, y=X_char_merged['target'].mean(), data=X_char_merged, color='red')
   # sns.barplot(x=col, y="target",data=X_char_merged, estimator=mean )
#plt.show()

char=char.drop(['gender','PhoneService','MultipleLines'],axis=1)
X_char_dum = pd.get_dummies(char)  #利用pandas实现one hot encode的方式
print(X_char_dum)
X_char_dum.shape

        #第 2 部分 - 选择 K 最佳 

#为分类特征选择 K 最佳值
from sklearn.feature_selection import SelectKBest, chi2
selector = SelectKBest(chi2, k=20)
selector.fit_transform(X_char_dum, Y)
# 获取要保留的列并仅使用这些列创建新的数据框
cols = selector.get_support(indices=True)
select_features_df_char = X_char_dum.iloc[:,cols]


##特征选择 - int指标特征
X_ind_merged=pd.concat([Y,ind],axis=1,join='inner')
from numpy import mean
for col in (ind.columns):
    plt.figure()
    sns.lineplot(x=col, y=X_ind_merged['target'].mean(), data=X_ind_merged, color='red')
    sns.barplot(x=col, y="target",data=X_ind_merged, estimator=mean )
plt.show()
select_features_df_ind=ind

##为模型开发创建主特征集 
X_all=pd.concat([select_features_df_char,select_features_df_num,select_features_df_ind],axis=1,join="inner")

Y = df[['target']]

##Train Test Split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test=train_test_split(X_all, Y, test_size=0.3, random_state=99)
print("Shape of Training Data",X_train.shape)
print("Shape of Testing Data",X_test.shape)
print("Response Rate in Training Data",y_train.mean())
print("Response Rate in Testing Data",y_test.mean())

##Model Building
#逻辑回归
from sklearn.linear_model import LogisticRegression
logreg=LogisticRegression(random_state=0)
logreg.fit(X_train,y_train)
coeff_df=pd.DataFrame(X_all.columns)
coeff_df.columns=['features']
print(coeff_df)
coeff_df["Coefficient Estimate"] = pd.Series(logreg.coef_[0])
print(coeff_df)

#建立决策树模型
from sklearn.tree import DecisionTreeClassifier
dtree=DecisionTreeClassifier(criterion='gini',random_state=99)

from sklearn.model_selection import GridSearchCV
param_dist = {'max_depth': [3, 5, 6, 7], 'min_samples_split': [50, 100, 150, 200, 250] }
tree_grid = GridSearchCV(dtree, cv = 10, param_grid=param_dist,n_jobs = 3)
tree_grid.fit(X_train,y_train) 
print('Best Parameters using grid search: \n', tree_grid.best_params_)
dtree=DecisionTreeClassifier(criterion='gini',random_state=99,max_depth=7,min_samples_split=150)
dtree.fit(X_train,y_train)
from sklearn import tree
import pydotplus
import matplotlib.pyplot as plt
plt.figure(figsize=[50,10])
tree.plot_tree(dtree,filled=True,rounded=True,fontsize=4,feature_names=X_all.columns)
plt.show()

##构建随机森林模型
from sklearn.ensemble import RandomForestClassifier
rf=RandomForestClassifier(criterion='gini',random_state=99,max_depth=7,min_samples_split=150)
rf.fit(X_train,y_train)

import pandas as pd
feature_importances=pd.DataFrame(rf.feature_importances_,
                                 index=X_train.columns,
                                 columns=['importance']).sort_values('importance',ascending=False)
print(feature_importances)

base_learners = [      ('rf', RandomForestClassifier(criterion='gini',random_state=0,max_depth=7,min_samples_split=150)),
                       ('dtree', DecisionTreeClassifier(criterion='gini',random_state=99,max_depth=7,min_samples_split=150))
                       ]
from sklearn.ensemble import StackingClassifier
clf = StackingClassifier(estimators=base_learners, final_estimator=LogisticRegression())
clf.fit(X_train, y_train)

## 模型评估 
y_pred_logreg=logreg.predict(X_test)
y_pred_tree=dtree.predict(X_test)
y_pred_rf=rf.predict(X_test)
y_pred_stacking=clf.predict(X_test)
from sklearn import metrics
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay
print('Logistic Regression')
print("Accuracy:",metrics.accuracy_score(y_test, y_pred_logreg))
print("Precision",metrics.precision_score(y_test,y_pred_logreg))
print("Recall",metrics.recall_score(y_test,y_pred_logreg))
print("f1_score",metrics.f1_score(y_test,y_pred_logreg))
ConfusionMatrixDisplay.from_estimator(logreg,X_test,y_test)
plt.show()

print('Decision Tree')
print("Accuracy:",metrics.accuracy_score(y_test, y_pred_tree))
print("Precision",metrics.precision_score(y_test,y_pred_tree))
print("Recall",metrics.recall_score(y_test,y_pred_tree))
print("f1_score",metrics.f1_score(y_test,y_pred_tree))
ConfusionMatrixDisplay.from_estimator(dtree,X_test,y_test)
plt.show()

print('Random Forest')
print("Accuracy:",metrics.accuracy_score(y_test, y_pred_rf))
print("Precision",metrics.precision_score(y_test,y_pred_rf))
print("Recall",metrics.recall_score(y_test,y_pred_rf))
print("f1_score",metrics.f1_score(y_test,y_pred_rf))
ConfusionMatrixDisplay.from_estimator(rf,X_test,y_test)
plt.show()

print('Stacking')
print("Accuracy:",metrics.accuracy_score(y_test, y_pred_stacking))
print("Precision",metrics.precision_score(y_test,y_pred_stacking))
print("Recall",metrics.recall_score(y_test,y_pred_stacking))
print("f1_score",metrics.f1_score(y_test,y_pred_stacking))
ConfusionMatrixDisplay.from_estimator(clf,X_test,y_test)
plt.show()

# 逻辑回归洛伦兹曲线
y_pred_prob = logreg.predict_proba(X_all)[:, 1]
df['pred_prob_logreg']=pd.DataFrame(y_pred_prob)
df['P_Rank_logreg']=pd.qcut(df['pred_prob_logreg'].rank(method='first').values,10,duplicates='drop').codes+1
for key, value in df.groupby('P_Rank_logreg'):
    print(key)
    print(value)
    print("")
rank_df_actuals=df.groupby('P_Rank_logreg')['target'].agg(['count','mean'])

rank_df_predicted=df.groupby('P_Rank_logreg')['pred_prob_logreg'].agg(['mean'])
rank_df_actuals=pd.DataFrame(rank_df_actuals)

rank_df_actuals.rename(columns={'mean':'Actutal_event_rate'},inplace=True)
rank_df_predicted=pd.DataFrame(rank_df_predicted)
rank_df_predicted.rename(columns={'mean':'Predicted_event_rate'},inplace=True)
rank_df=pd.concat([rank_df_actuals,rank_df_predicted],axis=1,join="inner")
print(rank_df)
sorted_rank_df=rank_df.sort_values(by='P_Rank_logreg',ascending=False)
sorted_rank_df['N_events']=rank_df['count']*rank_df['Actutal_event_rate']
sorted_rank_df['cum_events']=sorted_rank_df['N_events'].cumsum()
sorted_rank_df['event_cap']=sorted_rank_df['N_events']/max(sorted_rank_df['N_events'].cumsum())
sorted_rank_df['cum_event_cap']=sorted_rank_df['event_cap'].cumsum()

sorted_rank_df['N_non_events']=sorted_rank_df['count']-sorted_rank_df['N_events']
sorted_rank_df['cum_non_events']=sorted_rank_df['N_non_events'].cumsum()
sorted_rank_df['non_event_cap']=sorted_rank_df['N_non_events']/max(sorted_rank_df['N_non_events'].cumsum())
sorted_rank_df['cum_non_event_cap']=sorted_rank_df['non_event_cap'].cumsum()

sorted_rank_df['KS']=round((sorted_rank_df['cum_event_cap']-sorted_rank_df['cum_non_event_cap']),4)

sorted_rank_df['random_cap']=sorted_rank_df['count']/max(sorted_rank_df['count'].cumsum())
sorted_rank_df['cum_random_cap']=sorted_rank_df['random_cap'].cumsum()
print(sorted_rank_df)
sorted_reindexed=sorted_rank_df.reset_index()
sorted_reindexed['Decile']=sorted_reindexed.index+1
print(sorted_reindexed)
#增益图表 给定十分位级别的增益是达到该十分位的目标（事件）的累积数量与整个数据集中目标（事件）总数的比值。在给定的十分位级别覆盖的目标（事件）百分比。
#例如，80% 的目标覆盖在基于模型的前 20% 的数据中。 在购买倾向模型的情况下，我们可以说我们可以通过向 20% 的总客户发送电子邮件来识别和定位可能购买产品的 80% 的客户。 
#提升图它衡量与没有模型的情况相比，使用预测模型可以做得更好。 它是在给定的十分位水平下增益百分比与随机期望百分比的比率。 第 x 个十分位数的随机期望是 x%。



##实际对比预测 事件率

ax = sns.lineplot( x="Decile", y="Actutal_event_rate", data=sorted_reindexed,color='red')
ax = sns.lineplot( x="Decile", y="Predicted_event_rate", data=sorted_reindexed,color='green')
plt.show()

##增益图
ax = sns.lineplot( x="Decile", y="cum_non_event_cap", data=sorted_reindexed,color='red')
ax = sns.lineplot( x="Decile", y="cum_event_cap", data=sorted_reindexed,color='green')
plt.show()































