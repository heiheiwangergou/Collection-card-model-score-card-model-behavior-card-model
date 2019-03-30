import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from time import strptime,mktime
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn import cross_validation, metrics

def LogitRR(x):
    '''
    :param x: 划款率，有的超过1，有的为0.做截断处理
    :return: 将还款率转化成logit变换
    '''
    if x >= 1:
        y = 0.9999
    elif x == 0:
        y = 0.0001
    else:
        y = x
    return np.log(y/(1-y))

def MakeupMissingCategorical(x):
    if str(x) == 'nan':
        return 'Unknown'
    else:
        return x

def MakeupMissingNumerical(x,replacement):
    if np.isnan(x):
        return replacement
    else:
        return x

'''
第一步：文件准备
'''
foldOfData = 'C:/Users/OkO/Desktop/Financial Data Analsys/3nd Series/Data/'
mydata = pd.read_csv(foldOfData + "prosperLoanData_chargedoff.csv",header = 0)
#催收还款率等于催收金额/（所欠本息+催收费用）。其中催收费用以支出形式表示
mydata['rec_rate'] = mydata.apply(lambda x: x.LP_NonPrincipalRecoverypayments /(x.AmountDelinquent-x.LP_CollectionFees), axis=1)
mydata['rec_rate'] = mydata['rec_rate'].map(lambda x: min(x,1))
#mydata['recovery_status'] = mydata['rec_rate'].map(lambda x: x<=0.5)
#还款率是0~1之间的数，需要通过logit变换，映射到实数空间
#mydata['logit_rr'] = mydata['rec_rate'].map(LogitRR)
#整个开发数据分为训练集、测试集2个部分
trainData, testData = train_test_split(mydata,test_size=0.4)

'''
第二步：数据预处理
'''
categoricalFeatures = ['CreditGrade','Term','BorrowerState','Occupation','EmploymentStatus','IsBorrowerHomeowner','CurrentlyInGroup','IncomeVerifiable']

numFeatures = ['BorrowerAPR','BorrowerRate','LenderYield','ProsperRating (numeric)','ProsperScore','ListingCategory (numeric)','EmploymentStatusDuration','CurrentCreditLines',
                'OpenCreditLines','TotalCreditLinespast7years','CreditScoreRangeLower','OpenRevolvingAccounts','OpenRevolvingMonthlyPayment','InquiriesLast6Months','TotalInquiries',
               'CurrentDelinquencies','DelinquenciesLast7Years','PublicRecordsLast10Years','PublicRecordsLast12Months','BankcardUtilization','TradesNeverDelinquent (percentage)',
               'TradesOpenedLast6Months','DebtToIncomeRatio','LoanFirstDefaultedCycleNumber','LoanMonthsSinceOrigination','PercentFunded','Recommendations','InvestmentFromFriendsCount',
               'Investors']

'''
类别型变量需要用目标变量的均值进行编码
'''
encodedFeatures = []
encodedDict = {}
for var in categoricalFeatures:
    trainData[var] = trainData[var].map(MakeupMissingCategorical)
    avgTarget = trainData.groupby([var])['rec_rate'].mean()
    avgTarget = avgTarget.to_dict()
    newVar = var + '_encoded'
    trainData[newVar] = trainData[var].map(avgTarget)
    encodedFeatures.append(newVar)
    encodedDict[var] = avgTarget

#对数值型数据的缺失进行补缺
trainData['ProsperRating (numeric)'] = trainData['ProsperRating (numeric)'].map(lambda x: MakeupMissingNumerical(x,0))
trainData['ProsperScore'] = trainData['ProsperScore'].map(lambda x: MakeupMissingNumerical(x,0))

avgDebtToIncomeRatio = np.mean(trainData['DebtToIncomeRatio'])
trainData['DebtToIncomeRatio'] = trainData['DebtToIncomeRatio'].map(lambda x: MakeupMissingNumerical(x,avgDebtToIncomeRatio))
numFeatures2 = numFeatures + encodedFeatures

# cls = DecisionTreeRegressor()
# cls.fit(trainData[numFeatures2], trainData['logit_rr'])
# trainData['pred'] = cls.predict(trainData[numFeatures2])
# trainData['less_rr'] = trainData.apply(lambda x: int(x.pred > x.logit_rr), axis=1)
# np.mean(trainData['less_rr'])
# err = trainData.apply(lambda x: np.abs(x.pred - x.logit_rr), axis=1)
# np.mean(err)


'''
第三步：调参
对基于CART的随机森林的调参，主要有：
1，树的个数
2，树的最大深度
3，内部节点最少样本数与叶节点最少样本数
4，特征个数

此外，调参过程中选择的误差函数是均值误差，5倍折叠
'''
X, y= trainData[numFeatures2],trainData['rec_rate']

param_test1 = {'n_estimators':range(10,80,5)}
gsearch1 = GridSearchCV(estimator = RandomForestRegressor(min_samples_split=50,min_samples_leaf=10,max_depth=8,max_features='sqrt' ,random_state=10),
                       param_grid = param_test1, scoring='neg_mean_squared_error',cv=5)
gsearch1.fit(X,y)
gsearch1.best_params_, gsearch1.best_score_
best_n_estimators = gsearch1.best_params_['n_estimators']

param_test2 = {'max_depth':range(3,21), 'min_samples_split':range(10,100,10)}
gsearch2 = GridSearchCV(estimator = RandomForestRegressor(n_estimators=best_n_estimators, min_samples_leaf=10,max_features='sqrt' ,random_state=10,oob_score=True),
                       param_grid = param_test2, scoring='neg_mean_squared_error',cv=5)
gsearch2.fit(X,y)
gsearch2.best_params_, gsearch2.best_score_
best_max_depth = gsearch2.best_params_['max_depth']
best_min_sample_split = gsearch2.best_params_['min_samples_split']

param_test3 = {'min_samples_split':range(50,201,10), 'min_samples_leaf':range(1,20,2)}
gsearch3 = GridSearchCV(estimator = RandomForestRegressor(n_estimators=best_n_estimators, max_depth = best_max_depth,max_features='sqrt',random_state=10,oob_score=True),
                       param_grid = param_test3, scoring='neg_mean_squared_error',cv=5)
gsearch3.fit(X,y)
gsearch3.best_params_, gsearch3.best_score_
best_min_samples_leaf = gsearch3.best_params_['min_samples_leaf']
best_min_samples_split = gsearch3.best_params_['min_samples_split']

numOfFeatures = len(numFeatures2)
mostSelectedFeatures = numOfFeatures/2
param_test4 = {'max_features':range(3,numOfFeatures+1)}
gsearch4 = GridSearchCV(estimator = RandomForestRegressor(n_estimators=best_n_estimators, max_depth=best_max_depth,min_samples_leaf=best_min_samples_leaf,
                                                          min_samples_split=best_min_samples_split,random_state=10,oob_score=True),
                       param_grid = param_test4, scoring='neg_mean_squared_error',cv=5)
gsearch4.fit(X,y)
gsearch4.best_params_, gsearch4.best_score_
best_max_features = gsearch4.best_params_['max_features']

cls = RandomForestRegressor(n_estimators=best_n_estimators,
                            max_depth=best_max_depth,
                            min_samples_leaf=best_min_samples_leaf,
                            min_samples_split=best_min_samples_split,
                            max_features=best_max_features,
                            random_state=10,
                            oob_score=True)
cls.fit(X,y)
trainData['pred'] = cls.predict(trainData[numFeatures2])
trainData['less_rr'] = trainData.apply(lambda x: int(x.pred > x.rec_rate), axis=1)
np.mean(trainData['less_rr'])
err = trainData.apply(lambda x: np.abs(x.pred - x.rec_rate), axis=1)
np.mean(err)



'''
第四步：在测试集上测试效果
'''

for var in categoricalFeatures:
    testData[var] = testData[var].map(MakeupMissingCategorical)
    newVar = var + '_encoded'
    testData[newVar] = testData[var].map(encodedDict[var])
    avgnewVar = np.mean(trainData[newVar])
    testData[newVar] = testData[newVar].map(lambda x: MakeupMissingNumerical(x, avgnewVar))

testData['ProsperRating (numeric)'] = testData['ProsperRating (numeric)'].map(lambda x: MakeupMissingNumerical(x,0))
testData['ProsperScore'] = testData['ProsperScore'].map(lambda x: MakeupMissingNumerical(x,0))
testData['DebtToIncomeRatio'] = testData['DebtToIncomeRatio'].map(lambda x: MakeupMissingNumerical(x,avgDebtToIncomeRatio))

testData['pred'] = cls.predict(testData[numFeatures2])
testData['less_rr'] = testData.apply(lambda x: int(x.pred > x.rec_rate), axis=1)
np.mean(testData['less_rr'])
err = testData.apply(lambda x: np.abs(x.pred - x.rec_rate), axis=1)
np.mean(err)
