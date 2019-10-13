#!/usr/bin/env python
# coding: utf-8

# # Importing data and libraries

# In[1]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O 
import matplotlib.pyplot as plt # for visualization
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns # also for visualization
from scipy import stats # general statistical functions
from scipy.stats import norm, skew #for some statistics
from scipy.special import boxcox1p
from scipy.stats import boxcox_normmax
pd.options.display.max_columns = None # show all columns
import missingno as msno # missing data visualizations and utilities
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import warnings # ignore file warnings
warnings.filterwarnings('ignore')


# In[2]:


pd.set_option('display.float_format', lambda x: '{:.3f}'.format(x)) #Limiting floats output to 3 decimal points


# In[3]:


train = pd.read_csv('train.csv')


# In[4]:


test = pd.read_csv('test.csv')


# # Data description

# In[5]:


train.head()


# In[6]:


train.tail()


# In[7]:


print(train.shape)


# In[8]:


print(test.shape)


# In[9]:


train.describe()


# # Exploring Numeric and categorical features in the dataset

# In[10]:


train.select_dtypes(include=[np.number]).columns, train.select_dtypes(include=[np.object]).columns


# In[11]:


# Drop the 'Id' colum since it's unnecessary for prediction process
train.drop("Id", axis = 1, inplace = True)
test.drop("Id", axis = 1, inplace = True)


# In[12]:


#check again the data size after dropping the 'Id' variable
print("\nThe train data size after dropping Id feature is : {} ".format(train.shape)) 
print("The test data size after dropping Id feature is : {} ".format(test.shape))


# In[13]:


train.head()


# # Checking for missing and null data

# In[14]:


NAs = pd.concat([train.isnull().sum(), train.isnull().sum()], axis=1, keys=['Train', 'Test'])
NAs[NAs.sum(axis=1) > 0]


# In[15]:


# Visualising missing values of numeric features for sample of 500
msno.matrix(train.select_dtypes(include=[np.number]).sample(500))


# In[16]:


# Visualising missing values of categorical features for sample of 200
msno.matrix(train.select_dtypes(include=[np.object]).sample(500))


# In[17]:


# Visualization of nullity by column
msno.bar(train.sample(1000))


# In[18]:


train.SalePrice.describe()


# # Distribution plot for SalePrice

# In[19]:


from scipy.stats import norm
sns.distplot(train['SalePrice'] , fit=norm)

# Get the fitted parameters used by the function
(mu, sigma) = norm.fit(train['SalePrice'])

#Now plot the distribution
plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],
            loc='best')
plt.ylabel('Frequency')
plt.title('SalePrice distribution')


# In[20]:


#normal probability plot

fig = plt.figure()
res = stats.probplot(train['SalePrice'], plot=plt)


# In[21]:


quantitative = [f for f in train.columns if train.dtypes[f] != 'object']
quantitative.remove('SalePrice')
qualitative = [f for f in train.columns if train.dtypes[f] == 'object']


# In[22]:


test_normality = lambda x: stats.shapiro(x.fillna(0))[1] < 0.01
normal = pd.DataFrame(train[quantitative])
normal = normal.apply(test_normality)
print(not normal.any())


# In[23]:


# Determine which kind of variables are present
train.dtypes.unique()


# In[24]:


print("Original size: {}".format(train.shape))

# Drop extreme observations
conditions = [train['LotFrontage'] > 300,
             train['LotArea'] > 200000,
             train['BsmtFinSF1'] > 5000,
             train['TotalBsmtSF'] > 6000,
             train['1stFlrSF'] > 4000,
             np.logical_and(train['GrLivArea'] > 4000, train['SalePrice'] < 300000)]

print("Outliers: {}".format(sum(np.logical_or.reduce(conditions))))


# # Find and drop outliers

# In[25]:


train = train[np.logical_or.reduce(conditions)==False]


# # Features correlations

# In[26]:


print("Find most important features relative to target")
corr_overallqual=train.corr()["SalePrice"]
print (corr_overallqual[np.argsort(corr_overallqual, axis=0)[::-1]])

fig, ax = plt.subplots(figsize = (6, 10))
corr_overallqual[np.argsort(corr_overallqual, axis=0)[::-1]].plot(kind='barh')
plt.tick_params(labelsize=12)
plt.ylabel("Pearson correlation",size=12)
plt.title('Correlated features with SalePrice', size=13)
plt.tight_layout()


# # Univariate plotting

# In[27]:


def explore_variables(target_name, dt):
    for col in dt.drop(target_name, 1).columns:
        if train.dtypes[train.columns.get_loc(col)] == 'O': # categorical variable
            f, ax = plt.subplots()
            fig = sns.boxplot(x=col, y=target_name, data=dt)
            ax = sns.swarmplot(x=col, y=target_name, data=dt, color=".25", alpha=0.2)
            fig.axis(ymin=0, ymax=800000)
        else: # numerical variable
            fig, ax = plt.subplots()
            ax.scatter(x=dt[col], y=dt[target_name])
            plt.ylabel(target_name, fontsize=13)
            plt.xlabel(col, fontsize=13)
            plt.show()


# In[28]:


#explore_variables('SalePrice', train)


# # Log-transformation of the target variable

# In[29]:


#We use the numpy fuction log1p which  applies log(1+x) to all elements of the column
train["SalePrice"] = np.log1p(train["SalePrice"])

#Check the new distribution 
sns.distplot(train['SalePrice'] , fit=norm);

# Get the fitted parameters used by the function
(mu, sigma) = norm.fit(train['SalePrice'])
print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))

#Now plot the distribution
plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],
            loc='best')
plt.ylabel('Frequency')
plt.title('SalePrice distribution')

#Get also the QQ-plot
fig = plt.figure()
res = stats.probplot(train['SalePrice'], plot=plt)
plt.show()


# In[30]:


train["SalePrice"]


# # Concatenating datasets

# In[31]:


ntrain = train.shape[0]
ntest = test.shape[0]
y_train = train.SalePrice.values
alldata = pd.concat((train, test)).reset_index(drop=True)
alldata.drop(['SalePrice'], axis=1, inplace=True)
print("alldata size is : {}".format(alldata.shape))


# In[32]:


alldata_na = (alldata.isnull().sum() / len(alldata)) * 100
alldata_na = alldata_na.drop(alldata_na[alldata_na == 0].index).sort_values(ascending=False)[:30]
missing_data = pd.DataFrame({'Missing Ratio' :alldata_na})
missing_data.head(20)


# # Visualizing Missing data

# In[33]:


f, ax = plt.subplots(figsize=(15, 12))
plt.xticks(rotation='90')
sns.barplot(x=alldata_na.index, y=alldata_na)
plt.xlabel('Features', fontsize=15)
plt.ylabel('Percent of missing values', fontsize=15)
plt.title('Percent missing data by feature', fontsize=15)


# # Exploring features

# In[34]:


alldata['PoolQC'].value_counts()


# In[35]:


alldata["PoolQC"] = alldata["PoolQC"].fillna("None")


# In[36]:


alldata['MiscFeature'].value_counts()


# In[37]:


alldata["MiscFeature"] = alldata["MiscFeature"].fillna("None")


# In[38]:


alldata['Alley'].value_counts()


# In[39]:


alldata["Alley"] = alldata["Alley"].fillna("None")


# In[40]:


alldata['Fence'].value_counts()


# In[41]:


alldata["Fence"] = alldata["Fence"].fillna("None")


# In[42]:


alldata['FireplaceQu'].value_counts()


# In[43]:


alldata["FireplaceQu"] = alldata["FireplaceQu"].fillna("None")


# In[44]:


alldata['LotFrontage'].value_counts()


# # Filling missing values

# In[45]:


#Group by neighborhood and fill in missing value by the median LotFrontage of all the neighborhood
alldata["LotFrontage"] = alldata.groupby("Neighborhood")["LotFrontage"].transform(
    lambda x: x.fillna(x.median()))


# In[46]:


alldata['GarageType'].value_counts()
alldata['GarageCond'].value_counts()
alldata['GarageQual'].value_counts()
alldata['GarageFinish'].value_counts()


# In[47]:


for col in ('GarageType', 'GarageFinish', 'GarageQual', 'GarageCond'):
    alldata[col] = alldata[col].fillna('None')


# In[48]:


alldata['GarageYrBlt'].value_counts()
alldata['GarageArea'].value_counts()
alldata['GarageCars'].value_counts()


# In[49]:


for col in ('GarageYrBlt', 'GarageArea', 'GarageCars'):
    alldata[col] = alldata[col].fillna(0)


# In[50]:


alldata['BsmtFinSF1'].value_counts()
alldata['BsmtFinSF2'].value_counts()
alldata['BsmtUnfSF'].value_counts()
alldata['TotalBsmtSF'].value_counts()
alldata['BsmtFullBath'].value_counts()
alldata['BsmtHalfBath'].value_counts()


# In[51]:


for col in ('BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath'):
    alldata[col] = alldata[col].fillna(0)


# In[52]:


alldata['BsmtExposure'].value_counts()


# In[53]:


alldata['BsmtFinType2'].value_counts()


# In[54]:


alldata['BsmtFinType1'].value_counts()


# In[55]:


alldata['BsmtCond'].value_counts()


# In[56]:


alldata['BsmtQual'].value_counts()


# In[57]:


for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):
    alldata[col] = alldata[col].fillna('None')


# In[58]:


alldata['MasVnrArea'].value_counts()
alldata['MasVnrType'].value_counts()


# In[59]:


alldata["MasVnrType"] = alldata["MasVnrType"].fillna("None")
alldata["MasVnrArea"] = alldata["MasVnrArea"].fillna(0)


# In[60]:


alldata['MSZoning'].value_counts()


# In[61]:


alldata['MSZoning'] = alldata['MSZoning'].fillna(alldata['MSZoning'].mode()[0])


# In[62]:


alldata['Utilities'].value_counts()


# # Dropping variables

# In[63]:


alldata = alldata.drop(['Utilities'], axis=1)


# In[64]:


alldata['Functional'].value_counts()


# In[65]:


alldata["Functional"] = alldata["Functional"].fillna("Typ")


# In[66]:


alldata['Electrical'].value_counts()


# In[67]:


alldata['Electrical'] = alldata['Electrical'].fillna(alldata['Electrical'].mode()[0])


# In[68]:


alldata['KitchenQual'].value_counts()


# In[69]:


alldata['KitchenQual'] = alldata['KitchenQual'].fillna(alldata['KitchenQual'].mode()[0])


# In[70]:


alldata['Exterior1st'].value_counts()
alldata['Exterior2nd'].value_counts()


# In[71]:


alldata['Exterior1st'] = alldata['Exterior1st'].fillna(alldata['Exterior1st'].mode()[0])
alldata['Exterior2nd'] = alldata['Exterior2nd'].fillna(alldata['Exterior2nd'].mode()[0])


# In[72]:


alldata['SaleType'].value_counts()


# In[73]:


alldata['SaleType'] = alldata['SaleType'].fillna(alldata['SaleType'].mode()[0])


# In[74]:


alldata['MSSubClass'].value_counts()


# In[75]:


alldata['MSSubClass'] = alldata['MSSubClass'].fillna("None")


# In[76]:


#Check remaining missing values if any 
alldata_na = (alldata.isnull().sum() / len(alldata)) * 100
alldata_na = alldata_na.drop(alldata_na[alldata_na == 0].index).sort_values(ascending=False)
missing_data = pd.DataFrame({'Missing Ratio' :alldata_na})
missing_data.head()


# # Changing numerical features to categorical features

# In[77]:


#MSSubClass=The building class
alldata['MSSubClass'] = alldata['MSSubClass'].apply(str)


#Changing OverallCond into a categorical variable
alldata['OverallCond'] = alldata['OverallCond'].astype(str)


#Year and month sold are transformed into categorical features.
alldata['YrSold'] = alldata['YrSold'].astype(str)
alldata['MoSold'] = alldata['MoSold'].astype(str)


# In[78]:


train['Street'].value_counts()
train['Condition2'].value_counts()
train['RoofMatl'].value_counts()
train['Heating'].value_counts()
train['Utilities'].value_counts()


# # apply LabelEncoder to categorical features

# In[79]:


from sklearn.preprocessing import LabelEncoder
cols = ('FireplaceQu', 'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond', 
        'ExterQual', 'ExterCond','HeatingQC', 'PoolQC', 'KitchenQual', 'BsmtFinType1', 
        'BsmtFinType2', 'Functional', 'Fence', 'BsmtExposure', 'GarageFinish', 'LandSlope',
        'LotShape', 'PavedDrive', 'Street', 'Alley', 'CentralAir', 'MSSubClass', 'OverallCond', 
        'YrSold', 'MoSold')
# process columns, apply LabelEncoder to categorical features
for c in cols:
    lbl = LabelEncoder() 
    lbl.fit(list(alldata[c].values)) 
    alldata[c] = lbl.transform(list(alldata[c].values))

# shape        
print('Shape alldata: {}'.format(alldata.shape))


# # Creating new features

# In[80]:


# Adding total sqfootage feature 
alldata['TotalSF'] = alldata['TotalBsmtSF'] + alldata['1stFlrSF'] + alldata['2ndFlrSF']


# # Checking assimetry of numerical features

# In[81]:


# First we need to find all numeric features in the data
numeric_dtypes = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
numerics2 = []
for i in alldata.columns:
    if alldata[i].dtypes in numeric_dtypes:
        numerics2.append(i)


# In[82]:


# Box plots for all our numeric features
sns.set_style("white")
f, ax = plt.subplots(figsize=(8, 7))
ax.set_xscale("log")
ax = sns.boxplot(data=alldata[numerics2] , orient="h", palette="Set1")
# Tweak the visual presentation
ax.xaxis.grid(False)
ax.set(ylabel="Feature names")
ax.set(xlabel="Numeric values")
ax.set(title="Numeric Distribution of Features")
sns.despine(trim=True, left=True)


# # Transforming to a normal shape

# In[83]:


# Find the skewed  numerical features
skew_features = alldata[numerics2].apply(lambda x: skew(x)).sort_values(ascending=False)

high_skew = skew_features[skew_features > 0.5]
skew_index = high_skew.index
print("There are {} numerical features with Skew > 0.5 :".format(high_skew.shape[0]))
skewness = pd.DataFrame({'Skew' :high_skew})
skew_features.head(10)


# In[84]:


# Normalise skewed features
for i in skew_index:
    alldata[i] = boxcox1p(alldata[i], boxcox_normmax(alldata[i] + 1))
    


# In[85]:


sns.set_style("white")
f, ax = plt.subplots(figsize=(8, 7))
ax.set_xscale("log")
ax = sns.boxplot(data=alldata[skew_index] , orient="h", palette="Set1")
# Tweak the visual presentation
ax.xaxis.grid(False)
ax.set(ylabel="Feature names")
ax.set(xlabel="Numeric values")
ax.set(title="Numeric Distribution of Features")
sns.despine(trim=True, left=True)   


# In[ ]:





# In[86]:


sns.set_style("white")
f, ax = plt.subplots(figsize=(8, 7))
ax.set_xscale("log")
ax = sns.boxplot(data=alldata[skew_index] , orient="h", palette="Set1")
# Tweak the visual presentation
ax.xaxis.grid(False)
ax.set(ylabel="Feature names")
ax.set(xlabel="Numeric values")
ax.set(title="Numeric Distribution of Features")
sns.despine(trim=True, left=True)


# In[87]:


alldata = pd.get_dummies(alldata)
print(alldata.shape)


# In[88]:


sns.set_style("white")
f, ax = plt.subplots(figsize=(8, 7))
ax.set_xscale("log")
ax = sns.boxplot(data=alldata[skew_index] , orient="h", palette="Set1")
# Tweak the visual presentation
ax.xaxis.grid(False)
ax.set(ylabel="Feature names")
ax.set(xlabel="Numeric values")
ax.set(title="Numeric Distribution of Features")
sns.despine(trim=True, left=True)


# In[89]:


print(y_train.shape)


# # Divide the data set as required(training set, validation set,test set)

# In[90]:


# Spliting the data back to train(X,y) and test(X_sub)
X = alldata.iloc[:len(y_train), :]
X_test = alldata.iloc[len(y_train):, :]
print('Features size for train(X,y) and test(X_test):')
print('X', X.shape, 'y', y_train.shape, 'X_test', X_test.shape)


# In[91]:


print(X.shape)


# In[92]:


from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import ElasticNetCV, LassoCV, RidgeCV
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn import ensemble
from xgboost import XGBRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.ensemble import GradientBoostingRegressor
import lightgbm as lgb


# In[93]:


kfolds = KFold(n_splits=10, shuffle=True, random_state=42)

# model scoring and validation function
def cv_rmse(model, X=X):
    rmse = np.sqrt(-cross_val_score(model, X, y_train,scoring="neg_mean_squared_error",cv=kfolds))
    return (rmse)

# rmsle scoring function
def rmsle(y_train, y_pred):
    return np.sqrt(mean_squared_error(y_train, y_pred))


# In[94]:


e_alphas = [0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007]
e_l1ratio = [0.8, 0.85, 0.9, 0.95, 0.99, 1]
alphas_alt = [14.5, 14.6, 14.7, 14.8, 14.9, 15, 15.1, 15.2, 15.3, 15.4, 15.5]
alphas2 = [5e-05, 0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007, 0.0008]


# Kernel Ridge Regression : made robust to outliers
ridge = make_pipeline(RobustScaler(), RidgeCV(alphas=alphas_alt, cv=kfolds))

# LASSO Regression : made robust to outliers
lasso = make_pipeline(RobustScaler(), LassoCV(max_iter=1e7, 
                    alphas=alphas2,random_state=42, cv=kfolds))

# Elastic Net Regression : made robust to outliers
elasticnet = make_pipeline(RobustScaler(), ElasticNetCV(max_iter=1e7, 
                         alphas=e_alphas, cv=kfolds, l1_ratio=e_l1ratio))

# store models, scores and prediction values 
models = {'Ridge': ridge,
          'Lasso': lasso, 
          'ElasticNet': elasticnet}
predictions = {}
scores = {}

for name, model in models.items():
    
    model.fit(X, y_train)
    predictions[name] = np.expm1(model.predict(X))
    
    score = cv_rmse(model, X=X)
    scores[name] = (score.mean(), score.std())


# In[95]:


# get the performance of each model on training data(validation set)
print('---- Score with CV_RMSLE-----')
score = cv_rmse(ridge)
print("Ridge score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))

score = cv_rmse(lasso)
print("Lasso score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))

score = cv_rmse(elasticnet)
print("ElasticNet score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))


# In[96]:


#Fit the training data X, y
print('----START Fit----')
print('Elasticnet')
elastic_model = elasticnet.fit(X, y_train)
print('Lasso')
lasso_model = lasso.fit(X, y_train)
print('Ridge')
ridge_model = ridge.fit(X, y_train)


# In[97]:


# model blending function using fitted models to make predictions
def blend_models(X):
    return ((elastic_model.predict(X)) + (lasso_model.predict(X)) + (ridge_model.predict(X)))/3
blended_score = rmsle(y_train, blend_models(X))
print('RMSLE score on train data:')
print(rmsle(y_train, blend_models(X)))


# In[98]:


# visualise model performance
sns.set_style("white")
fig, axs = plt.subplots(ncols=0, nrows=3, figsize=(8, 7))
plt.subplots_adjust(top=3.5, right=2)

for i, model in enumerate(models, 1):
    plt.subplot(3, 1, i)
    plt.scatter(predictions[model], np.expm1(y_train))
    plt.plot([0, 800000], [0, 800000], '--r')

    plt.xlabel('{} Predictions (y_pred)'.format(model), size=15)
    plt.ylabel('Real Values (y_train)', size=13)
    plt.tick_params(axis='x', labelsize=12)
    plt.tick_params(axis='y', labelsize=12)

    plt.title('{} Predictions vs Real Values'.format(model), size=15)
    plt.text(0, 700000, 'Mean RMSE: {:.6f} / Std: {:.6f}'.format(scores[model][0], scores[model][1]), fontsize=15)
    ax.xaxis.grid(False)
    sns.despine(trim=True, left=True)
plt.show()


# In[99]:


scores['Blender'] = (blended_score, 0)
sns.set_style("dark")
fig = plt.figure(figsize=(24, 12))

ax = sns.pointplot(x=list(scores.keys()), y=[score for score, _ in scores.values()], markers=['o'], linestyles=['-'])
for i, score in enumerate(scores.values()):
    ax.text(i, score[0] + 0.002, '{:.6f}'.format(score[0]), horizontalalignment='left', size='large', color='black', weight='semibold')

plt.ylabel('Score (RMSE)', size=20, labelpad=12.5)
plt.xlabel('Model', size=20, labelpad=12.5)
plt.tick_params(axis='x', labelsize=13.5)
plt.tick_params(axis='y', labelsize=12.5)

plt.title('Scores of Models', size=20)

plt.show()


# In[100]:


# get the target variable/ y_test with X_test
print('Predict submission')
y_test_r = pd.read_csv("test.csv")
submission = pd.read_csv("rob.csv")
y_test = np.log1p(y_test_r.iloc[:,1].values)
submission.iloc[:,1] = np.expm1(blend_models(X_test))
blended_score = rmsle(y_test, blend_models(X_test))


# In[101]:


base_model = ensemble.RandomForestRegressor(n_estimators = 100, random_state = 42, max_features=
                                           150, max_depth=25)
base_model.fit(X, y_train)
score = cv_rmse(base_model)
print("Random Forest Regressor score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))


# In[102]:


modelXGB = XGBRegressor(learning_rate=0.08, max_depth=4, n_estimators=298)
modelXGB.fit(X,y_train)
score = cv_rmse(modelXGB)
print("XGBRegressor score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))


# In[103]:


from sklearn import svm
svr_opt = svm.SVR(C = 100000, gamma = 1e-08)
svr_fit = svr_opt.fit(X, y_train)
score = cv_rmse(svr_opt)
print("SVR score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))


# In[104]:


submission.to_csv("new_submission.csv", index=False)
print('Save submission')


# In[105]:


KRR = KernelRidge(alpha=0.6, kernel='polynomial', degree=2, coef0=2.5)


# In[106]:


score = cv_rmse(KRR)
print("Kernel Ridge score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))


# In[107]:


GBoost = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,
                                   max_depth=4, max_features='sqrt',
                                   min_samples_leaf=15, min_samples_split=10, 
                                   loss='huber', random_state =5)


# In[108]:


score = cv_rmse(GBoost)
print("Gradient Boosting score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))


# In[109]:


model_lgb = lgb.LGBMRegressor(objective='regression',num_leaves=5,
                              learning_rate=0.05, n_estimators=720,
                              max_bin = 55, bagging_fraction = 0.8,
                              bagging_freq = 5, feature_fraction = 0.2319,
                              feature_fraction_seed=9, bagging_seed=9,
                              min_data_in_leaf =6, min_sum_hessian_in_leaf = 11)


# In[110]:


score = cv_rmse(model_lgb)
print("LGBM score: {:.4f} ({:.4f})\n" .format(score.mean(), score.std()))


# In[ ]:





# In[ ]:





# In[ ]:




