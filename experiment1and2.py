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


ntrain = 1019
nvalidation = 1456
n = 2915
train_features = alldata[:ntrain]
validation_features = alldata[ntrain:nvalidation]
test_features = alldata[nvalidation:n]


# In[91]:


ytrain = y_train[:ntrain]
yvalidation = y_train[ntrain:nvalidation]
ytest = []


# In[ ]:





# In[92]:


print(ytrain.shape)


# In[93]:


ytrain


# In[94]:


print(train_features.shape)


# In[95]:


print(validation_features.shape)


# In[96]:


print(test_features.shape)


# # Importing models libraries

# In[97]:


from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, LassoLarsIC
from sklearn.linear_model import LinearRegression, RidgeCV, Lasso, ElasticNetCV, LassoCV, Ridge
from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LinearRegression
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error
from sklearn import ensemble
from sklearn.metrics import r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from sklearn.neighbors import KNeighborsRegressor
import xgboost as xgb
import lightgbm as lgb


# In[98]:


#define the evaluation function standard validation
def evaluate(model, validation_features, yvalidation, train_features, train_labels):
    predictions = model.predict(validation_features)
    errors = abs(predictions - yvalidation)
    mape = 100 * np.mean(errors / yvalidation)
    accuracy = 100 - mape
    print('Model Performance')
    print('Average Error: {:0.4f} degrees.'.format(np.mean(errors)))
    print('Accuracy = {:0.2f}%.'.format(accuracy))
    print('the goodness of fit r square for the train dateset is:',model.score(train_features, train_labels))
    print('the goodness of fit r square for the test dateset is:',r2_score(yvalidation, predictions))
    print('the RMSE is: ', np.sqrt(mean_squared_error(yvalidation, predictions)))


# In[99]:


#Validation function using cross validation
n_folds = 5

def rmsle_cv(model):
    kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(train_features)
    rmse= np.sqrt(-cross_val_score(model, train_features, ytrain, scoring="neg_mean_squared_error", cv = kf))
    return(rmse)


# # Linear Regression

# In[100]:


#create linear regression model
lm=LinearRegression()
model=lm.fit(train_features, ytrain)
evaluate(model,validation_features, yvalidation, train_features, ytrain)


# # Lasso regression standard validation

# In[101]:


lasso = Lasso(normalize = True)
lasso.fit(train_features,ytrain)
evaluate(lasso,validation_features, yvalidation, train_features, ytrain)


# # Lasso regression cross validation

# In[102]:


lassocv = LassoCV(alphas = None, max_iter = 50000, cv = 10, normalize = True)
lassocv.fit(train_features,ytrain)
evaluate(lassocv,validation_features, yvalidation, train_features, ytrain)


# In[103]:


ytest =lassocv.predict(test_features)
plt.figure(figsize=(8,8))
plt.scatter(lassocv.predict(validation_features) , yvalidation,marker = "^", label = "Validation data", s=9) 
plt.scatter(lassocv.predict(train_features), ytrain,marker = ".", label = 'Training data', s=9)
plt.title("Linear regression with Lasso regularization(Real Vs. Predicted)")
plt.xlabel("Predicted values")
plt.ylabel("Real Values")
plt.legend(loc = "upper left")
plt.plot([10.5, 13.5], [10.5, 13.5], c = "purple")
plt.show()


# In[104]:


#plot feature importance (the magnitude of features in the regression funciton)
coefs = pd.Series(lassocv.coef_, index = train_features.columns)
top_coefs = pd.concat([coefs.sort_values().head(10),
                       coefs.sort_values().tail(10)])
print("Lasso picked " + str(sum(coefs != 0)) + " features and eliminated the other " +        str(sum(coefs == 0)) + " features")
top_coefs.plot(kind = 'barh')
plt.title('Important Features')
plt.xlim(-1,1)
plt.xlabel('Coefficents')


# # Tunning parameters lasso regression

# In[105]:


#coefficient change with alpha
lasso = Lasso(normalize=True)
alphas_lasso = np.logspace(-5, -1, 100)
coef_lasso = [] #going to get one coef_ value for each alpha
for i in alphas_lasso:
    lasso.set_params(alpha=i).fit(train_features,ytrain)
    coef_lasso.append(lasso.coef_)


# In[106]:


title = 'Lasso coefficients as a function of the regularization'
columns = train_features.columns
df_coef = pd.DataFrame(coef_lasso, index=alphas_lasso, columns=columns)
df_coef.plot(logx=True, title=title, legend=False)
plt.xlabel('alpha')
plt.ylabel('coefficients')


# # Ridge Regression standard validation

# In[107]:


ridge = Ridge(normalize = True)
ridge.fit(train_features,ytrain)
evaluate(ridge,validation_features, yvalidation, train_features, ytrain)


# # Ridge Regression cross validation

# In[108]:


alphas_ridge = np.logspace(-5, 2, 100)
ridge = RidgeCV(alphas = alphas_ridge, cv = 10, normalize = True)
ridge.fit(train_features,ytrain)
print('The ridge lambda is:',ridge.alpha_)
evaluate(ridge,validation_features, yvalidation, train_features, ytrain)


# In[109]:


plt.figure(figsize=(8,8))
plt.scatter(ridge.predict(validation_features) , yvalidation,marker = "^", label = "Validation data", s=9) 
plt.scatter(ridge.predict(train_features), ytrain,marker = ".", label = 'Training data', s=9)
plt.title("Linear regression with Ridge regularization(Real Vs. Predicted)")
plt.xlabel("Predicted values")
plt.ylabel("Real Values")
plt.legend(loc = "upper left")
plt.plot([10.5, 13.5], [10.5, 13.5], c = "purple")
plt.show()


# In[110]:


coefs = pd.Series(ridge.coef_, index = validation_features.columns)
top_coefs = pd.concat([coefs.sort_values().head(10),
                       coefs.sort_values().tail(10)])
print("ridge picked " + str(sum(coefs != 0)) + " features and eliminated the other " +        str(sum(coefs == 0)) + " features")
top_coefs.plot(kind = 'barh')
plt.title('Important Features')
plt.xlim(-1,1)
plt.xlabel('Coefficents')


# # Tunning parameters

# In[111]:


#coefficient change with alpha
ridge = Ridge(normalize=True)
alphas_ridge = np.logspace(-3, 3, 100)
coef_ridge = [] #going to get one coef_ value for each alpha
for i in alphas_ridge:
    ridge.set_params(alpha=i).fit(train_features,ytrain)
    coef_ridge.append(ridge.coef_)


# In[112]:


title = 'Ridge coefficients as a function of the regularization'
columns = train_features.columns
df_coef = pd.DataFrame(coef_ridge, index=alphas_ridge, columns=columns)
df_coef.plot(logx=True, title=title, legend=False)
plt.xlabel('alpha')
plt.ylabel('coefficients')


# # Elastic net reggression standard validation

# In[113]:


elastic = ElasticNet(normalize = True)
elastic.fit(train_features,ytrain)
evaluate(elastic,validation_features, yvalidation, train_features, ytrain)


# # Elastic net reggression cross validation

# In[114]:


elastic = ElasticNetCV(alphas = None, max_iter = 50000, cv = 10, normalize = True)
elastic.fit(train_features,ytrain)
print('The elastic lambda is:',elastic.alpha_)
evaluate(elastic,validation_features, yvalidation, train_features, ytrain)


# In[115]:


plt.figure(figsize=(8,8))
plt.scatter(elastic.predict(validation_features) , yvalidation,marker = "^", label = "Validation data", s=9) 
plt.scatter(elastic.predict(train_features), ytrain,marker = ".", label = 'Training data', s=9)
plt.title("Linear regression with ElasticNet regularization(Real Vs. Predicted)")
plt.xlabel("Predicted values")
plt.ylabel("Real Values")
plt.legend(loc = "upper left")
plt.plot([10.5, 13.5], [10.5, 13.5], c = "purple")
plt.show()


# In[116]:


coefs = pd.Series(elastic.coef_, index = train_features.columns)
top_coefs = pd.concat([coefs.sort_values().head(10),
                       coefs.sort_values().tail(10)])
print("ElasticNet picked " + str(sum(coefs != 0)) + " features and eliminated the other " +        str(sum(coefs == 0)) + " features")
top_coefs.plot(kind = 'barh')
plt.title('Important Features')
plt.xlim(-1,1)
plt.xlabel('Coefficents')


# # Tunning parameters

# In[117]:


#coefficient change with alpha
elastic = ElasticNet(normalize=True)
alphas_elastic = np.logspace(-5, -1, 100)
coef_elastic = [] #going to get one coef_ value for each alpha
for i in alphas_elastic:
    elastic.set_params(alpha=i).fit(train_features,ytrain)
    coef_elastic.append(elastic.coef_)


# In[118]:


title = 'ElasticNet coefficients as a function of alpha'
columns = train_features.columns
df_coef = pd.DataFrame(coef_elastic, index=alphas_elastic, columns=columns)
df_coef.plot(logx=True, title=title, legend=False)
plt.xlabel('alpha')
plt.ylabel('coefficients')


# # Kernel Ridge

# In[119]:


KRR = KernelRidge(alpha=0.6, kernel='polynomial', degree=2, coef0=2.5)


# In[120]:


score = rmsle_cv(KRR)
print("Kernel Ridge score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))


# # Gradient Boosting

# In[121]:


GBoost = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,
                                   max_depth=4, max_features='sqrt',
                                   min_samples_leaf=15, min_samples_split=10, 
                                   loss='huber', random_state =5)


# In[122]:


score = rmsle_cv(GBoost)
print("Gradient Boosting score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))


# # XGBRegressor

# In[123]:


model_xgb = xgb.XGBRegressor(colsample_bytree=0.4603, gamma=0.0468, 
                             learning_rate=0.05, max_depth=3, 
                             min_child_weight=1.7817, n_estimators=2200,
                             reg_alpha=0.4640, reg_lambda=0.8571,
                             subsample=0.5213, silent=1,
                             random_state =7, nthread = -1)


# In[124]:


score = rmsle_cv(model_xgb)
print("Xgboost score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))


# # LGBMRegressor

# In[125]:


model_lgb = lgb.LGBMRegressor(objective='regression',num_leaves=5,
                              learning_rate=0.05, n_estimators=720,
                              max_bin = 55, bagging_fraction = 0.8,
                              bagging_freq = 5, feature_fraction = 0.2319,
                              feature_fraction_seed=9, bagging_seed=9,
                              min_data_in_leaf =6, min_sum_hessian_in_leaf = 11)


# In[126]:


score = rmsle_cv(model_lgb)
print("LGBM score: {:.4f} ({:.4f})\n" .format(score.mean(), score.std()))


# # Random Forest

# In[127]:


base_model = ensemble.RandomForestRegressor(n_estimators = 100, random_state = 42, max_features=
                                           150, max_depth=25)
base_model.fit(train_features, ytrain)
base_accuracy = evaluate(base_model, validation_features, yvalidation, train_features, ytrain)


# In[128]:


plt.figure(figsize=(8,8))
plt.scatter(base_model.predict(validation_features) , yvalidation,marker = "^", label = "Validation data", s=9) 
plt.scatter(base_model.predict(train_features), ytrain,marker = ".", label = 'Training data', s=9)
plt.title("Random Forest (Real Vs. Predicted)")
plt.xlabel("Predicted values")
plt.ylabel("Real Values")
plt.legend(loc = "upper left")
plt.plot([10.5, 13.5], [10.5, 13.5], c = "purple")
plt.show()


# In[129]:


feature_importance_base = list(zip(train_features.columns, base_model.feature_importances_))
dtype = [('feature', 'S10'), ('importance', 'float')]
feature_importance_base = np.array(feature_importance_base, dtype=dtype)
feature_sort_base = np.sort(feature_importance_base, order='importance')[::-1]
name, score = zip(*list(feature_sort_base))
pd.DataFrame({'name':name,'score':score})[:15].sort_values('score').plot.barh(x='name', y='score', legend=None)
plt.xlabel('Feature Importance')
plt.ylabel('Features')


# In[130]:


randomForest=ensemble.RandomForestRegressor()
n_trees_range = range(30, 200, 10)  # OOB score will warning if too few trees
train_error2 = []
test_error2 = []
oob_error = []

for n_trees in n_trees_range:
    randomForest.set_params(n_estimators=n_trees, random_state=42, oob_score=True)
    randomForest.fit(train_features, ytrain)
    train_error2.append(1 - randomForest.score(train_features, ytrain))
    test_error2.append(1 - randomForest.score(validation_features, yvalidation))
    oob_error.append(1 - randomForest.oob_score_)


# In[131]:


plt.plot(n_trees_range, train_error2, c='red', label='training error')
plt.plot(n_trees_range, test_error2, c='blue', label='test error')
plt.plot(n_trees_range, oob_error, c='pink', label='oob error')
plt.ylabel('Error')
plt.xlabel('Number of trees')
plt.title('Errors as a function of number of trees')
plt.legend()
plt.show()


# # XGBRegressor

# In[132]:


model = XGBRegressor(learning_rate=0.08, max_depth=4, n_estimators=298)
model.fit(train_features,ytrain)
XGboost_accuracy = evaluate(model, validation_features, yvalidation, train_features, ytrain)


# In[133]:


imp = pd.DataFrame({'Variable':train_features.columns,
              'Importance':model.feature_importances_}).sort_values('Importance', ascending=False)

sns.barplot(x='Importance', y='Variable', data=imp.head(15))
plt.title('Feature Importance XGBRegressor')
plt.ylabel('Feature')


# In[134]:


plt.figure(figsize=(8,8))
plt.scatter(model.predict(validation_features) , yvalidation,marker = "^", label = "Validation data", s=9) 
plt.scatter(model.predict(train_features), ytrain,marker = ".", label = 'Training data', s=9)
plt.title("XGboost (Real Vs. Predicted)")
plt.xlabel("Predicted values")
plt.ylabel("Real Values")
plt.legend(loc = "upper left")
plt.plot([10.5, 13.5], [10.5, 13.5], c = "purple")
plt.show()


# In[135]:


scaler = StandardScaler().fit(train_features)
rescaledX = scaler.transform(train_features)
Kmodel = KNeighborsRegressor(algorithm='auto', leaf_size=15, n_neighbors=4, p=1, weights='distance')
Kmodel.fit(rescaledX, ytrain)
XGboost_accuracy = evaluate(Kmodel, validation_features, yvalidation, train_features, ytrain)


# # SVR

# In[136]:


from sklearn import svm
svr_opt = svm.SVR(C = 10000, gamma = 1e-08)
svr_fit = svr_opt.fit(train_features, ytrain)
svm_accuracy = evaluate(svr_opt, validation_features, yvalidation, train_features, ytrain)


# # Kaggle Submission File

# In[137]:


# Let's make some predictions and submit it to the lb
test_preds = np.expm1(model.predict(test_features))
submission = pd.DataFrame()
number = 2919
submission['Id'] = range(1461, number+1)
submission["SalePrice"] = test_preds
submission.to_csv("rob.csv", index=False)

