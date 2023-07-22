#!/usr/bin/env python
# coding: utf-8

# #Import Library

# In[100]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


# #Read Data

# In[101]:


data = pd.read_csv('loan_data_2007_2014.csv')
data.head()


# # Data Understanding

# In[102]:


data.info()


# In[103]:


(data.isna().mean()*100).sort_values(ascending=False).head(30)


# In[104]:


values_miss = data.isna().mean() * 100
column_values_miss = values_miss[values_miss > 40].index
column_values_miss


# In[105]:


data.drop(column_values_miss, axis = 1, inplace = True)


# In[106]:


data.drop('Unnamed: 0', inplace = True, axis = 1)


# In[107]:


print('id',data['id'].nunique())
print('id_member',data['member_id'].nunique())


# In[108]:


column = ['id','member_id','url','sub_grade','zip_code']
data.drop(column, axis=1, inplace= True)


# # Variable Target

# Tujuan utama pada credit risk adalah untuk melakukan prediksi terhadap suatu individu tentang seberapa mampu mereka untuk melakukan pembayaran terhadap pinjaman atau kredit yang diberikan.
# 
# Variabel yang digunakan sebagai variabel target dalam kasus ini adalah 'loan_status', alasannya adalah mencerminkan performa tiap individu terhadap pembayaran pada pinjaman atau kredit

# In[109]:


data.loan_status.value_counts(normalize = True) * 100


# In[110]:


status_loan_bad = ['Charged Off', 'Late (31-120 days)', 'Default', 'Does not meet the credit policy. Status:Charged Off']
data['status_loan_bad'] = np.where(data['loan_status'].isin(status_loan_bad), 1, 0)

data.drop(columns = ['loan_status'], inplace = True)


# In[111]:


data['status_loan_bad'].value_counts(normalize = True)*100


# # Data Cleaning

# In[112]:


data.iloc[:,0:16].head(10)


# In[113]:


data.iloc[:,16:32].head(10)


# In[114]:


data.iloc[:,32:48].head(10)


# In[115]:


data['term'] = data['term'].str.replace(' months', '')
data['term'] = data['term'].astype(float)


# In[116]:


data['emp_length_num'] = data['emp_length'].str.replace('\+ years', '')
data['emp_length_num'] = data['emp_length_num'].str.replace('< 1 year', str(0))
data['emp_length_num'] = data['emp_length_num'].str.replace(' years', '')
data['emp_length_num'] = data['emp_length_num'].str.replace(' year', '')

data['emp_length_num'] = data['emp_length_num'].astype(float)
data.drop('emp_length', axis=1, inplace=True)


# In[117]:


data['issue_d'] = pd.to_datetime(data['issue_d'], format='%b-%y')
data['issue_d_new'] = round(pd.to_numeric((pd.to_datetime('2023-06-03') - data['issue_d']) / np.timedelta64(1, 'M')))


# In[ ]:


#any(data['issue_d_new'] < 0)


# In[118]:


data.drop('issue_d', axis=1, inplace=True)


# In[119]:


data['earliest_cr_line'] = pd.to_datetime(data['earliest_cr_line'], format='%b-%y')
data['earliest_cr_line_new'] = round(pd.to_numeric((pd.to_datetime('2023-06-03') - data['earliest_cr_line']) / np.timedelta64(1, 'M')))


# In[ ]:


#any(data['earliest_cr_line_new'] < 0)


# In[120]:


data.loc[data['earliest_cr_line_new'] < 0, 'earliest_cr_line_new'] = data['earliest_cr_line_new'].max()


# In[121]:


data.drop(['earliest_cr_line'], axis=1, inplace=True)


# In[122]:


data['last_pymnt_d'] = pd.to_datetime(data['last_pymnt_d'], format='%b-%y')
data['last_pymnt_d_new'] = round(pd.to_numeric((pd.to_datetime('2023-06-03') - data['last_pymnt_d']) / np.timedelta64(1, 'M')))


# In[ ]:


#any(data['last_pymnt_d_new'] < 0)


# In[123]:


data.drop('last_pymnt_d', axis=1, inplace=True)


# In[124]:


data['last_credit_pull_d'] = pd.to_datetime(data['last_credit_pull_d'], format='%b-%y')
data['last_credit_pull_d_new'] = round(pd.to_numeric((pd.to_datetime('2023-06-03') - data['last_credit_pull_d']) / np.timedelta64(1, 'M')))


# In[ ]:


#any(data['last_credit_pull_d_new'] < 0)


# In[125]:


data.drop('last_credit_pull_d', axis=1, inplace=True)


# In[126]:


num = data.select_dtypes(exclude='object')  #num = numerik
cat = data.select_dtypes(include='object')  #cat = kategorik


# In[127]:


col_cat = cat.columns
col_num = num.columns

for i in col_cat:
    df = data[i]
    print('variable nunique for {} : {}'.format(i, df.nunique()))
    
print('\n')

for j in col_num:
    df = data[j]
    print('variable nunique for {} : {}'.format(j, df.nunique()))


# In[128]:


col_cat = cat.columns

for i in col_cat:
    df = data[i]
    print('distribution of label on variable {} : \n{}'.format(i, df.value_counts()))
    print('\n')


# In[129]:


data.drop(['policy_code', 'application_type', 'emp_title', 'title', 'pymnt_plan'],axis = 1, inplace= True)


# In[130]:


num = data.select_dtypes(exclude='object')
num_miss = num.isna().mean() * 100
col_num_miss = num_miss[num_miss > 0].index
print(len(col_num_miss))

cat = data.select_dtypes(include='object')
cat_miss = cat.isna().mean() * 100
col_cat_miss = cat_miss[cat_miss > 0].index
print(len(col_cat_miss))


# In[131]:


plt.figure(figsize=(30, 20))
for i,j in enumerate(col_num_miss):
    plt.subplot(4, 4, i+1)
    sns.distplot(num[j],color='blue')
    plt.tight_layout()


# In[132]:


count_by_median =  ['open_acc', 'revol_util', 'total_acc', 'tot_cur_bal', 'earliest_cr_line_new', 'last_pymnt_d_new', 'last_credit_pull_d_new']
count_by_modus = ['annual_inc', 'delinq_2yrs', 'inq_last_6mths', 'pub_rec', 'acc_now_delinq', 'tot_coll_amt', 'total_rev_hi_lim', 'emp_length_num', 'collections_12_mths_ex_med']

for i in count_by_median:
    data.loc[data.loc[:,i].isnull(),i] = data.loc[:,i].median()

for j in count_by_modus:
    data.loc[data.loc[:,j].isnull(),j] = data.loc[:,j].mode()[0]
    
data.isna().mean() * 100


# In[133]:


from scipy import stats

num = data.select_dtypes(exclude='object')
cat = data.select_dtypes(include='object')

num = num[(np.abs(stats.zscore(num)) < 3).all(axis=1)]
data = num.join(cat)
data.shape


# In[134]:


num = data.select_dtypes(exclude='object')
for j in num.columns:
    df = data[j]
    print('variable nunique for {} : {}'.format(j, df.nunique()))


# In[135]:


convert_col = ['term','pub_rec','delinq_2yrs','inq_last_6mths']

for i in convert_col:
    data[i] = data[i].astype(str)
    
data.drop(['collections_12_mths_ex_med', 'acc_now_delinq'], axis = 1, inplace =True)


# # Exploratory Data Analysis

# In[136]:


data.loc[data['status_loan_bad']==0,'status']='Success'
data.loc[data['status_loan_bad']==1,'status']='Failed'

num = data.select_dtypes(exclude='object')
cat = data.select_dtypes(include='object')


# In[137]:


col = num.columns

plt.figure(figsize=(40,20))
for i in range(0, len(col)):
    plt.subplot(6, 6, i+1)
    sns.distplot(num[num.columns[i]],color='green')
    plt.tight_layout()


# In[138]:


num.describe().T


# In[139]:


data['total_rec_late_fee_new'] = np.where(data['total_rec_late_fee']==0, 'Not Paid', 'Paid')
data['collection_recovery_fee_new'] = np.where(data['collection_recovery_fee']==0, 'Not Paid', 'Paid')
data['tot_coll_amt_new'] = np.where(data['tot_coll_amt']==0, 'Not Paid', 'Paid')
data['recoveries_new'] = np.where(data['recoveries']==0, 'Not Paid', 'Paid')


# In[140]:


data.drop(['total_rec_late_fee','recoveries','collection_recovery_fee','tot_coll_amt'], axis=1, inplace=True)


# In[141]:


col = cat.columns

plt.figure(figsize=(30, 20))
for i in range(0, len(col)):
    plt.subplot(4, 3, i+1)
    sns.countplot(cat[cat.columns[i]],color='yellow')
    plt.tight_layout()


# In[142]:


data['home_ownership'].replace({'NONE':'RENT', 'ANY':'RENT', 'OTHER':'RENT'},inplace=True)

data['purpose'].replace({'educational':'major_purchase',
                         'house':'major_purchase',
                         'medical':'major_purchase',
                         'moving':'major_purchase',
                         'vacation':'other',
                         'wedding':'other',
                         'renewable_energy':'home_improvement'},inplace=True)

data['addr_state'].replace({'IA':'OTHER', 'ID':'OTHER', 'NE':'OTHER', 'ME':'OTHER'},inplace=True)


# UNIVARIAT

# In[144]:


num = data.select_dtypes(exclude='object').drop('status_loan_bad',axis=1)
col = num.columns
data["all"] = ""

plt.figure(figsize=(40, 20))
for i,j in enumerate(col):
    plt.subplot(5, 5, i+1)
    sns.violinplot(x="all", y=j, hue="status", data=data, split=True)
    plt.xlabel("")
    plt.legend(loc="lower center", bbox_to_anchor=(0.5, 1), ncol=2)
    plt.tight_layout()

data.drop('all', axis = 1, inplace = True)


# In[146]:


cat = data.select_dtypes(include='object').drop(['status','purpose','addr_state'],axis=1)
col = cat.columns

fig = plt.figure(figsize=(40, 30))
for i,j in enumerate(col, start=1):
    ax = plt.subplot(4, 3, i)
    pd.crosstab(data[j], data['status']).sort_values(by=['Success']).plot.bar(stacked=True,ax=ax)
    plt.legend(loc="lower center", bbox_to_anchor=(0.5, 1), ncol=2)
    plt.xticks(rotation='0')


# In[147]:


data.drop('collection_recovery_fee_new', axis=1, inplace=True)


# In[148]:


fig = plt.figure(figsize=(30, 20))
ax = plt.subplot(121)
pd.crosstab(data['purpose'], data['status']).sort_values(by=['Success']).plot.barh(stacked=True,ax=ax)
plt.legend(loc="lower center", bbox_to_anchor=(0.5, 1), ncol=2)
plt.xticks(rotation='0')

ax = plt.subplot(122)
pd.crosstab(data['addr_state'], data['status']).sort_values(by=['Success']).plot.barh(stacked=True,ax=ax)
plt.legend(loc="lower center", bbox_to_anchor=(0.5, 1), ncol=2)
plt.xticks(rotation='0')


# In[ ]:


BIVARIAT


# Anova Test

# In[149]:


x = data.select_dtypes(exclude='object').drop('status_loan_bad',axis=1)
y = data['status']
num_col = x.columns

from sklearn.feature_selection import f_classif
f_value, p_value = f_classif(x,y)
for i,j in enumerate(num_col): 
    print('p value variable for {} = {}'.format(j, round(p_value[i], 3)))


# Chi Square Test

# In[150]:


x = data.select_dtypes(include='object').drop('status',axis=1)
y = data['status']
cat_col = x.columns

from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import chi2
for i in cat_col:
    cat_x = LabelEncoder().fit_transform(data[i]).reshape(-1,1)
    f_value, p_value = chi2(cat_x,y)
    print('p value variable for {} = {}'.format(i, p_value))
    #print(i, p_value)


# In[151]:


data.drop(['pub_rec'], axis=1, inplace=True)


# MULTIVARIAT

# In[155]:


df = data.select_dtypes(exclude='object').drop('status_loan_bad',axis=1)

kor_label = df.corr()
mask = np.zeros_like(kor_label, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

f, ax = plt.subplots(figsize=(40, 30))
cmap = sns.diverging_palette(220, 10, as_cmap=True)
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.9, center=0, annot=True, square=True, linewidths=.5, cbar_kws={"shrink": .5})


# In[157]:


matrix_kor = data.corr().abs()
upper = matrix_kor.where(np.triu(np.ones(matrix_kor.shape), k=1).astype(np.bool))
kor_up = [column for column in upper.columns if any(upper[column] > 0.7)]
kor_up


# In[159]:


data.drop(kor_up, axis=1, inplace=True)


# # Data Preprocessing

# Label Encoding

# In[160]:


data['grade'] = data['grade'].astype('category').cat.codes


# One Hot Encoding

# In[161]:


cat = data.select_dtypes(include='object').drop('status',axis=1)
cat_col = cat.columns.tolist()

one_hot = pd.get_dummies(data[cat_col], drop_first=True)
cath = pd.concat([one_hot, data['grade']],axis=1)


# Data Partition

# In[162]:


from sklearn.model_selection import train_test_split

num = data.select_dtypes(exclude='object')
data_set = pd.concat([num, cath],axis=1)

X = data_set.drop('status_loan_bad', axis = 1)
y = data_set['status_loan_bad']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)


# Transformation

# In[164]:


from sklearn.preprocessing import MinMaxScaler, FunctionTransformer

min_max = ['dti', 'open_acc', 'revol_util', 'total_acc', 'earliest_cr_line_new']
log_col = ['loan_amnt', 'int_rate', 'annual_inc', 'revol_bal', 'out_prncp', 'last_pymnt_amnt',
           'tot_cur_bal', 'total_rev_hi_lim', 'emp_length_num', 'issue_d_new',
           'last_pymnt_d_new', 'last_credit_pull_d_new']

min_max_SC = MinMaxScaler()
X_train.loc[:, min_max] = min_max_SC.fit_transform(X_train.loc[:, min_max])
X_test.loc[:, min_max] = min_max_SC.transform(X_test.loc[:, min_max])

X_train.loc[:, log_col] = np.log1p(X_train.loc[:, log_col])
X_test.loc[:, log_col] = np.log1p(X_test.loc[:, log_col])


# Balance Target

# In[165]:


y_train.value_counts()


# In[169]:


pip install --upgrade scikit-learn


# In[171]:


from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from collections import Counter

under_sampler = RandomOverSampler(random_state=123)

X_resampled, y_resampled = under_sampler.fit_resample(X_train.values, y_train.ravel())
Counter(y_resampled)


# In[172]:


count_col = X_train.columns.to_list()

X_train = pd.DataFrame(X_resampled, 
             columns = count_col)

y_train = pd.Series(y_resampled)


# # Training ML

# Compare Algorithm

# In[173]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

random_forest = RandomForestClassifier()
random_forest.fit(X_train, y_train)
y_pred_random_forest = random_forest.predict(X_train)
y_pred_random_forest_test = random_forest.predict(X_test)

logis_regres = LogisticRegression()
logis_regres.fit(X_train, y_train)
y_pred_logis_regres = logis_regres.predict(X_train)
y_pred_logis_regres_test = logis_regres.predict(X_test)

decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train, y_train)
y_pred_decision_tree = decision_tree.predict(X_train)
y_pred_decision_tree_test = decision_tree.predict(X_test)


# In[174]:


from sklearn.metrics import accuracy_score

machine_learning = ['RandomForest','LogisticRegression', 'DecisionTree']
data_train_predict = [y_pred_random_forest, y_pred_logis_regres, y_pred_decision_tree]
data_test_predict = [y_pred_random_forest_test, y_pred_logis_regres_test, y_pred_decision_tree_test]

train_accuracy = []
test_accuracy = []

for i in data_train_predict:
    train_accuracy.append(accuracy_score(y_train, i))

for i in data_test_predict:
    test_accuracy.append(accuracy_score(y_test, i))

n = list(zip(machine_learning, train_accuracy, test_accuracy))
pd.DataFrame(n, columns = ['Model','Train Accuracy', 'Test Accuracy']).sort_values(['Train Accuracy'],ascending=False)


# In[175]:


feature = pd.Series(random_forest.feature_importances_, index=X_train.columns).sort_values(ascending=False)
plt.figure(figsize=(20, 40))
feature.plot(kind='barh', title='Priority Feature On Random Forest')


# In[176]:


X_train.columns


# In[177]:


new_col = ['last_pymnt_d_new','last_pymnt_amnt','recoveries_new_Paid','out_prncp','last_credit_pull_d_new','issue_d_new',
       'int_rate','annual_inc','loan_amnt','grade','tot_cur_bal','dti','total_rev_hi_lim','revol_bal','revol_util','earliest_cr_line_new',
       'total_acc','open_acc','emp_length_num','total_rec_late_fee_new_Paid','term_60.0','initial_list_status_w','home_ownership_RENT']


# Overfitting Check

# In[178]:


X_train_2 = X_train[new_col]
X_test_2 = X_test[new_col]

random_forest = RandomForestClassifier()
random_forest.fit(X_train_2, y_train)
y_pred_random_forest = random_forest.predict(X_train_2)
y_pred_random_forest_test = random_forest.predict(X_test_2)

print('Train Accuracy',accuracy_score(y_train, y_pred_random_forest))
print('Test Accuracy',accuracy_score(y_test, y_pred_random_forest_test))


# # Model Evaluation

# In[188]:


from sklearn.metrics import confusion_matrix

matrix_confusion = confusion_matrix(y_test, y_pred_random_forest_test)

print('\nTrue Positives(TP) = ', matrix_confusion[0,0])
print('True Negatives(TN) = ', matrix_confusion[1,1])
print('False Positives(FP) = ', matrix_confusion[0,1])
print('False Negatives(FN) = ', matrix_confusion[1,0])
print('\n')

new_matrix = pd.DataFrame(data=matrix_confusion, columns=['False Negative', 'True Negative'], 
                                 index=['True Positive', 'False Positive'])

sns.heatmap(new_matrix, annot=True, fmt='d', cmap = 'Reds')


# In[189]:


TP = matrix_confusion[0,0] 
TN = matrix_confusion[1,1] 
FP = matrix_confusion[0,1] 
FN = matrix_confusion[1,0] 


# In[190]:


accuracy = (TP+TN) / float(TP+TN+FP+FN)
print('Accuracy : {0:0.4f}'.format(accuracy))

precision = TP / float(TP + FP)
print('\nPrecision: {0:0.4f}'.format(precision))

recall = TP / float(TP + FN)
print('\nRecall : {0:0.4f}'.format(recall))


# ROC - AUC Curve

# In[192]:


from sklearn.metrics import roc_curve, precision_recall_curve, auc

FP_rate, TP_rate, thresholds = roc_curve(y_test, y_pred_random_forest_test)
roc_auc_curve = auc(FP_rate, TP_rate)
print(roc_auc_curve)


# In[193]:


plt.figure(figsize=(10 , 10))

plt.plot(FP_rate,TP_rate, color='orange',label = 'AUC = %0.2f' % roc_auc_curve)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],linestyle='--')
plt.axis('tight')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')


# In[ ]:


#Good Classification

