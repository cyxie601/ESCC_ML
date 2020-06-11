rd=0
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import f_classif, f_regression
from sklearn.feature_selection import SelectKBest, SelectPercentile
from sklearn.metrics import roc_auc_score, confusion_matrix, roc_curve
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()

# Load data
data = pd.read_csv('.../ESCC_ML-master/robust_harmonized_features_cohort1.csv')
data.shape
y_train = data['pCR']
first_column = data.columns[0]
X_train= data.drop([first_column], axis=1)
                  
data1 = pd.read_csv('.../ESCC_ML-master/robust_harmonized_features_cohort2.csv')
data1.shape
y_test = data1['pCR']
first_column1 = data1.columns[0]
X_test = data1.drop([first_column1], axis=1)
X_train.shape, X_test.shape

features_ori = X_train.columns
features_ori_y = y_train.name
X_train_ori =X_train
y_train_ori =y_train
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
X_train= pd.DataFrame(X_train)
X_train.columns = features_ori
X_test= pd.DataFrame(X_test)
X_test.columns = features_ori

#Feature selection
#Univariate (top 20% best features predictive of a pCR response calculated by univariate analysis)
univariate = f_classif(X_train, y_train)
univariate = pd.Series(univariate[1])
univariate.index = X_train.columns
univariate.sort_values(ascending=False, inplace=True)
sel_ = SelectPercentile(f_regression, percentile=20).fit(X_train.fillna(0), y_train)
features_to_keep = X_train.columns[sel_.get_support()]
X_train = sel_.transform(X_train.fillna(0))
X_train= pd.DataFrame(X_train)
X_train.columns = features_to_keep
X_test = X_test.loc[:, X_train.columns]

X_train.shape,X_test.shape

#Recursive feature addition 
model_all_features =DecisionTreeClassifier(random_state=rd)
model_all_features.fit(X_train, y_train)
y_pred_test = model_all_features.predict_proba(X_test)[:, 1]

features = pd.Series(model_all_features.feature_importances_)
features.index = X_train.columns
features.sort_values(ascending=False, inplace=True)
features = list(features.index)

model_one_feature = SVC(kernel = 'rbf', random_state = 0,probability=True)#ori
model_one_feature.fit(X_train[features[0]].to_frame(), y_train)
y_pred_test = model_one_feature.predict_proba(X_test[features[0]].to_frame())[:, 1]
auc_score_first = roc_auc_score(y_train, y_pred_train)

print('doing recursive feature addition')
features_to_keep = [features[0]]
count = 1
for feature in features[1:]:
    print()
    print('testing feature: ', feature, ' which is feature ', count,
          ' out of ', len(features))
    count = count + 1
    model_int  = SVC(kernel = 'rbf', random_state = rd,probability=True)
    
    model_int.fit(
        X_train[features_to_keep + [feature] ], y_train)

    y_pred_train = model_int.predict_proba(
        X_train[features_to_keep + [feature] ])[:, 1]

    auc_score_int = roc_auc_score(y_train, y_pred_train)
    print('New Test ROC AUC={}'.format((auc_score_int)))
    print('All features Test ROC AUC={}'.format((auc_score_first)))

    diff_auc = auc_score_int - auc_score_first

    if diff_auc >= 0.02:
        print('Increase in ROC AUC={}'.format(diff_auc))
        print('keep: ', feature)
        print

        auc_score_first = auc_score_int
        features_to_keep.append(feature)
    else:
        print('Increase in ROC AUC={}'.format(diff_auc))
        print('remove: ', feature)
        print
print('DONE!!')
print('total features to keep: ', len(features_to_keep))

#Model construction
final_model= SVC(kernel = 'rbf', random_state = rd,probability=True)
final_model.fit(X_train[features_to_keep], y_train)

y_pred_train = final_model.predict_proba(X_train[features_to_keep])[:, 1]
y_pred_test = final_model.predict_proba(X_test[features_to_keep])[:, 1]
fpr,tpr,thres = roc_curve(y_train, y_pred_train)


# Classification
optimal = thres[np.argmax(tpr+(1-fpr))]


def GetMetrics(true_lab, prediction, thres=None):
    auc = roc_auc_score(y_true=true_lab, y_score=prediction)
    
    if thres:
        hlab = np.zeros(len(prediction))
        hlab[np.array(prediction) > thres] = 1
    else:
        hlab = np.round(prediction)
    
    cm = confusion_matrix(y_true=true_lab,y_pred=hlab)
    tn = cm[0,0]
    fn = cm[1,0]
    tp = cm[1,1]
    fp = cm[0,1]
    sen = tp/(tp+fn)
    spec = tn/(tn+fp)   
    acc = (tp+tn)/(tn+fn+tp+fp)
    
    return auc, acc, sen, spec,cm

int_auc,int_acc,int_sen,int_spec,int_cm = GetMetrics(y_train, y_pred_train, optimal)

ext_auc,ext_acc,ext_sen,ext_spec,ext_cm = GetMetrics(y_test, y_pred_test, optimal)

