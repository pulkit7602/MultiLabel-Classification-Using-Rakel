import pandas as pd
import warnings
import random
import itertools
from sklearn.model_selection import KFold
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.impute import SimpleImputer  # Import the imputer

warnings.filterwarnings("ignore")

dset = pd.read_csv("db3_yeast.csv")
print(dset.shape)

k = 3
kp2 = 2 ** k
print("2 ^ K : ", kp2)
pwrset = [x for x in range(kp2)]
print("Length of Power Set : ", len(pwrset))
print("Power Set : ", pwrset)
pwrset_elements = list(map(list, itertools.product([0, 1], repeat=k)))
print("Power Set Elements : ", pwrset_elements)
print("Length of Power Set : ", len(pwrset_elements))

dset_features = dset.iloc[:, 1:104]
features = dset_features.columns
dset_classes = dset.iloc[:, 104:]
classes = dset_classes.columns
print("Attributes : ", features)
print("Labels : ", classes)

m = int(len(classes) / 3)
print("Label Sets of size K : ", m)
m1 = int((len(classes) % k) / k) + 1
print("Label Sets of Pendingsize LESS THAN K if any : ", m1)
models = m + m1
print("Total Models to Build : ", models)

rand_lbl = random.sample(range(0, len(classes)), len(classes))
print(rand_lbl)

labelsets = []
for i in range(0, len(classes), k):
    set_i = rand_lbl[i: i + k]
    lset = []
    set_i.sort()

    for j in set_i:
        lset.append(classes[j])

    labelsets.append(lset)

print("Label Sets : ", labelsets)

model_dset = []
for ls in labelsets:
    pd_i = pd.DataFrame()

    for f in features:
        pd_i[f] = dset[f].values

    for c in ls:
        pd_i[c] = dset[c].values

    model_dset.append(pd_i)

overallf1ma = []
overallf1mi = []
overalljma = []
overalljmi = []

for mod_dset in model_dset:
    print("Size of Dataset to Model in RAkEL is : ", mod_dset.shape)

    mod_dset_f = mod_dset.iloc[:, 0:103]
    mod_dset_l = mod_dset.iloc[:, 103:]

    ftr = mod_dset_f.columns
    lbls = mod_dset_l.columns
    print("Attributes : ", ftr)
    print("Labels : ", lbls)

    c_id = []
    comb_id = []
    for i in range(len(mod_dset_l)):
        c_comb = mod_dset_l.iloc[i, :].values.tolist()
        if c_comb in pwrset_elements:
            c_id.append(pwrset_elements.index(c_comb))
            comb_id.append(c_comb)
        else:
            c_id.append(-1)  # Append a placeholder value for combinations not found in pwrset_elements

    unq_c_id = []
    unq_comb = []

    for temp_id in c_id:
        if temp_id != -1 and temp_id not in unq_c_id:
            unq_c_id.append(temp_id)

    for temp_comb in comb_id:
        if temp_comb not in unq_comb:
            unq_comb.append(temp_comb)

    print("Label Combinations : ", unq_comb)
    print("Class ID : ", unq_c_id)

    dset_mcc = pd.DataFrame()
    for fid in ftr:
        dset_mcc[fid] = mod_dset[fid].values

    dset_mcc["cid"] = c_id
    print("Shape of MCC Data Set : ", dset_mcc.shape)

    # Handle missing values by imputing with the mean
    imputer = SimpleImputer(strategy='mean')
    X = imputer.fit_transform(dset_mcc.iloc[:, 0:103])
    Y = dset_mcc.iloc[:, 103:]

    repeat = 3
    folds = 5
    f1ma = []
    jma = []
    f1mi = []
    jmi = []

    MCC_Model = OneVsRestClassifier(LogisticRegression())

    for lp in range(repeat):
        print("Repeat : ", lp)
        kfold = KFold(n_splits=folds, shuffle=True)

        for tr_id, tst_id in kfold.split(X):
            print("##### FOLD #####")

            X_Tr, X_Tst = X[tr_id, :], X[tst_id, :]
            Y_Tr, Y_Tst = Y.iloc[tr_id, :], Y.iloc[tst_id, :]

            MCC_Model.fit(X_Tr, Y_Tr)
            Output = MCC_Model.predict(X_Tst)

            f1ma_i = metrics.f1_score(Y_Tst, Output, average="macro")
            f1mi_i = metrics.f1_score(Y_Tst, Output, average="micro")
            jma_i = metrics.jaccard_score(Y_Tst, Output, average="macro")
            jmi_i = metrics.jaccard_score(Y_Tst, Output, average="micro")

            f1ma.append(f1ma_i)
            f1mi.append(f1mi_i)
            jma.append(jma_i)
            jmi.append(jmi_i)

    print("Macro F1 : ", f1ma)
    avg_f1ma = sum(f1ma) / len(f1ma)
    print("Average Macro F1 : ", avg_f1ma)
    print("Micro F1 : ", f1mi)
    avg_f1mi = sum(f1mi) / len(f1mi)
    print("Average Micro F1 : ", avg_f1mi)
    print("Macro Jaccard : ", jma)
    avg_jma = sum(jma) / len(jma)
    print("Average Macro Jaccard : ", avg_jma)
    print("Micro Jaccard : ", jmi)
    avg_jmi = sum(jmi) / len(jmi)
    print("Average Micro Jaccard : ", avg_jmi)

    overallf1ma.append(avg_f1ma)
    overallf1mi.append(avg_f1mi)
    overalljma.append(avg_jma)
    overalljmi.append(avg_jmi)

print("Macro F1 Score : ", (sum(overallf1ma) / models))
print("Micro F1 Score : ", (sum(overallf1mi) / models))
print("Macro Jaccard Score : ", (sum(overalljma) / models))
print("Micro Jaccard Score : ", (sum(overalljmi) / models))
