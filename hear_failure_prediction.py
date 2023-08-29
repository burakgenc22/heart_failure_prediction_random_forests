# Kütüphane importları ve pd-set_option ayarları

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV, cross_validate
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, GradientBoostingClassifier, AdaBoostClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
import warnings
warnings.simplefilter(action="ignore")

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 170)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

df = pd.read_csv("heart_failure_clinical_records_dataset.csv")
df.head()
df.info()

def check_df(dataframe, head=5):
    print("######################### SHAPE ######################### ")
    print(dataframe.shape)
    print("######################### DTYPES ######################### ")
    print(dataframe.dtypes)
    print("######################### HEAD ######################### ")
    print(dataframe.head(head))
    print("######################### NA ######################### ")
    print(dataframe.isnull().sum())
    print("######################### QUANTİLES ######################### ")
    print(dataframe.quantile([0, 0.05, 0.5, 0.95, 0.99, 1]).T)

check_df(df)

def grab_col_names(dataframe, cat_th=10, car_th=20):
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].dtypes != "O"
                   and dataframe[col].nunique() < cat_th]
    cat_but_car= [col for col in dataframe.columns if dataframe[col].dtypes == "O"
                  and dataframe[col].nunique() > car_th]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observation : {dataframe.shape[0]}")
    print(f"Variables : {dataframe.shape[1]}")
    print(f"Cat_cols : {len(cat_cols)}")
    print(f"Num_cols : {len(num_cols)}")
    print(f"Cat_but_car : {len(cat_but_car)}")
    print(f"num_but_cat : {len(num_but_cat)}")
    return cat_cols, num_cols, cat_but_car

 cat_cols, num_cols, cat_but_car = grab_col_names(df)


#kategorik değişken analizi

def cat_summary(dataframe, cat_col, plot=False):
    print(pd.DataFrame({cat_col : dataframe[cat_col].value_counts(),
                        "RATIO" : dataframe[cat_col].value_counts() / len(dataframe) * 100}))
    if plot:
        sns.countplot(x=dataframe[cat_col], data=dataframe)
        plt.show(block=True)

for col in cat_cols:
    cat_summary(df, col, plot=True)


#nümerik değişken analizi

def num_summary(dataframe, num_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[num_col].describe(quantiles).T)

    if plot:
        dataframe[num_col].hist(bins=20)
        plt.xlabel(num_col)
        plt.title(num_col)
        plt.show(block=True)

for col in num_cols:
    num_summary(df, col, plot=True)


#nümerik değişkenlerin hedef değişkene göre analizi

def target_w_num(dataframe, target, num_col):
    print(dataframe.groupby(target)[num_col].mean(), end="\n\n\n")

for col in num_cols:
    target_w_num(df, "DEATH_EVENT", col )

#kategorik değişkenlerin hedef deişkene göre analizi

def target_w_cat(dataframe, target, cat_col):
    print(cat_col)
    print(pd.DataFrame({"target_mean" : dataframe.groupby(cat_col)[target].mean(),
                        "count" : dataframe[cat_col].value_counts(),
                        "ratio" : dataframe[cat_col].value_counts() / len(dataframe) * 100}), end="\n\n\n")


for col in cat_cols:
    target_w_cat(df,"DEATH_EVENT", col )


#korelasyon
f, ax = plt.subplots(figsize=[18, 13])
sns.heatmap(df[num_cols].corr(), annot=True, fmt=".2f", ax=ax, cmap="magma")
ax.set_title("Correlation Matrix", fontsize=20)
plt.show(block=True)


#özellik mühendisliği

##aykırı değer analizi

def outlier_thresholds(dataframe, variable, q1=0.05, q3=0.95):
    quartile1= dataframe[variable].quantile(q1)
    quartile3= dataframe[variable].quantile(q3)
    interquantile = quartile3 - quartile1
    low_limit = quartile1 - 1.5 * interquantile
    up_limit = quartile3 + 1.5 * interquantile
    return low_limit, up_limit

def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False

for col in num_cols:
    print(col, check_outlier(df, col))

def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(df, variable, q1=0.25, q3=0.75)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

for col in num_cols:
    replace_with_thresholds(df, col)

for col in num_cols:
    print(col, check_outlier(df, col))


##eksik değer analizi

def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]
    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / len(dataframe) * 100).sort_values(ascending=False)

    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=["n_miss", "ratio"])
    print(missing_df, end="\n\n\n")

    if na_name:
        return na_columns

na_columns = missing_values_table(df, na_name=True)


#modelleme

y = df["DEATH_EVENT"]
X = df.drop(["DEATH_EVENT"], axis=1)

##random forests

rf_model = RandomForestClassifier()
cv_results = cross_validate(rf_model, X, y, cv=10, scoring=["accuracy", "f1", "roc_auc"])
cv_results["test_accuracy"].mean()
cv_results["test_f1"].mean()
cv_results["test_roc_auc"].mean()

#rf_model: accuracy = 0.7756 , f1 = 0.5883 , roc_auc = 0.8933


##hiperparametre optimizasyonu

rf_model.get_params()

rf_params = {"max_depth": [5, 8, None],
             "max_features": [3, 5, 7, "auto"],
             "min_samples_split": [2, 5, 8, 15, 20],
             "n_estimators": [100, 200, 500]}

rf_best_grid = GridSearchCV(rf_model, rf_params, cv=10, n_jobs=-1, verbose=True).fit(X, y)

rf_final = rf_model.set_params(**rf_best_grid.best_params_, random_state=17).fit(X, y)

cv_results_1_1 = cross_validate(rf_final, X, y, cv=5, scoring=["accuracy", "f1", "roc_auc"])
cv_results_1_1["test_accuracy"].mean()
cv_results_1_1["test_f1"].mean()
cv_results_1_1["test_roc_auc"].mean()

#rf_final: accuracy = 0.6955 , f1 = 0.4662 , roc_auc = 0.8880

random_user = X.sample(1)
random_user

rf_final.predict(random_user)













































































































































































































































