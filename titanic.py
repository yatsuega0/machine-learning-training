import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# test
test = pd.read_csv('./testset/test.csv')
test

# train
train = pd.read_csv('./testset/train.csv')
train

# test train 連結
df = pd.concat([train, test], sort=False, axis="rows")

df.isna().sum()

# nameから敬称を取り出して特徴量に
df["Salutation"] = df["Name"].str.extract("([A-Za-z]+)\.", expand=False)                                  

df["Salutation"].replace(["Mme", "Ms"], "Mrs", inplace=True)                                              
df["Salutation"].replace("Mlle", "Miss", inplace=True)                                                    
df["Salutation"].replace(["Capt", "Col", "Dr", "Major", "Rev"], "Officer", inplace=True)                  
df["Salutation"].replace(["Countess", "Don", "Dona", "Jonkheer", "Lady", "Sir"], "Royalty", inplace=True)
# name変換
df.loc[df["Name"].str.contains("Mr. ") == True, "Name"] = 0
df.loc[df["Name"].str.contains("Miss. ") == True, "Name"] = 1
df.loc[df["Name"].str.contains("Mrs. ") == True, "Name"] = 2
df.loc[df["Name"].str.contains("Master. ") == True, "Name"] = 3
df.loc[df["Name"].str.contains("Dr. ") == True, "Name"] = 3
df.loc[df["Name"].str.contains("Rev. ") == True, "Name"] = 4
df.loc[df["Name"].str.contains("Col. ") == True, "Name"] = 5
df.loc[df["Name"].str.contains("Major. ") == True, "Name"] = 6
df.loc[df["Name"].str.contains("Jonkheer. ") == True, "Name"] = 7
df.loc[df["Name"].str.contains("Mme. ") == True, "Name"] = 8
df.loc[df["Name"].str.contains("Capt. ") == True, "Name"] = 9
df.loc[df["Name"].str.contains("Ms. ") == True, "Name"] = 10
df.loc[df["Name"].str.contains("Mlle. ") == True, "Name"] = 11
df.loc[df["Name"].str.contains("Don. ") == True, "Name"] = 12
df.loc[df["Name"].str.contains("Countess. ") == True, "Name"] = 13
df.loc[df["Name"].str.contains("Sir. ") == True, "Name"] = 14
df.loc[df["Name"].str.contains("Dona. ") == True, "Name"] = 15

# print(df["Name"].value_counts())

# Sex変換
df["Sex"] = df["Sex"].replace({"male":0, "female":1})
# print(df['Sex'].value_counts())

# Cabin type変換 >>> 欠損値は0
df["Cabin"] = df["Cabin"].fillna(0)
df.loc[df["Cabin"].str.contains("C") == True, "Cabin"] = 1
df.loc[df["Cabin"].str.contains("B") == True, "Cabin"] = 2
df.loc[df["Cabin"].str.contains("D") == True, "Cabin"] = 3
df.loc[df["Cabin"].str.contains("E") == True, "Cabin"] = 4
df.loc[df["Cabin"].str.contains("A") == True, "Cabin"] = 5
df.loc[df["Cabin"].str.contains("F") == True, "Cabin"] = 6
df.loc[df["Cabin"].str.contains("G") == True, "Cabin"] = 7
df.loc[df["Cabin"].str.contains("T") == True, "Cabin"] = 8

# print(df["Cabin"].value_counts())

# Embarked type変換 >>> 欠損値は最頻値Sに
df["Embarked"] = df["Embarked"].fillna("S")

df.loc[df["Embarked"] == "S", "Embarked"] = 0
df.loc[df["Embarked"] == "C", "Embarked"] = 1
df.loc[df["Embarked"] == "Q", "Embarked"] = 2

# print(df["Embarked"].value_counts())

# Age補完
# 年齢予測用 >>> ランダムフォレストにより予測
df_age = df.loc[:, ["Age", "Sex", "SibSp", "Parch", "Salutation"]]
df_age = pd.get_dummies(df_age, columns=["Sex", "Salutation"])

df_age_notnull = df_age[df_age["Age"].notna()] 
df_age_null = df_age[df_age["Age"].isna()]     

X = df_age_notnull.iloc[:, 1:]                   # 説明
y = df_age_notnull.iloc[:, 0]                    # 目的

pipeline = Pipeline([("scl", StandardScaler()),
                    ("est", RandomForestRegressor(random_state=0))])
pipeline.fit(X, y)
age_predicted = pipeline.predict(df_age_null.iloc[:, 1:])

df.loc[(df["Age"].isna()), "Age"] = age_predicted


# Fare >>> 欠損値0
df["Fare"] = df["Fare"].fillna(0)

# train test分割
train = df.iloc[:891]

X_test = df.iloc[891:]
X_test = X_test.drop(["Perished", "Ticket", "Salutation"], axis="columns")

# Xとyを定義
X_train = train.drop(["Perished", "Ticket", "Salutation"], axis="columns")
y_train = train["Perished"].astype(int)

print(X_train.head())
print(y_train.head())

# ランダムフォレストでtrain学習
model = RandomForestClassifier(n_estimators=200, random_state=71)
model.fit(X_train, y_train)

# モデル予測
y_pred = model.predict(X_test)
y_pred

# 正解データ読み込み
y_true = pd.read_csv('gender_submission.csv',  index_col="PassengerId")
y_true

# 効果（精度の確認）の確認
print(f"正答率： {accuracy_score(y_true=y_true, y_pred=y_pred)}")

# csvでデータ保存
submission = pd.DataFrame({"PassengerId":X_test["PassengerId"], "Perished":y_pred})
submission.to_csv("submission.csv", index=False)