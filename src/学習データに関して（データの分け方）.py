# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.6
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# ### 学習データに関して

# 予測モデルを作成するときには全てのデータを学習に使用するのは良くない。
# →なぜなら、全てのデータを学習に使用してしまうと、未知のデータに対して上手く予測できているかどうかを評価（汎化性能）することができないからである。  
# →逆に学習のデータとテストデータを分けることで、作成したモデルが上手く予測できているモデルなのか、はたまた過学習により予測が上手くいっていないのかをテストデータを使用して判断することが出来る。  
#
# ・実際にはデータを分割する際には3つのデータに分ける必要がある。（学習用・テスト用・検証用）  
# →なぜなら、精度が良くないと判断するのはテスト用のデータの時で、パラメータの調整はテスト用データに対して調整している。そのため、テスト用のデータに対して精度が良くなるようにしているのであって、本当の未知のデータに対して精度がいいのか把握することが出来ない。そのため未知のデータに対して精度がいいのか最終的に確認するために検証用のデータが必要になってくる。
#
#
# ・データを分割するときの注意
# →データを分割する時はstratified（層化）に注意する必要がある。  
# →なぜなら、学習データにはクラス１のみでテストデータはクラス２のみであった場合、予測が上手くいかないからである。
# →このようにクラス分類などstratifiedを気にしなければならない場合は、クラスのバランスを気にしてデータを分割する必要がある。

# +
import numpy as np
from sklearn.datasets import load_breast_cancer
data = load_breast_cancer()
X = data.data
y = data.target

from sklearn import linear_model
clf = linear_model.LogisticRegression()

# -

data

# ### データを分割する方法
#
# 大きく3種類　  
#
# *　hold-out法  
# *　cross validation法  
# *　leave one out法  
#



# ### hold-out法
#
# 一言で言うとデータを学習データとテストデータに分割しモデルの制度を確かめる手法
# 例えば、データが100個ある場合は、６対４の割合で分割し、学習用データを60個、テストデータを４０個に分割する。
# データが多いときにはこちらを使用。
#
#
# ![image.png](attachment:image.png)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test  = train_test_split(X ,y , test_size=0.4)

# 全データのサイズ
print("全データサイズ")
print("Xデータ")
print(X.shape)
print()
print("yデータ")
print(y.shape)
print()

print("全データサイズ")
print("X_trainデータ")
print(X_train.shape)
print("X_testデータ")
print(X_test.shape)
print()
print("y_trainデータ")
print(y_train.shape)
print("y_testデータ")
print(y_test.shape)
print()



# +
# トレーニングデータで学習
clf.fit(X_train,y_train)
print("トレーニングデータの性能")
print(clf.score(X_train,y_train))

print()

print("テストデータの性能")
print(clf.score(X_test,y_test))
# -

# ### 層化分割法
#

# データのそれぞれのクラスに関して
one_list = []
zero_list = []
for target in y:
    if target == 0:
        zero_list.append(target)
    else:
        one_list.append(target)
print("それぞれのクラスの数")
print("０クラス")
print(len(zero_list)/len(y))
print("1クラス")
print(len(one_list)/len(y))

# stratifiedを考慮せずに分割した場合。
X_train, X_test, y_train, y_test  = train_test_split(X ,y , test_size=0.4)
one_list = []
zero_list = []
for target in y_train:
    if target == 0:
        zero_list.append(target)
    else:
        one_list.append(target)
print("stratifiedを考慮せずに分割した場合") 
print("それぞれのクラスの数")
print("０クラス")
print(len(zero_list)/len(y_train))
print("1クラス")
print(len(one_list)/len(y_train))



# +
# stratifiedを考慮した場合

X_train, X_test, y_train, y_test  = train_test_split(X ,y , test_size=0.4 ,stratify=y)
one_list = []
zero_list = []
for target in y_train:
    if target == 0:
        zero_list.append(target)
    else:
        one_list.append(target)

print("stratifiedを考慮した場合")
print("それぞれのクラスの数")
print("０クラス")
print(len(zero_list)/len(y_train))
print("1クラス")
print(len(one_list)/len(y_train))
# -





# ### cross validation法
#
# クロスバリデーション法(K-分割交差検証)はデータをK個に分割してそのうちの一つをテストデータにし、残りのk-1個のデータを学習用のデータに分割。その後、テストデータと学習用データを入れ替えてモデルの評価を図ると言うもの。つまり、K個に分解されたものはK回の検証が行われることになる。
# これらのデータを平均してモデルの精度を確かめる。
# データが10,000個以下ぐらいであればK分割交差検証を使用して行うのはあり。（データ量が多すぎると処理に時間がかかるため、ホールドアウト法を使用するか、クロスバリデーション法を使用するかは見極めが必要。。。）
#
# 以下の例では100個のデータに対して４分割(K=4)した場合の例
# ![image.png](attachment:image.png)

# +
from sklearn.model_selection import KFold
kf = KFold(n_splits=10 , shuffle=True)

for train , test in kf.split(X,y):
    X_train , X_test = X[train] , X[test]
    y_train , y_test = y[train] , y[test]
    print("学習データの確認")
    print(X_train.shape)
    print(y_train.shape)
    print()
    print("テストデータの確認")
    print(X_test.shape)
    print(y_test.shape)
    print()
    clf.fit(X_train,y_train)
    print("結果")
    print(clf.score(X_test,y_test))
    print()

# +
from sklearn.model_selection import  StratifiedKFold
skf = KFold(n_splits=10 , shuffle=True)

for train , test in skf.split(X,y):
    X_train , X_test = X[train] , X[test]
    y_train , y_test = y[train] , y[test]
    print("学習データの確認")
    print(X_train.shape)
    print(y_train.shape)
    print()
    print("テストデータの確認")
    print(X_test.shape)
    print(y_test.shape)
    print()
    clf.fit(X_train,y_train)
    print("結果")
    print(clf.score(X_test,y_test))
    print()
# -

#  クロスバリデーションのスコア(デフォルトで層化分類になっている。)
# １０分割を行なっている。
from sklearn.model_selection import cross_val_score
ave_score = cross_val_score(clf , X , y , cv=10)
print("平均スコア")
print(ave_score.mean()*100)
print(ave_score.std()*100)




# ### leave one out法
#
# データ全体のうち1つだけをテストデータとする方法。データの量が少ない場合に行う方法でもある。
# データが１００個あれば１つをテストデータとし、99個を学習データとして使用する。これを１００回繰り返して精度を検証する。
# データが少ない時はこちらを使用する。
#
# ![image.png](attachment:image.png)

# +
from sklearn.model_selection import LeaveOneOut

loo = LeaveOneOut()
for train , test in loo.split(X,y):
    X_train , X_test = X[train] , X[test]
    y_train , y_test = y[train] , y[test]
    print("学習データの確認")
    print(X_train.shape)
    print(y_train.shape)
    print()
    print("テストデータの確認")
    print(X_test.shape)
    print(y_test.shape)
    print()
    clf.fit(X_train,y_train)
    print("結果")
    print(clf.score(X_test,y_test))
    print()

# +
from sklearn.model_selection import LeaveOneOut

loo = LeaveOneOut()
from sklearn.model_selection import cross_val_score
ave_score = cross_val_score(clf , X , y , cv=loo)
print("平均スコア")
print(ave_score.mean()*100)
print(ave_score.std()*100)
# -


























