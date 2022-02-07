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

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# %matplotlib inline
plt.set_cmap(plt.cm.Paired)

# pandasでの読み込み
df = pd.read_csv("./2D_example.csv" , header=None )
df.head()

# numpy での読み込み
data = np.loadtxt("./2D_example.csv" , delimiter= ',')

# 1列目の取得
label = data[:, 0].astype(int)
label

# 2,3列目に取得
x = data[:, 1:3]
x

# 実際にプロットしてみる。
plt.scatter(x[:,0] , x[:,1] , c = label , s=50)



# +
# k近傍法での識別(近くのデータが何であるかで識別する。)
from sklearn import neighbors

# オブジェクトの作成
clf = neighbors.KNeighborsClassifier(n_neighbors=1)

# 学習
clf.fit(x, label) 

plt.scatter(x[:, 0], x[:, 1], marker='o', s=50, c=label, edgecolors='k')
plotBoundary(x, clf) # 境界線の描画


# +
# 境界線を引く関数の定義

def plotBoundary(X, clf, mesh=True, boundary=True, n_neighbors=1):

    # plot range
    x_min = min(X[:,0])
    x_max = max(X[:,0])
    y_min = min(X[:,1])
    y_max = max(X[:,1])

    # visualizing decision function
    XX, YY = np.mgrid[x_min:x_max:200j, y_min:y_max:200j] # make a grid
    
    Z = clf.predict(np.c_[XX.ravel(), YY.ravel()]) # evaluate the value 
    
    Z = Z.reshape(XX.shape) # just reshape

    if mesh:
        plt.pcolormesh(XX, YY, Z, zorder=-10) # paint in 2 colors, if Z > 0 or not

    if boundary:
        plt.contour(XX, YY, Z, 
                    colors='k', linestyles='-', levels=[0.5])

    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
# -



#
# #### 癌のデータでやってみる。
#
#
#
#

from sklearn.datasets import load_breast_cancer

data = load_breast_cancer()

X = data.data

# データの大きさを確認
# 569のデータで30個の特徴量
X.shape

# それぞれのデータの特徴量に関して
data.feature_names

y = data.target

# 569個のデータが入っている。
y.shape

data.target_names

# +
# ロジスティック回帰で分類モデルを作成してみる。
from sklearn import linear_model

# 識別器の作成
clf = linear_model.LogisticRegression()
# -

clf

# 学習データとテストデータに分ける.（今回は半分ずつに分けて行く。）
n_samples = X.shape[0]
n_train = n_samples // 2
n_test = n_samples - n_train

train_index = range(0, n_train)
test_index = range(n_train , n_samples)

# +
# データを訓練用とテスト用に分けてみる。
X_train = X[train_index ]
X_test  = X[test_index]

y_train = y[train_index]
y_test = y[test_index]
# -

# 識別器の学習
clf.fit(X_train, y_train)

# 96% 成功している。
clf.score(X_train, y_train)

# テストデータのスコア確認
clf.score(X_test, y_test)

# データの予測に関して
clf.predict(X_test)






