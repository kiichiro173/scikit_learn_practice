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

# ### 特徴量に関して
# 大まかな流れ  
# データの取得→特徴量抽出→特徴選択→特徴変換→正規化→識別
#
# * 欠損値の取り扱いに関して  
# 欠損値があるデータは取り除くかもしくは何かしら穴埋めする必要がある。
# 穴埋めする際は、平均値を代入したり他の特徴量から予測値を代入したりする方法がある。
# →https://uribo.github.io/practical-ds/03/handling-missing-data.html
#
#
# * 外れ値に関して    
# 外れ値に関してはモデルを学習する際には取り除く事が大切になってくる。    
# なぜなら外れ値が入っているデータを使用して学習を行うと、正確な予測モデルが作れなくなってしまうから。  
# しかしながら、どれを外れ値とするのかはデータを見ながら議論が必要になってくる。（飛び抜けて大きな値があったとしてもそれが発生しうるものなのかそれともありえない値なのかによって学習する際のデータに組み込むかを考えなければならない。）  
# 外れ値だからと言って、すぐに取り除くと言う判断は良くない！！  
# 外れ値の取り扱いに関しては以下  
# https://www.codexa.net/python-outlier/  
#
# * テキストデータに関して    
# テキストデータはそのままではデータとして使用する事が出来ない。    
# そのため、テキストをベクトル（数値に変換する必要がある。）  
# CountVectorizer：出現する単語のカウントを特徴量にする手法  
# TfidfVectorizer:TF（単語の出現頻度）とIDF（単語のレア度）とを掛け合わせた  
# word2vec:https://ainow.ai/2021/04/08/254071/  
#
# * 特徴量の抽出に関して  
# 特徴量が何個がベストなのかはわからないので、特徴量の数を変更しながら確認する必要がある  
# →特徴量の個数ごとにテストのスコアと標準偏差を出して一番ベストなスコアを出した特徴量の数を選択すると良い。  
#
# 特徴量同士で相関がある場合はどちらか一方を使用すればよいと言う判断もする事ができる。  
#
# SelectKBestを使用して取得する方法がある。  
# カイ二乗検定はカテゴリカルデータを対象として検定手法  
# →独立性の検定とも言われている。  
# →https://best-biostatistics.com/contingency/chi-square.html  
#
# SelectKBestに関して  
# →https://sstudydays.com/python/selectkbest/  
# →https://aotamasaki.hatenablog.com/entry/2018/04/18/201127  
# →https://www.dskomei.com/entry/2018/03/20/003752  
#
# * 特徴量変換  
# それぞれの値の組合せ（足したり、引いたり、かけたり、割ったりしたもの）で作成する方法もある  
# 特徴量同士で相関がある場合はどちらか一方を使用すればよいと言う判断もする事ができる。  
# →もしくはPCA（主成分分析）を使用して次元圧縮を行なって使用する方法もあったりする。  
#
# * 正規化/標準化に関して  
# 機械学習のモデルを作成する際には標準化もしくは正規化が必要になってくる。  
# →なぜなら特徴量によってデータのスケールは違うため、他の特徴量とスケールを合わせるためにも正規化もしくは標準化は必要になってくる。  
# →https://qiita.com/yShig/items/dbeb98598abcc98e1a57
#
# min-maxでの正規化を行う方法もある。(最大値を１最小値を０に変換するやり方)
# →https://miningoo.com/1032/

import pandas as pd

# テキストからデータを取得する
data = np.loadtxt("2D_example_dame.csv" , delimiter=",")

y = data[: , 0].astype(int)
y

X = data[:,1:3].astype(float)
X
"""
memo
以下のデータを見ると欠損値が存在しているのがわかる。
"""

# +
# matplotlibの表示に関しては以下のサイトを参照
# https://qiita.com/nkay/items/d1eb91e33b9d6469ef51#32-%E6%95%A3%E5%B8%83%E5%9B%B3axesscatter
import matplotlib.pyplot as plt
plt.scatter(X[:,0],X[:,1] ,c=y )
plt.show()

"""
memo
以下プロットした結果を確認してみると、黄色の二つの値で外れ値になっていそうな値が存在するのがわかる。
"""
# -

# 欠損値の確認
np.isnan(X[:,0])# nanのところがTrueになる

~np.isnan(X[:,0]) # nanのところがFalseになる

~np.isnan(X[:,1]) # nanのところがFalseになる

# 特徴量が両方ともTrueのところを取得する
~np.isnan(X[:,0])  & ~np.isnan(X[:,1])

# 欠損値を排除
X1 = X[~np.isnan(X[:,0])  & ~np.isnan(X[:,1])]
y1 = y[~np.isnan(X[:,0])  & ~np.isnan(X[:,1])]

print(X1.shape)
print(y1.shape)

# 外れ値の削除(+/−10より大きいものは排除)
X2 = X1[(abs(X1[:,0])<10) & (abs(X1[:,1])<10)]
y2 = y1[(abs(X1[:,0])<10) & (abs(X1[:,1])<10)]

print(X2.shape)
print(y2.shape)

# 欠損値と外れ値を排除したデータを確認してみる。
plt.scatter(X2[:,0],X2[:,1] ,c=y2 )
plt.show()




# ## テキストデータの取り扱い

import urllib.request
# データのダウンロード
urllib.request.urlretrieve("http://www.gutenberg.org/files/11/11-0.txt", "allice.txt")

with open('allice.txt', 'r', encoding='UTF-8') as f:
    print(f.read()[710:1400])

# CountVectorizerの実装
from sklearn.feature_extraction.text import CountVectorizer
txt_vec = CountVectorizer(input="filename")

txt_vec.fit(["allice.txt"])

txt_vec.get_feature_names()[100:120]

# 単語の数の確認
len(txt_vec.get_feature_names())

# ベクトル化
allice_vec = txt_vec.transform(["allice.txt"])
# ３１２２次元のスパース行列になっている。
allice_vec 

allice_vec.shape

for word,count in zip(txt_vec.get_feature_names()[200:220], allice_vec[0, 200:220]):
    print(word, count)

# TF/IDFの実装
from sklearn.feature_extraction.text import TfidfVectorizer
# TfidfVectorizer
vec_tfidf = TfidfVectorizer()
sample = np.array(['Apple computer of the apple mark', 'linux computer', 'windows computer'])
X = vec_tfidf.fit_transform(sample)


pd.DataFrame(X.toarray(), columns=vec_tfidf.get_feature_names())



# ## 画像データに関して

#

from sklearn.datasets import load_sample_image
china = load_sample_image("china.jpg")

plt.imshow(china)

china.shape



# ### 特徴量の抽出に関して
#

# +
import numpy as np
from sklearn.model_selection import ShuffleSplit
from sklearn.datasets import load_breast_cancer

data = load_breast_cancer()

X = data.data
y = data.target


ss = ShuffleSplit(n_splits=1, 
                  train_size=0.8, 
                  test_size=0.2, 
                  random_state=0)

train_index, test_index = next(ss.split(X, y))

X_train, X_test = X[train_index], X[test_index]
y_train, y_test = y[train_index], y[test_index]
# -

data.feature_names

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
# カイ二乗検定を使用した変数選択(20個選択)
skb = SelectKBest(chi2 , k=20)

skb.fit(X_train , y_train)

# 20個の特徴量を取得
X_train_new = skb.transform(X_train)
print(X_train_new.shape)

# どの特徴量が使用されたのか確認
skb.get_support()
data.feature_names[skb.get_support()]

# +
# 特徴量が何個がベストなのかを実際に計算しても止めてみる。
from sklearn.model_selection import StratifiedKFold
from sklearn import linear_model
clf = linear_model.LogisticRegression(solver='liblinear')
k_range = np.arange(1, 31)
scores = []
std = []

for k in k_range:
    # 特徴量の数ごとのループ処理

    ss = StratifiedKFold(n_splits=10, 
                         shuffle=True, 
                         random_state=2)
    score = []
    for train_index, val_index in ss.split(X_train,
                                           y_train):
        # k分割交差検証のループ処理

        X_train2, X_val = X[train_index], X[val_index]
        y_train2, y_val = y[train_index], y[val_index]

        skb = SelectKBest(chi2, k=k)
        skb.fit(X_train2, y_train2)
        
        X_new_train2 = skb.transform(X_train2)
        X_new_val    = skb.transform(X_val)
        
        clf.fit(X_new_train2, y_train2)
        score.append( clf.score(X_new_val, y_val) )

    scores.append( np.array(score).mean() )
    std.append( np.array(score).std() )
    
scores = np.array(scores)
std = np.array(std)

# +
import matplotlib.pyplot as plt
# %matplotlib inline 

plt.plot(k_range, scores)
plt.errorbar(k_range, scores, yerr=std)
plt.ylabel("accuracy")
# -

# bestなスコアの変数の数
best_k = k_range[np.argmax(scores)]
print(best_k)
"""
いかにより今回は１９個の変数がベストであると判断する事ができる。
"""



# ## 次元圧縮に関して
#
#

from pandas.plotting import scatter_matrix
from sklearn.decomposition import PCA


data = load_breast_cancer()
df = pd.DataFrame(data.data[:, [0,2]],
                  columns=data.feature_names[[0,2]])
scatter_matrix(df, figsize=(3,3));
"""
下のデータを確認すると二つの特徴量に相関がある事がわかる→次元圧縮する。
"""

X = data.data[:, [0,2]]
y = data.target
names = data.feature_names[[0,2]]

# 変換する前のデータ
plt.scatter(X[:, 0], X[:, 1])
plt.xlim(0, 180)
plt.ylim(20, 200)
plt.xlabel(names[0])
plt.ylabel(names[1])

pca = PCA()
pca.fit(X)
X_new = pca.transform(X)

# 変換後のデータ
plt.scatter(X_new[:, 0], X_new[:, 1])
plt.ylim(-60, 120)



# ## 標準化
#

# +
data = load_breast_cancer()
X = data.data
y = data.target

from sklearn.model_selection import ShuffleSplit

ss = ShuffleSplit(n_splits=1, 
                  train_size=0.8, 
                  test_size=0.2, 
                  random_state=0)

train_index, test_index = next(ss.split(X, y))

X_train, X_test = X[train_index], X[test_index]
y_train, y_test = y[train_index], y[test_index]
# -

# 標準化を行う
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

scaler.fit(X_train)

# 標準化したデータに変換
X_train_scale = scaler.transform(X_train)

# 標準化なので平均０、分散１になっているか確認する。
print("平均値")
print(X_train_scale.mean())
print("分散")
print(X_train_scale.std())












