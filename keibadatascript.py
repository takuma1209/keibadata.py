import pandas as pd
import numpy as np

yearStart = 2010#開始年を入力
yearEnd = 2023#終了年を入力

yearList = np.arange(yearStart, yearEnd+1, 1, int) 
data=[]

for for_year in yearList:
    var_path ="C:/Users/bergt/OneDrive/デスクトップ/keibadeta/" +str(for_year)+".csv"
    var_data = pd.read_csv(var_path,encoding="SHIFT-JIS",header=None)
    data.append(var_data)

nameList,jockyList,umabanList,timeList,oddsList,passList,weightList,dWeightList,sexList,oldList,handiList,agariList,ninkiList = [],[],[],[],[],[],[],[],[],[],[],[],[]
umaList = [nameList,jockyList,umabanList,timeList,oddsList,passList,weightList,dWeightList,sexList,oldList,handiList,agariList,ninkiList]

raceNameList,dateList,courseList,classList,surfaceList,distanceList,rotationList,surCondiList,weatherList = [],[],[],[],[],[],[],[],[]
infoList=[raceNameList,dateList,courseList,classList,surfaceList,distanceList,rotationList,surCondiList,weatherList]

tanList,fukuList,umarenList,wideList,umatanList,renpukuList,rentanList = [],[],[],[],[],[],[]
paybackList = [tanList,fukuList,umarenList,wideList,umatanList,renpukuList,rentanList]


for for_year in range(len(data)):
    for for_race in range(len(data[for_year][0])):
        var_dataReplaced = data[for_year][0][for_race].replace(' ','').replace('[','').replace('\'','').split("]")
        var = var_dataReplaced[0].split(",")
        var_allNumber = len(var)#出走馬の数
        #馬の名前
        nameList.append(var)
        #騎手
        jockyList.append(var_dataReplaced[1].split(",")[1:])
        #馬番
        umabanList.append(list(map(int,var_dataReplaced[2].split(",")[1:])))
        #走破時間
        var = var_dataReplaced[3].split(",")[1:]
        var1 = []
        for n in range(var_allNumber):
            try:
                var2 = var[n].split(":")
                var1.append(float(var2[0])*60 + float(var2[1]))
            except ValueError:
                var1.append(var1[-1])#ひとつ前の値で補間する
        timeList.append(var1)
        #オッズ
        var = var_dataReplaced[4].split(",")[1:]
        var1 = []
        for n in range(var_allNumber):
            try:
                var1.append(float(var[n]))
            except ValueError:
                var1.append(var1[-1])#ひとつ前の値で補間する
        oddsList.append(var1)
        #通過
        var = var_dataReplaced[5].split(",")[1:]
        var1 = []
        for n in range(var_allNumber):
            try:
                var1.append(np.average(np.array(list(map(int,var[n].split("-")))))/var_allNumber)
            except ValueError:
                var1.append(var1[-1])#ひとつ前の値で補間する    
        passList.append(var1)
        #体重
        var = var_dataReplaced[6].split(",")[1:]
        var1 = []
        var2 = []
        for n in range(var_allNumber):
            try:
                var1.append(int(var[n].split("(")[0]))
                var2.append(int(var[n].split("(")[1][0:-1]))
            except ValueError:
                var1.append(var2[-1])
                var2.append(var2[-1])
        weightList.append(var1)
        dWeightList.append(var2)
        #性齢
        var = var_dataReplaced[7].split(",")[1:]
        var1 = []
        var2 = []
        for n in range(var_allNumber):
            var11 = var[n][0]
            if "牡" in var11:
                var1.append(0)
            elif "牝" in var11:
                var1.append(1)
            elif "セ" in var11:
                var1.append(2)
            else:
                print(var11)
            var2.append(int(var[n][1:]))
        sexList.append(var1)
        oldList.append(var2)
        #斤量
        handiList.append(list(map(float,var_dataReplaced[8].split(",")[1:])))
        #上がり
        var = var_dataReplaced[9].split(",")[1:]
        var1 = []
        for n in range(var_allNumber):
            try:
                var1.append(float(var[n]))
            except ValueError:
                var1.append(var1[-1])#ひとつ前の値で補間する
        agariList.append(var1)
        #人気
        var = var_dataReplaced[10].split(",")[1:]
        var1 = []
        for n in range(var_allNumber):
            try:
                var1.append(int(var[n]))
            except ValueError:
                var1.append(var1[-1])#ひとつ前の値で補間する
        ninkiList.append(var1)

        var_infoReplaced = data[for_year][1][for_race].replace(' ','').replace('[','').replace('\'','').split("]")[:-2]
        #レースの名前
        raceNameList.append(var_infoReplaced[0])
        #日付
        var = var_infoReplaced[1]
        var1 = var.split("年")
        var2 = var1[1].split("月")
        dateList.append((int(var1[0].replace(",",""))-yearStart)*365+int(var2[0])*30+int(var2[1].split("日")[0]))
        #競馬場
        var = var_infoReplaced[2]
        if "札幌" in var:
            courseList.append(0)
        elif "函館" in var:
            courseList.append(1)
        elif "福島" in var:
            courseList.append(2)
        elif "新潟" in var:
            courseList.append(3)
        elif "東京" in var:
            courseList.append(4)
        elif "中山" in var:
            courseList.append(5)
        elif "中京" in var:
            courseList.append(6)
        elif "京都" in var:
            courseList.append(7)
        elif "阪神" in var:
            courseList.append(8)
        elif "小倉" in var:
            courseList.append(9)
        else:
            print(var)
        #クラス
        var = var_infoReplaced[0]
        var1 = var_infoReplaced[3]
        if "障害" in var1:
            classList.append(0)
        elif "G1" in var:
            classList.append(10)
        elif "G2" in var:
            classList.append(9)
        elif "G3" in var:
            classList.append(8)
        elif ("(L)" in var) or ("オープン" in var1):
            classList.append(7)
        elif ("3勝" in var1) or ("1600" in var1):
            classList.append(6)
        elif ("2勝" in var1) or ("1000" in var1):
            classList.append(5)
        elif ("1勝" in var1) or ("500" in var1):
            classList.append(4)
        elif "新馬" in var1:
            classList.append(3)
        elif "未勝利" in var1:
            classList.append(2)
        else:
            print(var)
        #芝、ダート
        var = var_infoReplaced[4]
        if "芝" in var:
            surfaceList.append(0)
        elif "ダ" in var:
            surfaceList.append(1)
        elif "障" in var:
            surfaceList.append(2)
        else:
            print(var)
        #距離
        distanceList.append(int(var_infoReplaced[5].replace(",","")))
        #回り
        var = var_infoReplaced[6]
        if "右" in var:
            rotationList.append(0)
        elif "左" in var:
            rotationList.append(1)
        elif ("芝" in var) or ("直" in var):
            rotationList.append(2)
        else:
            print(var)
        #馬場状態
        var = var_infoReplaced[7]
        if "良" in var:
            surCondiList.append(0)
        elif "稍" in var:
            surCondiList.append(1)
        elif "重" in var:
            surCondiList.append(2)
        elif "不" in var:
            surCondiList.append(3)
        else:
            print(var)
        #天気
        var = var_infoReplaced[8]
        if "晴" in var:
            weatherList.append(0)
        elif "曇" in var:
            weatherList.append(1)
        elif "小" in var:
            weatherList.append(2)
        elif "雨" in var:
            weatherList.append(3)
        elif "雪" in var:
            weatherList.append(4)
        else:
            print(var)
        
        #単勝、複勝、馬連、ワイド、馬単、三連複、三連単の順番で格納
        var_paybackReplaced = data[for_year][2][for_race].replace('[','').replace(",","").replace('\'','').split("]")
        #単勝
        tanList.append(int(var_paybackReplaced[0].split(" ")[1]))
        #複勝
        var = var_paybackReplaced[1].split(" ")[1:]
        var_list = []
        for for_var in range(int(len(var)/2)):
            var_list.append(int(var[2*for_var+1]))
        fukuList.append(var_list)
        #馬連
        umarenList.append(int(var_paybackReplaced[2].split(" ")[-1]))
        #ワイド
        var = var_paybackReplaced[3].split(" ")[1:]
        var_list = []
        for for_var in range(int(len(var)/4)):
            var_list.append(int(var[4*for_var+3]))
        wideList.append(var_list)
        #馬単
        umatanList.append(int(var_paybackReplaced[4].split(" ")[-1]))
        #三連複
        renpukuList.append(int(var_paybackReplaced[5].split(" ")[-1]))
        #三連単
        rentanList.append(int(var_paybackReplaced[6].split(" ")[-1]))

data = []
for for_races in range(len(nameList)):
    var_list = []#uma,info,payback
    for for_lists in umaList:
        var_list.append(for_lists[for_races])
    for for_lists in infoList:
        var_list.append(for_lists[for_races])
    for for_lists in paybackList:
        var_list.append(for_lists[for_races])
    data.append(var_list)
data = sorted(data, key = lambda x: x[14],reverse = True)#日付が大きい順番に並べる
'''
data
第一指数：全レース数
第二指数：0~28でレースの情報
'''
#前走のインデックスのリストを生成する
dataGet = 20000#何レース分のデータを取得するか
pastRaces = 6000#過去何レース調べるか
pastResults = 5#前走何レースを参考にするか
pastIndexList = []
lenData = len(data)


var_name = [ 'c{0:02d}'.format(i) for i in range(18) ]#列名を先に作らないと読み込めない
var_pastIndex = np.array(pd.read_csv("C:/Users/bergt/OneDrive/デスクトップ/keibadeta2010_2022_index.csv",encoding="SHIFT-JIS",header=None,names = var_name))
#第一インデックス：レース数、　第二インデックス：馬番、　第三インデックス：過去何レース前で何着か
pastIndex = []
for for_races in range(len(var_pastIndex)):
    var0 = var_pastIndex[for_races]
    var_listUmaban = []
    for for_umaban in range(len(var0)):
        var1 = var0[for_umaban]
        if var1 is np.nan:#nanを削除するために
            continue
        else:
            var2 = var1.replace(", [","A").replace("[","").replace("]","").split("A")
        var_listPast = []
        for for_past in range(len(var2)):
            try:
                var_listPast.append(np.array(var2[for_past].replace(",","").split(" "),dtype=np.int64))
            except ValueError:#前走がない時、空のリストで埋める
                var_listPast.append([])
        var_listUmaban.append(var_listPast)
    pastIndex.append(var_listUmaban)









import itertools
xListBefore = []
yListBefore = []
oddsListBefore = []
oddsListBeforeFuku = []
umarenListBefore = []
umatanListBefore = []
sanrentanListBefore = []
sanrenpukuListBefore = []
wideListBefore = []
indexList = []
#pastIndex = pastIndexComped
var_addNum = 5#他の馬のタイム上位何頭分の過去データを説明変数に加えるか
decreaseRate = 1#ダミーへのペナルティー
need = 5#前何走使うか
otherNeed = 1#他の馬は前何走使うか

def makeXParamList(n):#n番目のレースの説明変数のリストを作る、後々のことを考えて関数化
    var_data = data[n]
    allNum = len(var_data[0])
    xNList = []
    yNList = []
    for nn in range(allNum):#nnは出走馬数に対応
        if len(pastIndex[n][nn][0]) == 0: #過去データが全くない馬が1頭でもいるか
            break
        if len(pastIndex[n][nn]) <= 0:#過去データが2以下の馬が1頭でもいるか
            break
        if allNum<var_addNum+2:#var_addNum+1いないと説明変数が足りないため
            break
    else:#過去データ数が条件を満たしていると確認した時
        for nn in range(allNum):
            var_list = []
            var_list.append(var_data[2][nn])#馬番
            var_list.append(var_data[9][nn])#年齢
            var_list.append(var_data[8][nn])#性
            var = var_data[6][nn]
            if var == 0:
                var = 480
            var_list.append(var)#体重
            var_list.append(var_data[7][nn])#体重変化
            var_list.append(var_data[10][nn])#斤量
            var_list.append(var_data[16])#クラス
            var_list.append(allNum)#出走馬数
            var_list.append(var_data[18])#距離
            var_list.append(var_data[17])#芝、ダート
            counter = 0#ダミーの枚数
            for nnn in range(need):#nnnは過去5レースに対応
                try:
                    var_ind = pastIndex[n][nn][nnn][0]
                    var_chakujun = pastIndex[n][nn][nnn][1]
                    var_data1 = data[var_ind]
                    var_allNum = len(data[var_ind][0])
                    if nnn == 0:
                        pass
                except IndexError:#ちなみにnnnはXXX以上であることは確定している
                    counter += 1
                    try:
                        var_ind = pastIndex[n][nn][counter-1][0]#最近のデータ、最近2番目のデータで補完
                        var_chakujun = pastIndex[n][nn][counter-1][1]
                        var_data1 = data[var_ind]
                        var_allNum = len(data[var_ind][0])
                    except IndexError:
                        var_ind = pastIndex[n][nn][0][0]#最近のデータで補完
                        var_chakujun = pastIndex[n][nn][0][1]
                        var_data1 = data[var_ind]
                        var_allNum = len(data[var_ind][0])
                var_list.append(var_chakujun)#着順
                var_list.append(var_data1[3][var_chakujun]*decreaseRate**counter)#タイム
                var_list.append(var_data1[5][var_chakujun])#通過
                var_list.append(var_data1[11][var_chakujun])#上がり
                var_list.append(var_data1[3][0]-var_data1[3][var_chakujun])#着差
                var_list.append(var_data[14]-var_data1[14])#前走からの日数
                var_list.append(var_data[18]-var_data1[18])#距離変化
            #他の馬の情報を入れていく
            var_meanTimeList = []
            var_timeList = []
            var_classList = []
            var_oddsList = []
            counterList = []
            for nnn in range(allNum):#他の馬数
                var_MTimeList = []
                var_MClassList = []
                var_MOddsList = []
                if nnn!=nn:#自分以外の時
                    counter = 0#ダミーの枚数
                    for m in range(need):#他の馬の過去5レース
                        try:
                            var_ind = pastIndex[n][nnn][m][0]
                            var_chakujun = pastIndex[n][nnn][m][1]
                            var_data1 = data[var_ind]
                            var_allNum = len(data[var_ind][0])
                        except IndexError:
                            counter+=1
                            try:
                                var_ind = pastIndex[n][nnn][counter-1][0]
                                var_chakujun = pastIndex[n][nnn][counter-1][1]
                                var_data1 = data[var_ind]
                                var_allNum = len(data[var_ind][0])
                            except IndexError:
                                var_ind = pastIndex[n][nnn][0][0]
                                var_chakujun = pastIndex[n][nnn][0][1]
                                var_data1 = data[var_ind]
                                var_allNum = len(data[var_ind][0])
                        var_MTimeList.append(var_data1[3][var_chakujun]*decreaseRate**counter)#タイム
                        var_MClassList.append(var_data1[16])#クラス
                    var_meanTimeList.append(sum(var_MTimeList)/need)#平均タイムを追加する
                    var_timeList.append(var_MTimeList)#5走のタイム
                    var_classList.append(var_MClassList)#5走のクラス
            var_timeList = [i for _,i in sorted(zip(var_meanTimeList,var_timeList))]#[i for _,i in sorted(zip(B,A))]#BをキーにしてAのリストを返す
            var_classList = [i for _,i in sorted(zip(var_meanTimeList,var_classList))]
            for nnn in range(var_addNum):
                for m in range(otherNeed):
                    var_list.append(var_timeList[nnn][m])
                    var_list.append(var_classList[nnn][m])
            xNList.append(var_list)
            yNList.append(var_data[3][nn]-sum(var_data[3])/allNum)
        xListBefore.append(xNList)
        yListBefore.append(yNList)
        oddsListBefore.append(var_data[4])
        oddsListBeforeFuku.append(var_data[23])
        umarenListBefore.append(var_data[24])
        umatanListBefore.append(var_data[26])
        sanrentanListBefore.append(var_data[28])
        sanrenpukuListBefore.append(var_data[27])
        wideListBefore.append(var_data[25])
        indexList.append(n)

    
for n in range(len(pastIndex)):#nはレース数に対応
    makeXParamList(n)
    print("\r" + str(n+1)+"/" + str(len(pastIndex)) + " 完了" , end="")

xList = list(itertools.chain.from_iterable(xListBefore))
yList = list(itertools.chain.from_iterable(yListBefore))

ratio = 0.4#テストに回す割合を入力
numTrain = int(len(xList)*ratio)
xTrain = xList[numTrain:]
yTrain = yList[numTrain:]
xTest = xList[:numTrain]
yTest = yList[:numTrain]
xTrainDf = pd.DataFrame(xTrain)
yTrainDf = pd.DataFrame(yTrain)
xTestDf = pd.DataFrame(xTest)
yTestDf = pd.DataFrame(yTest)





from sklearn.model_selection import KFold
import lightgbm as lbg
from sklearn.model_selection import ParameterGrid

kf = KFold(n_splits=5, shuffle=True, random_state=0)#データをn個に分割する、型はscikitlearnで使うデータ分割方法みたいなやつ

#https://lightgbm.readthedocs.io/en/latest/Parameters.html
# LightGBMのハイパーパラメータを設定
paramsGrid = {
          'learning_rate': [0.1],#学習率[0.01,0.1,0.001]
            'metric': ["l1"],
          'num_leaves': [40], #ノードの数
          'reg_alpha': [0.5], #L1正則化係数
          'reg_lambda': [0.5],#L2正則化係数
          'colsample_bytree': [1], #各決定木においてランダムに抽出される列の割合(大きいほど過学習寄り)、0~1でデフォ1
          'subsample': [0.1], # 各決定木においてランダムに抽出される標本の割合(大きいほど過学習寄り)、0~1でデフォ1
          'subsample_freq': [0],#, # ここで指定したイテレーション毎にバギング実施(大きいほど過学習寄りだが、0のときバギング非実施となるので注意)
          'min_child_samples': [20],# 1枚の葉に含まれる最小データ数(小さいほど過学習寄り)、デフォ20
          'seed': [0],
          'n_estimators': [110]}#, # 予測器(決定木)の数:イタレーション
          #'max_depth': [70,90]} # 木の深さ

paramsList = list(ParameterGrid(paramsGrid))
len(paramsList)

from sklearn.model_selection import cross_validate
import matplotlib.pyplot as plt

estimatorParams = paramsList#ハイパーパラメータのグリッド
kfLength = kf.get_n_splits()#データを何個に分割して交差検証を行うか
totalSample = len(estimatorParams)*kfLength
counter = 0
estList = []
scoreList = []
estimatorParamsList = []
grid = False#グリッドサーチを行うかどうか

if grid==True:
    for n in range(len(estimatorParams)):
        var_regressor = lbg.LGBMRegressor(**estimatorParams[n])
        print(estimatorParams[n])
        CVResults = cross_validate(var_regressor,xTrainDf,np.ravel(yTrainDf),cv=kf, return_estimator=True)#estimatorParams[n]を使用してcrossValidate #結果は長さkfのリスト
        estList.append(CVResults["estimator"][0])#どれも同じなので0番目を取った
        var = CVResults["test_score"].mean()
        scoreList.append(var)
        print("\n" + str(var))
        estimatorParamsList.append(estimatorParams[n])
        counter += kfLength
        print("\r" + str(counter)+"/" + str(totalSample) + " 完了" , end="")
    var_maxInd = scoreList.index(max(scoreList))#最大の精度のインデックス
    bestEst = estList[var_maxInd]#最大の精度を与える予測器
    bestParam = estimatorParamsList[var_maxInd]
    print(bestParam)
else:
    bestEst = lbg.LGBMRegressor(**estimatorParams[0])

#最適なハイパーパラメータでトレーニングとテストを行う
bestEst.fit(xTrainDf,np.ravel(yTrainDf))
predict = bestEst.predict(xTestDf)
print(bestEst.score(xTestDf,yTestDf))

import matplotlib
lbg.plot_tree(bestEst.booster_, tree_index=0);
fig = matplotlib.pyplot.gcf()
fig.set_size_inches(150, 100)
fig.savefig('tree.png')

import matplotlib
lbg.plot_tree(bestEst.booster_, tree_index=0);
fig = matplotlib.pyplot.gcf()
fig.set_size_inches(150, 100)
fig.savefig('tree.png')

#テストデータを使用したときのプロット
fig, ax = plt.subplots(figsize=(5,5))
plt.scatter(predict, yTest, s=1, alpha = 0.1)
plt.xlim(-3,3)
plt.ylim(-3,3)
ax.set_title('Light GBM')
ax.set_xlabel('Predict')
ax.set_ylabel('Actual')
plt.grid()
plt.show()

print("データ数: " + str(len(data)))


import pandas as pd
import sklearn.datasets as skd

data = skd.load_boston()

df_X = pd.DataFrame(data.data, columns=data.feature_names)
df_y = pd.DataFrame(data.target, columns=['y'])

df_X.info()

import lightgbm as lgb
from sklearn.model_selection import train_test_split

df_X_train, df_X_test, df_y_train, df_y_test = train_test_split(df_X, df_y, test_size=0.2, random_state=4)

lgb_train = lgb.Dataset(df_X_train, df_y_train)
lgb_eval = lgb.Dataset(df_X_test, df_y_test)

params = {
    'seed':4,
    'metric':'rmse'}

lgbm = lgb.train(params,
                lgb_train,
                valid_sets=lgb_eval,
                num_boost_round=200,
                early_stopping_rounds=20,
                verbose_eval=50)

import xgboost as xgb

dtrain = xgb.DMatrix(df_X_train, label=df_y_train)
dtest = xgb.DMatrix(df_X_test, label=df_y_test)

params = {
    'objective': 'reg:squarederror',
    'eval_metric': 'rmse',
}

evals = [(dtrain, 'train'), (dtest, 'eval')]
evals_result = {}

xgbm = xgb.train(params,
                dtrain,
                num_boost_round=200,
                early_stopping_rounds=20,
                evals=evals,
                evals_result=evals_result
                )


  


import lightgbm as lgb
from sklearn.model_selection import train_test_split

df_X_train, df_X_test, df_y_train, df_y_test = train_test_split(df_X, df_y, test_size=0.2, random_state=4)

lgb_train = lgb.Dataset(df_X_train, df_y_train)
lgb_eval = lgb.Dataset(df_X_test, df_y_test)

params = {
    'seed':4,
    'metric':'rmse'}

lgbm = lgb.train(params,
                lgb_train,
                valid_sets=lgb_eval,
                num_boost_round=200,
                early_stopping_rounds=20,
                verbose_eval=50)

import xgboost as xgb

dtrain = xgb.DMatrix(df_X_train, label=df_y_train)
dtest = xgb.DMatrix(df_X_test, label=df_y_test)

params = {
    'objective': 'reg:squarederror',
    'eval_metric': 'rmse',
}

evals = [(dtrain, 'train'), (dtest, 'eval')]
evals_result = {}

xgbm = xgb.train(params,
                dtrain,
                num_boost_round=200,
                early_stopping_rounds=20,
                evals=evals,
                evals_result=evals_result
                )
from dtreeviz.trees import *

viz = dtreeviz(lgbm,
               x_data=df_X_train,
               y_data=df_y_train['y'],
               target_name='y',
               feature_names=df_X_train.columns.tolist(),
              tree_index=0)

viz.save('./lgb_tree.svg')   

from dtreeviz.trees import *

viz = dtreeviz(xgbm,
               x_data=df_X_train,      # x_dataはDataFrame
               y_data=df_y_train['y'], # y_dataはSeries
               target_name='y',
               feature_names=df_X_train.columns.tolist(), # StringのList
              tree_index=0 ) # アンサンブルモデルを使用するときはindexを指定しないとエラーになる


viz.save('./xgb_tree.svg')   



