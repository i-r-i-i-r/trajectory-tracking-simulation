# 大体いつも使う保存形式が同じなので関数化
# ファイルを開いていて保存できないエラーを回避する
# indexはデフォルトで出力なしとする。出力したい場合は引数に index=True を入れる
import pandas as pd

_MSG_SAVED = " を保存しました"
_MSG_NOT_SAVED = " は開かれています"

def save_csv(df:pd.DataFrame, save_name:str, index:bool = False) -> None:
    try:
        save_name_0 = save_name + ".csv"
        df.to_csv(save_name_0, index = index)
        print(save_name_0 + _MSG_SAVED)
        
    except PermissionError:
        print(save_name_0 + _MSG_NOT_SAVED)
        for i in range(100):
            save_name_i = save_name + str(i).zfill(3) + ".csv"
            try:
                df.to_csv(save_name_i, index = index)
                print(save_name_i + _MSG_SAVED)
                break
            except:
                print(save_name_i + _MSG_NOT_SAVED)
