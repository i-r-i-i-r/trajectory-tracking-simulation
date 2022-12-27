# jsonファイルの読み込みと書き出しを簡単にする
import json
import collections as cl

def write_json(dict:dict, save_name:str) -> None: # jsonファイルの書き出し
    fw = open(save_name,'w')
    json.dump(dict,fw,indent=4)

def read_json(file_name:str) -> dict: # jsonファイルの読み込み
    decoder = json.JSONDecoder(object_pairs_hook=cl.OrderedDict)
    with open(file_name) as conf_file:
        data = decoder.decode(conf_file.read())
    return data