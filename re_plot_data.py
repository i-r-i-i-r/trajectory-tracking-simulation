import os
import numpy as np

from constant.set_constant import set_index, set_config, set_ref, set_controller
from sim.kinematics_model import kinematics_model
from sim.simulate_rk4 import simulate_rk4
from sim.output_figure import output_video, plot_final_state, plot_lat_error


# 出力フォルダの生成
main_path = os.getcwd() # 現在の作業中フォルダのパスの取得
output_path = main_path + '\output' # 出力フォルダ名の設定
os.makedirs(output_path, exist_ok=True) # ない場合は作る
data_path = main_path + '\output\data' # 出力フォルダ名の設定
os.makedirs(data_path, exist_ok=True) # ない場合は作る


config = set_config()

# 経路の読み込み
try:
    path = np.load("./output/data/path.npy") # x, y, yaw
except FileNotFoundError:
    from constant.set_path import set_path
    set_path()
    path = np.load("./output/data/path.npy") # x, y, yaw

# リファレンス値の設定
ref = set_ref(path, config["velocity (ref) [m/s]"])

# 初期状態の設定 (x, y, yaw, delta)
x0 = np.array(config["simulation"]["initial position (x, y, yaw, delta)"])
ts = 0.
dt = config["simulation"]["time step [s]"]
tf = config["simulation"]["total time [s]"]
t_vec = np.arange(ts,tf,dt)



title_l=["pure_pursuit", "pid", "mpc"]
name_l=["X", "U", "debug", "lat_error_vec"]
os.chdir(data_path) # dataフォルダへの移動
for title in title_l:
    for name in name_l:
        var_name = title+"_"+name
        exec(var_name + " = np.load(\"" + var_name + ".npy\")" )

lat_error_vec_l = [eval(title+"_lat_error_vec") for title in title_l]

os.chdir(output_path) # 出力フォルダへの移動
plot_lat_error(t_vec, lat_error_vec_l)
os.chdir(main_path) # mainフォルダへの移動


