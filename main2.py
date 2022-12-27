# https://qiita.com/taka_horibe/items/47f86e02e2db83b0c570
# state = [x, y, yaw, delta]
# input = [v_des, delta_des]
# ref = [x_ref, y_ref, yaw_ref, v_ref, curvature, time]
# v_des(=v): 後輪の移動速さ(目標値と完全に一致していると仮定)
# yaw      : 慣性系に対する車両角度(解説ページにおけるtheta)
# delta    : 車体に対するタイヤ角度
# delta_des: 車体に対するタイヤ角度の目標値

import os
import numpy as np
from constant.set_constant import set_index, set_config, set_ref, set_controller
from sim.kinematics_model import kinematics_model
from sim.simulate_rk4 import simulate_rk4_ss
from sim.output_figure import output_video, plot_final_state, plot_lat_error

def output_data(arr_l, name_l, title):
    for i in range(len(arr_l)):
        np.save(title+"_"+name_l[i], arr_l[i])


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



from nfoursid.state_space import StateSpace
ssA = np.load("./output_id/ss_A.npy")
ssB = np.load("./output_id/ss_B.npy")
ssC = np.load("./output_id/ss_C.npy")
#ssD = np.load("./output_id/ss_D.npy")
ssdt = np.load("./output_id/ss_dt.npy")



title_l = []
lat_error_vec_l = []
"""
for i_c in config["control mode (1:pure_pursuit, 2:pid, 3:mpc, 4:mpc_no_constraints)"]:
# 制御方式の設定
controller, title = set_controller(i_c)
"""

from sim.controller.model_predictive_controller3 import model_predictive_controller3
controller = model_predictive_controller3
title = "mpc_ss"


## simulation
X, U, debug = simulate_rk4_ss(kinematics_model, controller, x0, ref, t_vec, dt, config, ssA, ssB, ssC, ssdt)
lat_error_vec = debug[:,-1]
#title_l += [title]
#lat_error_vec_l += [lat_error_vec]

os.chdir(data_path) # dataフォルダへの移動
output_data([X, U, debug, lat_error_vec], ["X", "U", "debug", "lat_error_vec"], title)

## output
L = config["dynamics model"]["wheelbase [m]"]
os.chdir(output_path) # 出力フォルダへの移動
plot_final_state(path, X, t_vec, lat_error_vec, L, title, config)

if config["output video"]["save (0:no, 1:yes)"]:
    output_video(path, X, t_vec, lat_error_vec, L, title, config)

#plot_lat_error(t_vec, lat_error_vec_l)
os.chdir(main_path) # mainフォルダへの移動
