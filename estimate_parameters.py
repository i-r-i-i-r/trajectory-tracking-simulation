import os
import numpy as np
import matplotlib.pyplot as plt

from constant.set_constant import set_index, set_config, set_ref, set_controller
from sim.kinematics_model import kinematics_model
from sim.simulate_rk4 import simulate_rk4_knownU
from sim.output_figure import output_video, plot_final_state, plot_lat_error
from optim.nonlinopt import nonlinopt # 非線形最適化
from selfmadeio.io_json import read_json
IDX = set_index()

# 目的関数
def objfun(param,x,y): 
    controller, U, x0, ref, t_vec, dt, config = x
    config["velocity (ref) [km/h]"], config["dynamics model"]["tau"], config["dynamics model"]["wheelbase"] = param
    U[:,IDX.V_DES] = config["velocity (ref) [km/h]"] / 3600. * 1000.
    X_pred, _, _ = simulate_rk4_knownU(kinematics_model, controller, U, x0, ref, t_vec, dt, config)
    residual2 = (y[:,0]-X_pred[:,0])**2 + (y[:,1]-X_pred[:,1])**2
    rmse = np.sqrt((residual2).sum()/len(residual2))
    #print(rmse)
    return rmse

# 制約式
def consfun(param): 
    cons = []
    return cons

def main(): # メイン関数
    ###############################
    ###############################
    ###############################
    # 
    main_path = os.getcwd() # 現在の作業中フォルダのパスの取得
    output_path = main_path + '\output_opt' # 出力フォルダ名の設定
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
    
    os.chdir(data_path) # dataフォルダへの移動
    U = np.load("mpc_U.npy")
    X = np.load("mpc_X.npy")
    """
    title_l=["mpc"]
    name_l=["X", "U", "debug", "lat_error_vec"]
    for title in title_l:
        for name in name_l:
            var_name = title+"_"+name
            print(name)
            #exec(var_name + " = np.load(\"" + var_name + ".npy\")" )
            exec(name + " = np.load(\"" + var_name + ".npy\")" )
    """
    os.chdir(main_path)
    
    for noise_gain in [1,5,10]:
        X_noised = X + np.random.randn(X.shape[0], X.shape[1]) * config["measurement noise"]*noise_gain
        
        # estimate model parameter (dummy)
        controller, title = set_controller(1)
        
        ###############################
        ###############################
        ###############################
        # 設定ファイルの読み込み
        config_opt = read_json("config_optim.json") 

        # パラメータの最適化
        param_init = [5, 0.5, 5]
        x_data = (controller, U, x0, ref, t_vec, dt, config)
        y_data = X_noised[:,IDX.XY]
        param_opt, fval, constype = nonlinopt(config_opt, param_init, (x_data, y_data), objfun, consfun)
        print(param_opt)
        # フィッティング
        config["velocity (ref) [km/h]"], config["dynamics model"]["tau"], config["dynamics model"]["wheelbase"] = param_opt
        U_pred = U.copy()
        U_pred[:,IDX.V_DES] = config["velocity (ref) [km/h]"] / 3600. * 1000.
        X_pred, _, _ = simulate_rk4_knownU(kinematics_model, controller, U_pred, x0, ref, t_vec, dt, config)
        
        os.chdir(output_path)
        
        # グラフの出力
        fig = plt.figure(facecolor="w",figsize=(6,4))
        ax = fig.add_subplot(1, 1, 1, aspect="equal")
        plt.plot(X_noised[:,IDX.X], X_noised[:,IDX.Y], "None", color="blue", linewidth=1.2)
        plt.plot(X_pred[:,IDX.X],   X_pred[:,IDX.Y],   "None", color="red",  linewidth=1.2)
        
        plt.xlabel("x [m]")
        plt.ylabel("y [m]")
        plt.xlim(-5, 118)
        plt.ylim(-5,  65)
        plt.title("estimation (noisegain = "+str(noise_gain))
        plt.grid()
        plt.savefig("estimation_noisegain_"+str(noise_gain).zfill(3)+".png", dpi=300)
        plt.clf()
        plt.close()
        
        # グラフの出力
        fig = plt.figure(facecolor="w",figsize=(6,4))
        ax = fig.add_subplot(1, 1, 1, aspect="equal")
        plt.plot(X_noised[:,IDX.X], X_noised[:,IDX.Y], "None", color="blue", linewidth=1.2)
        
        plt.xlabel("x [m]")
        plt.ylabel("y [m]")
        plt.xlim(69, 73)
        plt.ylim(21, 25)
        plt.title("estimation (noisegain = "+str(noise_gain)+")")
        plt.grid()
        plt.savefig("noisegain_"+str(noise_gain).zfill(3)+".png", dpi=300)
        plt.clf()
        plt.close()
        os.chdir(main_path)
    

main()


