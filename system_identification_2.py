import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from nfoursid.nfoursid import NFourSID
from nfoursid.state_space import StateSpace
from nfoursid.kalman import Kalman

from constant.set_constant import set_index, set_config, set_ref, set_controller
from sim.kinematics_model import kinematics_model
from sim.simulate_rk4 import simulate_rk4_knownU
from sim.output_figure import output_video, plot_final_state, plot_lat_error
from optim.nonlinopt import nonlinopt # 非線形最適化
from selfmadeio.io_json import read_json
IDX = set_index()


def main(): # メイン関数
    ###############################
    ###############################
    ###############################
    # 
    main_path = os.getcwd() # 現在の作業中フォルダのパスの取得
    output_path = main_path + '\output_id' # 出力フォルダ名の設定
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
    os.chdir(main_path)
    
    
    noise_gain = 1
    
    
    X_noised = X + np.random.randn(X.shape[0], X.shape[1]) * config["measurement noise"] * noise_gain
    N_thin = 50
    # estimate model parameter (dummy)
    #controller, title = set_controller(1)
    
    ###############################
    ###############################
    ###############################
    u_data = U[::N_thin, IDX.DELTA_DES]
    y_data = X_noised[::N_thin, IDX.YAW]
    u_col = ["delta_des"]
    y_col = ["theta"]
    df_data = pd.DataFrame(np.vstack([u_data, y_data]).T, columns=u_col+y_col)

    nfoursid = NFourSID(
        df_data,
        output_columns=y_col,
        input_columns =u_col,
        num_block_rows=10
        )

    nfoursid.subspace_identification()
    #"""
    fig, ax = plt.subplots(figsize=(12,8))
    nfoursid.plot_eigenvalues(ax)
    fig.tight_layout()
    plt.show()
    #"""
    
    ORDER_OF_MODEL_TO_FIT = 2
    ss_identified, covariance_matrix = nfoursid.system_identification(
        rank=ORDER_OF_MODEL_TO_FIT
    )
    #kalman = Kalman(ss_identified, covariance_matrix)
    
    
    # フィッティング
    model = StateSpace(ss_identified.a, ss_identified.b, ss_identified.c, ss_identified.d)
    
    dX_pred = [[0, 0]]
    for u in list(u_data):  # make a test-set
        theta = model.step(np.array([[u]]), np.zeros([1,1]))  # generate test-set
        vel = 30. / 3.6
        dX_pred += [[vel*np.cos(theta[0][0])*dt*N_thin,vel*np.sin(theta[0][0])*dt*N_thin]]
        #_ = kalman.step(theta, np.array([[u]]))  # the Kalman filter sees the output and input, but not the actual internal state
    
    X_pred = np.array(dX_pred).cumsum(axis=0)
    
    """
    fig, ax = plt.subplots(figsize=(12,8))
    model.plot_input_output(fig)
    fig.tight_layout()
    plt.show()
    """
    
    df_Y = model.to_dataframe()
    
    os.chdir(output_path)
    
    # グラフの出力
    fig = plt.figure(facecolor="w",figsize=(6,4))
    ax = fig.add_subplot(1, 1, 1, aspect="equal")
    plt.plot(t_vec[::N_thin], X_noised[::N_thin,IDX.YAW], "None", color="blue", linewidth=1.2)
    plt.plot(t_vec[::N_thin], df_Y["$y_0$"],              "None", color="red",  linewidth=1.2)
    
    plt.xlabel("time [s]")
    plt.ylabel("theta [rad]")
    plt.xlim(t_vec[0], t_vec[-1])
    plt.title("identification 2 (noisegain = "+str(noise_gain)+")")
    plt.grid()
    plt.savefig("identification2_theta_noisegain_"+str(noise_gain).zfill(3)+".png", dpi=300)
    plt.clf()
    plt.close()
    
    
    # グラフの出力
    fig = plt.figure(facecolor="w",figsize=(6,4))
    ax = fig.add_subplot(1, 1, 1, aspect="equal")
    plt.plot(X_noised[::N_thin,IDX.X], X_noised[::N_thin,IDX.Y], "None", color="blue", linewidth=1.2)
    plt.plot(X_pred[:,0],              X_pred[:,1],              "None", color="red",  linewidth=1.2)
    
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.xlim(-5, 118)
    plt.ylim(-5,  65)
    plt.title("identification 2 (noisegain = "+str(noise_gain)+")")
    plt.grid()
    plt.savefig("identification2_xy_noisegain_"+str(noise_gain).zfill(3)+".png", dpi=300)
    plt.clf()
    plt.close()
    
    output_data([ss_identified.a, ss_identified.b, ss_identified.c, ss_identified.d, np.array(dt*N_thin)], ["A", "B", "C", "D", "dt"], "ss")
    
    

def output_data(arr_l, name_l, title):
    for i in range(len(arr_l)):
        np.save(title+"_"+name_l[i], arr_l[i])


main()



