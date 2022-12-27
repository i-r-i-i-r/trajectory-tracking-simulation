import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from nfoursid.nfoursid import NFourSID
from nfoursid.state_space import StateSpace

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
    
    scale_ = 50
    noise_gain = 1
    N_thin = 50
    
    
    X_noised = X + np.random.randn(X.shape[0], X.shape[1]) * config["measurement noise"] * noise_gain
    
    # estimate model parameter (dummy)
    #controller, title = set_controller(1)
    
    ###############################
    ###############################
    ###############################
    u_data = U[::N_thin, IDX.DELTA_DES]
    y_data = X_noised[::N_thin, IDX.XY]/scale_
    u_col = ["delta_des"]
    y_col = ["x", "y"]
    df_data = pd.DataFrame(np.hstack([u_data.reshape(-1,1), y_data]), columns=u_col+y_col)
    
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
    
    ORDER_OF_MODEL_TO_FIT = 4
    ss_identified, covariance_matrix = nfoursid.system_identification(
        rank=ORDER_OF_MODEL_TO_FIT
    )
    #kalman = Kalman(state_space_identified, covariance_matrix)
    
    
    # フィッティング
    model = StateSpace(ss_identified.a, ss_identified.b, ss_identified.c, ss_identified.d)
    
    for u in list(u_data):  # make a test-set
        y = model.step(np.array([[u]]), np.zeros([2,1]))  # generate test-set
        #_ = kalman.step(y, input_state)  # the Kalman filter sees the output and input, but not the actual internal state
    
    fig, ax = plt.subplots(figsize=(12,8))
    model.plot_input_output(fig)
    fig.tight_layout()
    plt.show()
    
    df_Y = model.to_dataframe()
    
    os.chdir(output_path)
    
    # グラフの出力
    fig = plt.figure(facecolor="w",figsize=(6,4))
    ax = fig.add_subplot(1, 1, 1, aspect="equal")
    plt.plot(X_noised[::N_thin,IDX.X], X_noised[::N_thin,IDX.Y], "None", color="blue", linewidth=1.2)
    plt.plot(df_Y["$y_0$"]*scale_,     df_Y["$y_1$"]*scale_,     "None", color="red",  linewidth=1.2)
    
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.xlim(-5, 118)
    plt.ylim(-5,  65)
    plt.title("identification (noisegain = "+str(noise_gain)+")")
    plt.grid()
    plt.savefig("identification_noisegain_"+str(noise_gain).zfill(3)+".png", dpi=300)
    plt.clf()
    plt.close()

main()


