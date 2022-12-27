# design path from points by spline
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

PATH_SCALE = 15.
N_DIV=100

# 点で表された軌道をスプライン補完
def _spline_trajectory(points, N_DIV):
    
    s0 = np.arange(0,len(points),1) # 元データの各点に対応するインデックス
    s1 = np.arange(0,len(points)-1,1/N_DIV) # 補間後のデータの各店に対応するインデックス
    
    fx = interp1d(s0, points[:,0], kind="cubic") # xをスプライン補完するときの関数を求める
    fy = interp1d(s0, points[:,1], kind="cubic")
    #plt.plot(fx(s1),fy(s1),"o")
    return np.vstack([fx(s1),fy(s1)]).T # shape=(N_DIV,2)の配列を返す


# 経路からのyaw(方向角)の計算
def _calc_yaw(points):
    yaw = np.zeros([len(points), 1])
    for i in range(1,len(points)-1):
        x_forward  = points[i+1,0]
        x_backward = points[i-1,0]
        y_forward  = points[i+1,1]
        y_backward = points[i-1,1]
        yaw[i] = np.arctan2(y_forward-y_backward, x_forward-x_backward)
    
    yaw[0]  = yaw[1]
    yaw[-1] = yaw[-2]
    return yaw



def set_path():
    # 経路の形状を表す点群データを読み込む
    points = pd.read_csv("./input/points.csv").values * PATH_SCALE
    
    # スプライン補間
    p_spline = _spline_trajectory(points, N_DIV)
    
    # yawを計算
    yaw = _calc_yaw(p_spline)
    
    # 経路ファイルを出力
    path = np.hstack([p_spline, yaw])
    np.save('./output/data/path',path)
    
    # 経路の図示
    arrow_scale = 0.5 * PATH_SCALE
    N_mabiki = 36
    plt.plot(p_spline[:,0], p_spline[:,1], "-", color='r')
    plt.plot(points[:,0], points[:,1], 'o', color='k', markersize=5)
    plt.quiver(p_spline[:,0][::N_mabiki], p_spline[:,1][::N_mabiki], np.cos(yaw[::N_mabiki]), np.sin(yaw[::N_mabiki]), angles='xy', scale_units='xy', scale=1/arrow_scale, color='b')
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.grid()
    plt.savefig("./output/path.png",dpi=300)
    plt.clf()
    plt.close()

