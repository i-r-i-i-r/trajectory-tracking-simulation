import numpy as np
from constant.set_constant import set_index
IDX = set_index()

def pid_controller(state, t, ref, config, arg):
    # state = [x, y, yaw, delta]
    # u     = [v_des, delta_des]
    # ref   = [x_ref, y_ref, yaw_ref, v_ref]
    
    # 制御パラメータ
    kp = 0.3 * 1
    kd = 0.5 * 3
    
    # リファレンス位置と現在位置との差が最小となる点のref成分を抽出
    norm_ = (ref[:,IDX.XY] - state[IDX.XY])**2
    distance = norm_[:,0] + norm_[:,1]
    profile = ref[np.argmin(distance), :]
    v_des = profile[IDX.VEL]
    
    # feedforward input calculation
    ff_curvature = np.arctan(config["dynamics model"]["input delay [s]"] * profile[IDX.CURVATURE])
    
    # 車体上からみた座標に変換
    yaw = state[IDX.YAW]
    T = np.array([[np.cos(yaw), np.sin(yaw)],
                 [-np.sin(yaw), np.cos(yaw)]])
    error_xy = (state[IDX.XY] - profile[IDX.XY]).T
    error_lonlat = np.dot(T,error_xy)
    error_yaw = yaw - profile[IDX.YAW]
    
    # 割り算の商とあまりで計算する方法に変更
    while error_yaw < -2*np.pi or error_yaw > 2*np.pi:
        if error_yaw >= 2*np.pi:
            error_yaw = error_yaw - 2*np.pi
        elif error_yaw <= -2*np.pi:
            error_yaw = error_yaw + 2*np.pi
    
    if error_yaw > np.pi:
        error_yaw = error_yaw - 2*np.pi
    elif error_yaw < -np.pi:
        error_yaw = error_yaw + 2*np.pi
    
    delta_des = -kp * (error_lonlat[IDX.LAT]) - kd * error_yaw + ff_curvature
    fb_lat    = -kp * (error_lonlat[IDX.LAT])
    fb_yaw    = -kd *  error_yaw
    
    u = np.array([v_des, delta_des])
    debug_info = [profile[IDX.X], profile[IDX.Y], profile[IDX.YAW], fb_lat, fb_yaw, ff_curvature, error_lonlat[IDX.LAT]]
    ret = 0
    return (u, debug_info, ret)

