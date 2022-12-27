import numpy as np
from constant.set_constant import set_index
IDX = set_index()

# L2ノルムの計算
def l2_norm(x):
    return np.linalg.norm(x, ord=2)


def pure_pursuit(state, t, ref, config, arg):
    # state = [x, y, yaw, delta]
    # u = [v_des, delta_des]
    # ref = [x_ref; y_ref; yaw_ref; v_ref]
    
    lookahead_dist = config["pure pursuit"]["lookahead [m]"];
    
    # リファレンス位置と現在位置との差が最小となる点のref成分を抽出
    norm_ = (ref[:,IDX.XY] - state[IDX.XY])**2
    distance = norm_[:,0] + norm_[:,1]
    min_index = np.argmin(distance)
    profile = ref[min_index, :]
    v_des = profile[IDX.VEL]
    
    for i in range(min_index,len(ref)):
        distance = l2_norm(ref[i,IDX.XY] - state[IDX.XY])
        if distance > lookahead_dist:
            break
    
    lookahead_pt = ref[i, :]
    
    alpha = np.arctan2(lookahead_pt[IDX.Y] - state[IDX.Y], lookahead_pt[IDX.X] - state[IDX.X]) - state[IDX.YAW]
    
    omega_des = 2. * v_des * np.sin(alpha) / config["dynamics model"]["wheelbase [m]"]
    delta_des = np.arctan2(omega_des * config["dynamics model"]["wheelbase [m]"], v_des)
    
    u = np.array([v_des, delta_des])
    
    # lattitude error calc for debug
    sp_yaw = profile[IDX.YAW]
    T = np.array([
           [ np.cos(sp_yaw), np.sin(sp_yaw)],
           [-np.sin(sp_yaw), np.cos(sp_yaw)]])
    error_xy = (state[IDX.XY] - profile[IDX.XY]).T
    error_lat = np.dot(T,error_xy)[IDX.LAT]
    
    debug_info = [lookahead_pt[IDX.X], lookahead_pt[IDX.Y], lookahead_pt[IDX.YAW], error_lat]
    ret = 0
    
    return (u, debug_info, ret)