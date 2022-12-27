import numpy as np
from constant.set_constant import set_index
IDX = set_index()

def kinematics_model(state, input, config_dm):
    
    # input
    v_des = input[IDX.V_DES]
    delta_des = input[IDX.DELTA_DES]
    
    ## limit
    delta_des = max(min(delta_des, config_dm["steering limit [rad]"]), -config_dm["steering limit [rad]"])
    v_des     = max(min(v_des,     config_dm["velocity (max) [m/s]"]),  config_dm["velocity (min) [m/s]"])
    v = v_des
    
    yaw   = state[IDX.YAW]
    delta = state[IDX.DELTA]
    
    d_x     = v * np.cos(yaw)
    d_y     = v * np.sin(yaw)
    d_yaw   = v * np.tan(delta) / config_dm["wheelbase [m]"]
    d_delta = - (delta - delta_des) / config_dm["tau [s]"]
    
    ## add steady state error caused by friction
    if abs(delta - delta_des) < config_dm["steering steady state error [rad]"]:
        d_delta = 0
    
    return np.array([d_x, d_y, d_yaw, d_delta])



