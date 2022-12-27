import numpy as np
import collections as cl
import json
from selfmadeio.io_json import read_json

# L2ノルムの計算
def _l2_norm(x):
    return np.linalg.norm(x, ord=2)


def set_config():
    config = read_json("./input/config.json") # 設定の読み込み
    
    # 単位変換計算
    RAD2DEG = 180. / np.pi
    DEG2RAD = np.pi / 180.
    KMH2MS = 1000. / 3600.
    noise = config["dynamics model"]["measurement noise ([m], [m], [deg], [deg])"]
    config["measurement noise"]      = np.array([noise[0], noise[1], noise[2]*DEG2RAD, noise[3]*DEG2RAD])
    config["mpc"]["sencor delay"]    = config["dynamics model"]["input delay [s]"]
    config["mpc"]["delay comp step"] = int(np.round(config["dynamics model"]["input delay [s]"] / config["dynamics model"]["control dt [s]"]))
    config["mpc"]["Q"]               = eval(config["mpc"]["Q (function)"])
    config["velocity (ref) [m/s]"]   = config["velocity (ref) [km/h]"] * KMH2MS
    config["dynamics model"]["velocity (max) [m/s]"] = config["dynamics model"]["velocity (max) [km/h]"] * KMH2MS
    config["dynamics model"]["velocity (min) [m/s]"] = config["dynamics model"]["velocity (min) [km/h]"] * KMH2MS
    config["dynamics model"]["steering limit [rad]"] = config["dynamics model"]["steering limit [deg]"] * DEG2RAD
    config["dynamics model"]["steering steady state error [rad]"] = config["dynamics model"]["steering steady state error [deg]"] * DEG2RAD
    config["dynamics model"]["steering steady state error [rad]"] = config["dynamics model"]["steering steady state error [deg]"] * DEG2RAD
    config["mpc"]["constraint steer rate [rad]"] = config["mpc"]["constraint steer rate [deg]"] * DEG2RAD
    config["mpc"]["constraint steering [rad]"]   = config["mpc"]["constraint steering [deg]"]   * DEG2RAD
    
    return config


def set_index():
    # インデックス
    class Index:
        pass
    
    IDX = Index()
    
    # ref
    IDX.X = 0
    IDX.Y = 1
    IDX.XY = slice(0,2,1)
    IDX.XYYAW = slice(0,3,1)
    IDX.YAW = 2
    IDX.VEL = 3
    IDX.CURVATURE = 4
    IDX.TIME = 5
    
    # state
    IDX.DELTA = 3
    
    # input
    IDX.V_DES     = 0
    IDX.DELTA_DES = 1
    
    # lonlat
    IDX.LON = 0
    IDX.LAT = 1
    return IDX


# リファレンス値の設定
def set_ref(path, vel_ref):
    IDX = set_index()
    ref = np.zeros([len(path), 6])
    ref[:,IDX.XYYAW] = path[:,IDX.XYYAW] # x, y, yaw
    ref[:,IDX.VEL]   = np.ones([len(path)]) * vel_ref # velocity
    
    # curvature
    for i in range(1,len(ref)-1):
        p1_ = ref[i-1, IDX.XY]
        p2_ = ref[i,   IDX.XY]
        p3_ = ref[i+1, IDX.XY]
        A_ = ((p2_[IDX.X]-p1_[IDX.X])*(p3_[IDX.Y]-p1_[IDX.Y]) - (p2_[IDX.Y]-p1_[IDX.Y])*(p3_[IDX.X]-p1_[IDX.X])) / 2.
        ref[i, IDX.CURVATURE] = 4. * A_ / (_l2_norm(p1_-p2_) * _l2_norm(p2_-p3_) * _l2_norm(p3_-p1_))
    
    # time
    for i in range(1,len(ref)):
        v_ = ref[i,IDX.VEL]
        d_ = _l2_norm(ref[i,IDX.XY]-ref[i-1,IDX.XY])
        dt_ = d_ / v_
        ref[i, IDX.TIME] = ref[i-1, IDX.TIME] + dt_
    
    return ref


def set_controller(control_mode):
    if control_mode == 1:
        from sim.controller.pure_pursuit import pure_pursuit
        controller = pure_pursuit
        name = "pure_pursuit"
    elif control_mode == 2:
        from sim.controller.pid_controller import pid_controller
        controller = pid_controller
        name = "pid"
    elif control_mode == 3:
        from sim.controller.model_predictive_controller import model_predictive_controller
        controller = model_predictive_controller
        name = "mpc"
    elif control_mode == 4:
        from sim.controller.model_predictive_controller2 import model_predictive_controller2
        #from sim.controller.model_predictive_controller import model_predictive_controller
        controller = model_predictive_controller2
        name = "mpc_no_constraints)"
    return (controller, name)