import numpy as np



from scipy.interpolate import interp1d
from nfoursid.nfoursid import NFourSID
from nfoursid.state_space import StateSpace
import pandas as pd


def ____re_sampling_ssmodel(ssA, ssB, ssC, dt_old, t_sim, dt_new):
    model_old = StateSpace(ssA, ssB, ssC, np.array([[0]]))
    t_old = np.arange(0,t_sim,dt_old)
    u_old = np.hstack([ np.zeros([500]), np.ones([len(t_old)-500]) ])
    y_old = []
    for u in list(u_old):  # make a test-set
        y_old += [model_old.step(np.array([[u]]), np.zeros([1,1]))[0,0]]
    
    y_old = np.array(y_old)
    
    fy = interp1d(t_old, y_old, kind="linear")
    fu = interp1d(t_old, u_old, kind="linear")
    t_new = np.arange(0,2000-dt_old, dt_new)
    y_new = fy(t_new)
    u_new = fu(t_new)
    """
    import matplotlib.pyplot as plt
    plt.plot(t_old, y_old, "o")
    plt.plot(t_new, y_new, "+")
    plt.show()
    """
    
    nfoursid = NFourSID(
            pd.DataFrame(np.vstack([u_new,y_new]).T, columns = ["u","y"]),
            output_columns=["y"],
            input_columns =["u"],
            num_block_rows=10
            )
    
    nfoursid.subspace_identification()
    """
    fig, ax = plt.subplots(figsize=(12,8))
    nfoursid.plot_eigenvalues(ax)
    fig.tight_layout()
    plt.show()
    """
    ORDER_OF_MODEL_TO_FIT = 2
    ss_identified, covariance_matrix = nfoursid.system_identification(
        rank=ORDER_OF_MODEL_TO_FIT
    )
    
    """
    model_new = StateSpace(ss_identified.a, ss_identified.b, ss_identified.c, np.array([[0]]))
    t_new = np.arange(0,t_sim,dt_new)
    fu = interp1d(t_old, u_old, kind="linear")
    u_new = fu(t_new)
    y_new = []
    for u in list(u_new):  # make a test-set
        y_new += [model_new.step(np.array([[u]]), np.zeros([1,1]))[0,0]]
    
    import matplotlib.pyplot as plt
    plt.plot(t_old, y_old, "o")
    plt.plot(t_new, y_new, "+")
    plt.show()
    """
    return (ss_identified.a, ss_identified.b, ss_identified.c)







def simulate_rk4_ss(model, controller, x0, ref, t_vec, dt, config, ssA, ssB, ssC, ssdt):
    
    #ssA_c, ssB_c, ssC_c = re_sampling_ssmodel(ssA, ssB, ssC, ssdt, 2000, 1)
    
    # ss model dt: ssdt -> 1
    n_dim = ssA.shape[0]
    I = np.eye(n_dim)
    ssA_c = np.linalg.inv(I - 1./ssdt * 0.5 * ssA) @ (I + 1./ssdt * 0.5 * ssA)
    ssB_c = ssB * 1./ssdt
    ssC_c = ssC.copy()
    
    args_ = [], np.zeros([n_dim,1]), ssA_c, ssB_c, ssC_c, n_dim
    tmp_u, tmp_u_debug, ret = controller(x0, t_vec[0], ref, config, args_)
    
    
    state_log = np.zeros([len(t_vec), len(x0)])
    input_log = np.zeros([len(t_vec), len(tmp_u)])
    debug_info = np.zeros([len(t_vec), len(tmp_u_debug)])
    
    input_delay = config["dynamics model"]["input delay [s]"]
    delay_count = round(input_delay / dt)
    input_buf = np.zeros([delay_count, len(tmp_u)])
    u = np.zeros_like(tmp_u) # initial input
    u_debug = np.zeros_like(tmp_u_debug)
    
    control_dt = config["dynamics model"]["control dt [s]"]
    control_count = round(control_dt / dt)
    
    count = 0
    x = x0.copy()
    ret = ret[0], np.zeros([n_dim,1]), ssA_c, ssB_c, ssC_c, n_dim
    for t in list(t_vec):
        # -- control once per control_count time --
        
        if count%control_count == 0:
            # add noise
            x_noised = x + np.random.randn(len(x0)) * config["measurement noise"]
            u, u_debug, ret = controller(x_noised, t, ref, config, ret)
        
        # -- add input delay --
        input_buf = np.vstack([u, input_buf[1:-2,:]])#############
        u_delayed = input_buf[-1,:]
        
        # -- runge-kutta --
        k1 = model(x,            u_delayed, config["dynamics model"])
        k2 = model(x + k1*dt/2., u_delayed, config["dynamics model"])
        k3 = model(x + k2*dt/2., u_delayed, config["dynamics model"])
        k4 = model(x + k3*dt,    u_delayed, config["dynamics model"])
        x = x + (k1 + 2.*k2 + 2.*k3 + k4) * dt / 6.
        
        # -- save data --
        state_log[count,:]  = x
        input_log[count,:]  = u
        debug_info[count,:] = u_debug
        
        if count%int(1/dt)==0:
            print(str(t)+" s")
        
        count += 1
        
    
    return (state_log, input_log, debug_info)



def simulate_rk4(model, controller, x0, ref, t_vec, dt, config):
    
    tmp_u, tmp_u_debug, ret = controller(x0, t_vec[0], ref, config, [])
    
    
    state_log = np.zeros([len(t_vec), len(x0)])
    input_log = np.zeros([len(t_vec), len(tmp_u)])
    debug_info = np.zeros([len(t_vec), len(tmp_u_debug)])
    
    input_delay = config["dynamics model"]["input delay [s]"]
    delay_count = round(input_delay / dt)
    input_buf = np.zeros([delay_count, len(tmp_u)])
    u = np.zeros_like(tmp_u) # initial input
    u_debug = np.zeros_like(tmp_u_debug)
    
    control_dt = config["dynamics model"]["control dt [s]"]
    control_count = round(control_dt / dt)
    
    count = 0
    x = x0.copy()
    for t in list(t_vec):
        # -- control once per control_count time --
        
        if count%control_count == 0:
            # add noise
            x_noised = x + np.random.randn(len(x0)) * config["measurement noise"]
            u, u_debug, ret = controller(x_noised, t, ref, config, ret)
        
        # -- add input delay --
        input_buf = np.vstack([u, input_buf[1:-2,:]])#############
        u_delayed = input_buf[-1,:]
        
        # -- runge-kutta --
        k1 = model(x,            u_delayed, config["dynamics model"])
        k2 = model(x + k1*dt/2., u_delayed, config["dynamics model"])
        k3 = model(x + k2*dt/2., u_delayed, config["dynamics model"])
        k4 = model(x + k3*dt,    u_delayed, config["dynamics model"])
        x = x + (k1 + 2.*k2 + 2.*k3 + k4) * dt / 6.
        
        # -- save data --
        state_log[count,:]  = x
        input_log[count,:]  = u
        debug_info[count,:] = u_debug
        
        if count%int(1/dt)==0:
            print(str(t)+" s")
        
        count += 1
        
    
    return (state_log, input_log, debug_info)



# open loop sim
# without noise
def simulate_rk4_knownU(model, controller, U, x0, ref, t_vec, dt, config):
    tmp_u, tmp_u_debug, ret = controller(x0, t_vec[0], ref, config, [])
    
    state_log = np.zeros([len(t_vec), len(x0)])
    input_log = np.zeros([len(t_vec), len(tmp_u)])
    debug_info = np.zeros([len(t_vec), len(tmp_u_debug)])
    
    input_delay = config["dynamics model"]["input delay [s]"]
    delay_count = round(input_delay / dt)
    input_buf = np.zeros([delay_count, len(tmp_u)])
    u = np.zeros_like(tmp_u) # initial input
    u_debug = np.zeros_like(tmp_u_debug)
    
    control_dt = config["dynamics model"]["control dt [s]"]
    control_count = round(control_dt / dt)
    
    count = 0
    x = x0.copy()
    for t in list(t_vec):
        # -- control once per control_count time --
        
        u = U[count,:]
        # -- add input delay --
        input_buf = np.vstack([u, input_buf[1:-2,:]])#############
        u_delayed = input_buf[-1,:]
        
        # -- runge-kutta --
        k1 = model(x,            u_delayed, config["dynamics model"])
        k2 = model(x + k1*dt/2., u_delayed, config["dynamics model"])
        k3 = model(x + k2*dt/2., u_delayed, config["dynamics model"])
        k4 = model(x + k3*dt,    u_delayed, config["dynamics model"])
        x = x + (k1 + 2.*k2 + 2.*k3 + k4) * dt / 6.
        
        # -- save data --
        state_log[count,:]  = x
        input_log[count,:]  = u
        debug_info[count,:] = u_debug
        
        count += 1
        
    
    return (state_log, input_log, debug_info)


