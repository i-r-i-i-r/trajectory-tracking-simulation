# https://myenigma.hatenablog.com/entry/2016/10/16/131545
# http://cvxopt.org/userguide/index.html
#import matplotlib.pyplot as plt
import numpy as np
import cvxopt as co
from scipy.interpolate import interp1d
from constant.set_constant import set_index
IDX = set_index()

def _interp(x, y, x0):
    f = interp1d(x, y.T)
    return f(x0)


# time varying error dynamics model
# used for error dynamics update
def _get_error_dynamics_state_matrix(dt, v, L, tau, curvature):
    # linearization around delta = 0
    # A = [0, v, 0
    #      0, 0, v/L
    #      0, 0, -1/tau]
    # B = [0;0;1/tau]
    # C = [1, 0, 0
    #      0, 1, 0]
    # w = [0 
    #     -v*curvature
    #      0]
    
    # linearization around delta = delta_ref (better accuracy than delta=0)
    delta_r = np.arctan(L*curvature)
    if abs(delta_r) >= 40./180. * np.pi:
        delta_r = (40./180. * np.pi)*(delta_r/abs(delta_r))
    
    cos_delta_r_squared_inv = 1. / ((np.cos(delta_r))**2)
    
    # difinition for continuous model
    A = np.array([
         [0., v,  0.],
         [0., 0., v/L*cos_delta_r_squared_inv],
         [0., 0., -1./tau]])
    B = np.array([[0.], 
                  [0.], 
                  [1./tau]])
    C = np.array([
         [1., 0., 0.],
         [0., 1., 0.]])
    w = np.array([
         [0.],
         [-v*curvature + v/L*(np.tan(delta_r)-delta_r*cos_delta_r_squared_inv)],
         [0.]])
    
    # discretization
    # Ad = eye(3) + A * dt
    I = np.eye(3)
    Ad = np.linalg.inv(I - dt * 0.5 * A) @ (I + dt * 0.5 * A)
    Bd = B * dt
    Cd = C
    wd = w * dt
    
    return (Ad, Bd, wd, Cd)
#Ad, Bd, wd, Cd = _get_error_dynamics_state_matrix(1,2,3,4,5)
#Ad.shape, Bd.shape, wd.shape, Cd.shape # (3, 3), (3, 1), (3, 1), (2, 3)



# xy-yaw dynamics model
# used for debug (predicted trajectory calculation)
# u.shape=1
def _calc_kinematics_model(x, u, dt, v, L, tau):
    # x = [x, y, yaw, delta]
    x_next = np.zeros(4)
    yaw    = x[IDX.YAW]
    delta  = x[IDX.DELTA]
    
    x_next[IDX.X]     = x[IDX.X]     + v * np.cos(yaw)          * dt # x
    x_next[IDX.Y]     = x[IDX.Y]     + v * np.sin(yaw)          * dt # y
    x_next[IDX.YAW]   = x[IDX.YAW]   + v * np.tan(delta) / L    * dt # yaw
    x_next[IDX.DELTA] = x[IDX.DELTA] - (x[IDX.DELTA] - u) / tau * dt # delta
    
    return x_next

# state = np.array([0,1,0,0])
def model_predictive_controller(state, t, ref, config, delta_des_buffer):
    """
    state = [x, y, yaw, delta]
    u     = [v_des, delta_des]
    ref   = [x_ref, y_ref, yaw_ref, v_ref, k_ref, t_ref]
    """
    
    # =================== for delay compensation ==============================
    if delta_des_buffer == []:
        delta_des_buffer = np.zeros(config["mpc"]["delay comp step"])
    # =========================================================================
    
    delay_step = len(delta_des_buffer)
    
    # convert xy-yaw to error dynamics
    
    # calculate nearest point (use as initial state)
    norm_ = (ref[:,IDX.XY] - state[IDX.XY])**2
    distance = norm_[:,0] + norm_[:,1]
    profile = ref[np.argmin(distance), :]
    
    # convert x,y to lon,lat model
    sp_yaw = profile[IDX.YAW]
    T = np.array([[ np.cos(sp_yaw), np.sin(sp_yaw)],
                  [-np.sin(sp_yaw), np.cos(sp_yaw)]])
    error_xy = (state[IDX.XY] - profile[IDX.XY]).T
    error_lat = np.dot(T,error_xy)[1]
    
    # calculate yaw error
    error_yaw = state[IDX.YAW] - sp_yaw
    while -2*np.pi > error_yaw or error_yaw > 2*np.pi:
        if error_yaw >= 2*np.pi:
            error_yaw = error_yaw - 2*np.pi
        elif error_yaw <= -2*np.pi:
            error_yaw = error_yaw + 2*np.pi
    
    if error_yaw > np.pi:
        error_yaw = error_yaw - 2*np.pi
    elif error_yaw < -np.pi:
        error_yaw = error_yaw + 2*np.pi
    
    # initial state for error dynamics
    x0 = np.array([[error_lat], 
                   [error_yaw], 
                   [state[IDX.DELTA]]])
    
    ## update error dynamics for holizon
    
    # -- set mpc parameters --
    mpc_dt = config["mpc"]["dt"]
    mpc_n  = config["mpc"]["n"]
    Q      = config["mpc"]["Q"]
    R      = config["mpc"]["R"]
    mpc_t = profile[IDX.TIME]
    DIM_X = 3
    DIM_Y = 2
    DIM_U = 1
    Aex = np.zeros([DIM_X*mpc_n, DIM_X      ])
    Bex = np.zeros([DIM_X*mpc_n, DIM_U*mpc_n])
    Wex = np.zeros([DIM_X*mpc_n, 1          ])
    Cex = np.zeros([DIM_Y*mpc_n, DIM_X*mpc_n])
    Qex = np.zeros([DIM_Y*mpc_n, DIM_Y*mpc_n])
    Rex = np.zeros([DIM_U*mpc_n, DIM_U*mpc_n])
    mpc_ref_v     = np.zeros([mpc_n + delay_step, 1])
    debug_ref_mat = np.zeros([mpc_n + delay_step, 5])
    
    # =================== for delay compensation ==============================
    # -- apply delay compensation : update dynamics with increasing mpt_t --
    x_curr = x0.copy()
    for i in range(delay_step-1):
        if mpc_t > ref[-1, IDX.TIME]:
            mpc_t = ref[-1, IDX.TIME]
            print('[MPC] path is too short to predict dynamics')
        
        ref_now = _interp(ref[:, IDX.TIME], ref[:,:IDX.TIME], mpc_t)
        debug_ref_mat[i,:] = ref_now
        v_ = ref_now[IDX.VEL]
        k_ = ref_now[IDX.CURVATURE]
        
        # get discrete state matrix
        # NOTE : use control_dt as delta time, not mpc_dt. 
        Ad, Bd, wd, _ = _get_error_dynamics_state_matrix(config["dynamics model"]["control dt [s]"], v_, 
                                                        config["dynamics model"]["wheelbase [m]"], 
                                                        config["dynamics model"]["tau [s]"], k_)
        u_now = delta_des_buffer[-1-i]
        x_next = Ad @ x_curr + Bd * u_now + wd
        
        mpc_t = mpc_t + config["dynamics model"]["control dt [s]"] # THIS IS NOT mpc_dt, BUT control_dt
        x_curr = x_next
        
        mpc_ref_v[i] = v_
    
    x0 = x_curr.copy()
    # =========================================================================
    
    # -- mpc matrix for i = 1 --
    ref_i_ = _interp(ref[:, IDX.TIME], ref[:,:IDX.TIME], mpc_t)
    debug_ref_mat[delay_step,:] = ref_i_ # MODIFIED FOR DELAY
    v_ = ref_i_[IDX.VEL]
    k_ = ref_i_[IDX.CURVATURE]
    Ad, Bd, wd, Cd = _get_error_dynamics_state_matrix(mpc_dt, v_, config["dynamics model"]["wheelbase [m]"], config["dynamics model"]["tau [s]"], k_) 
    Aex[:DIM_X, :     ] = Ad
    Bex[:DIM_X, :DIM_U] = Bd
    Wex[:DIM_X        ] = wd
    Cex[:DIM_Y, :DIM_X] = Cd
    Qex[:DIM_Y, :DIM_Y] = Q
    Rex[:DIM_U, :DIM_U] = R
    
    mpc_ref_v[delay_step] = v_

    # -- mpc matrix for i = 2:n --
    for i in range(1,mpc_n):
        # update mpc time
        mpc_t = mpc_t + mpc_dt
        if mpc_t > ref[-1, IDX.TIME]:
            mpc_t = ref[-1, IDX.TIME]
            print('[MPC] path is too short to predict dynamics')
        
        # get reference information
        ref_i_ = _interp(ref[:, IDX.TIME], ref[:,:IDX.TIME], mpc_t)
        debug_ref_mat[i + delay_step,:] = ref_i_
        v_ = ref_i_[IDX.VEL]
        k_ = ref_i_[IDX.CURVATURE]
        
        # get discrete state matrix
        Ad, Bd, wd, Cd = _get_error_dynamics_state_matrix(mpc_dt, v_, config["dynamics model"]["wheelbase [m]"], config["dynamics model"]["tau [s]"], k_)
        
        # update mpc matrix #################################################
        idx_x_i      = slice( i   *DIM_X, (i+1)*DIM_X, 1)
        idx_x_i_prev = slice((i-1)*DIM_X,  i   *DIM_X, 1)
        idx_u_i      = slice( i   *DIM_U, (i+1)*DIM_U, 1)
        idx_y_i      = slice( i   *DIM_Y, (i+1)*DIM_Y, 1)
        Aex[idx_x_i, :] = Ad @ Aex[idx_x_i_prev, :]
        for j in range(i):
            idx_u_j = slice(j*DIM_U, (j+1)*DIM_U, 1)
            Bex[idx_x_i, idx_u_j] = Ad @ Bex[idx_x_i_prev, idx_u_j]
        
        Bex[idx_x_i, idx_u_i] = Bd
        Wex[idx_x_i] = Ad @ Wex[idx_x_i_prev] + wd
        Cex[idx_y_i, idx_x_i] = Cd
        Qex[idx_y_i, idx_y_i] = Q
        Rex[idx_u_i, idx_u_i] = R
        
        mpc_ref_v[i + delay_step] = v_
    
    
    ## convex optimization
    
    # The problem is to solve following for U.
    #   1/2 * U'* mat1 * U + mat2 * U + C = 0
    mat1 = Bex.T @ Cex.T @ Qex @ Cex @ Bex + Rex
    mat2 = (x0.T @ Aex.T + Wex.T) @ Cex.T @ Qex @ Cex @ Bex
    
    
    
    if config["mpc"]["constraints (0:off, 1:on)"] == 0:
        input_vec = -np.linalg.inv(mat1) @ mat2.T
    else:
        # --- convex optimization ---
        #   minimize for x, s.t.
        #   J(x) = 1/2 * x' * H * x + f' * x, 
        #   A*x <= b,   lb <= x <= ub
        
        H_ = (mat1 + mat1.T) / 2.
        f_ = mat2.copy().T
        
        # add steering rate constraint
        tmp = -np.hstack([np.eye(mpc_n-1), np.zeros([mpc_n-1, 1])])
        tmp[:,1:] += np.eye(mpc_n-1)
        T_ = np.kron(tmp, np.array([0,0,1])) / mpc_dt
        A_ = np.vstack([ T_ @ Bex, 
                        -T_ @ Bex,
                         np.eye(mpc_n * DIM_U),
                        -np.eye(mpc_n * DIM_U)])
        
        dsteer_vec_tmp_   = T_ @ (Aex @ x0 + Wex)
        """
        print("====")
        print(dsteer_vec_tmp_.shape)
        print(T_.shape)
        print(x0.shape)
        """
        steering_rate_lim = config["mpc"]["constraint steer rate [rad]"]
        bnd_ =  config["mpc"]["constraint steering [rad]"] * np.ones([2 * mpc_n * DIM_U, 1]) # config["mpc"]["constraint steer rate [rad]"] * np.ones([mpc_n * DIM_U, 1])
        b_  = np.vstack([steering_rate_lim * np.ones([mpc_n-1, 1]) - dsteer_vec_tmp_,
                         steering_rate_lim * np.ones([mpc_n-1, 1]) + dsteer_vec_tmp_,
                         bnd_])
        
        co.solvers.options['show_progress'] = False
        #solvers.options['maxiters'] = 1000
        sol = co.solvers.qp(co.matrix(H_), co.matrix(f_), co.matrix(A_), co.matrix(b_))
        #sol = cvxopt.solvers.qp(matrix(H_), matrix(f_)) # no constraints ver
        input_vec = np.array(sol["x"])
        
        """# for debug: compare with / without constraint optimization
        input_vec_LS = -np.linalg.inv(mat1) @ mat2.T
        plt.plot(input_vec_LS, "o", color="r", linestyle="None")
        plt.plot(input_vec, color="b")
        plt.grid()
        plt.show()
        """
    
    delta_des = input_vec[0][0]
    v_des     = profile[IDX.VEL]
    u         = np.array([v_des, delta_des])
    
    # =================== for delay compensation ==============================
    delta_des_buffer = np.hstack([delta_des, delta_des_buffer[:-1]])
    # =========================================================================
    
    ## (debug) calculate predicted trajectory 
    
    x_ = state.copy()
    predictd_states = np.zeros([len(input_vec), len(state)])
    for i in range(len(input_vec)):
        x_next = _calc_kinematics_model(x_, input_vec[i][0], mpc_dt, mpc_ref_v[i], config["dynamics model"]["wheelbase [m]"], config["dynamics model"]["tau [s]"])
        predictd_states[i,:] = x_next.T
        x_ = x_next.copy()
    
    predictd_states_vector = predictd_states.reshape(1,-1)[0]
    
    debug_ref_mat_no_delay_comp = debug_ref_mat[delay_step:, :]
    predicted_error = Aex @ x0 + Bex @ input_vec + Wex
    predicted_error = predicted_error.reshape(3,-1).T
    predicted_state_ideal = debug_ref_mat_no_delay_comp[:,IDX.XY] + \
        np.vstack([-np.sin(debug_ref_mat_no_delay_comp[:,IDX.YAW])*predicted_error[:,0], np.cos(debug_ref_mat_no_delay_comp[:,IDX.YAW])*predicted_error[:,0]]).T
    predicted_state_ideal = predicted_state_ideal.reshape(1,-1)[0]
    
    debug_info = np.hstack([(input_vec.T)[0], predictd_states_vector.T, predicted_state_ideal.T, error_lat])
    
    return (u, debug_info, delta_des_buffer)

