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
def _get_error_dynamics_state_matrix(dt, v, ssA, ssB, ssC):
    n_dim = ssA.shape[0]
    
    # difinition for continuous model
    A = np.vstack([
             np.hstack([np.array([[0.,   v ]]), np.zeros([1,n_dim]) ]),
             np.hstack([np.array([[0.,  -1.]]), ssC                 ]),
             np.hstack([np.zeros( [n_dim, 2] ), ssA - np.eye(n_dim) ])
         ])
    
    B = np.vstack([
            np.array([[0.]]),
            np.array([[0.]]),
            ssB
        ])
    
    C = np.array([
         [1., 0.]+[0.]*n_dim,
         [0., 1.]+[0.]*n_dim
         ])
    
    w = np.zeros([n_dim+2, 1])
    
    # discretization
    # Ad = eye(3) + A * dt
    I = np.eye(2+n_dim)
    Ad = np.linalg.inv(I - dt * 0.5 * A) @ (I + dt * 0.5 * A)
    Bd = B * dt
    Cd = C.copy()
    wd = w * dt
    
    return (Ad, Bd, wd, Cd)

# xy-yaw dynamics model
# used for debug (predicted trajectory calculation)
# u.shape=1
def _calc_kinematics_model(x, ssX, u, dt, v, ssA, ssB, ssC):
    # x = [x, y, yaw, delta]
    n_dim = ssA.shape[0]
    I = np.eye(n_dim)
    Ad = np.linalg.inv(I - dt * 0.5 * ssA) @ (I + dt * 0.5 * ssA)
    Bd = ssB * dt
    Cd = ssC.copy()
    
    x_next = np.zeros(4)
    yaw    = x[IDX.YAW]
    
    x_next[IDX.X]   = x[IDX.X] + v * np.cos(yaw) * dt # x
    x_next[IDX.Y]   = x[IDX.Y] + v * np.sin(yaw) * dt # y
    x_next[IDX.YAW] = Cd @ ssX # yaw
    ssX = Ad @ ssX + Bd * u#[IDX.DELTA_DES]
    #x_next[IDX.YAW] = model.step(np.array([[u[IDX.DELTA_DES]]]), np.zeros([1,1])) # yaw
    
    return (x_next, ssX)

# state = np.array([0,1,0,0])
def model_predictive_controller3(state, t, ref, config, args_):
    """
    state = [x, y, yaw, delta]
    u     = [v_des, delta_des]
    ref   = [x_ref, y_ref, yaw_ref, v_ref, k_ref, t_ref]
    """
    delta_des_buffer, ssX, ssA, ssB, ssC, n_dim = args_
    
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
    error_lat = np.dot(T, error_xy)[1]
    
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
    x0 = np.vstack([
                   np.array([[error_lat]]), 
                   np.array([[error_yaw]]), 
                   ssX
                   ])
    
    ## update error dynamics for holizon
    
    # -- set mpc parameters --
    mpc_dt = config["mpc"]["dt"]
    mpc_n  = config["mpc"]["n"]
    Q      = config["mpc"]["Q"]
    R      = config["mpc"]["R"]
    mpc_t = profile[IDX.TIME]
    DIM_X = n_dim + 2
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
        
        # get discrete state matrix
        # NOTE : use control_dt as delta time, not mpc_dt. 
        Ad, Bd, wd, _ = _get_error_dynamics_state_matrix(config["dynamics model"]["control dt [s]"], v_, ssA, ssB, ssC)
        u_now = delta_des_buffer[-1-i]
        
        x_next = Ad @ x_curr + Bd * u_now + wd
        
        mpc_t = mpc_t + config["dynamics model"]["control dt [s]"] # THIS IS NOT mpc_dt, BUT control_dt
        x_curr = x_next.copy()
        
        mpc_ref_v[i] = v_
    
    x0 = x_curr.copy()
    # =========================================================================
    
    # -- mpc matrix for i = 1 --
    ref_i_ = _interp(ref[:, IDX.TIME], ref[:,:IDX.TIME], mpc_t)
    debug_ref_mat[delay_step,:] = ref_i_ # MODIFIED FOR DELAY
    v_ = ref_i_[IDX.VEL]
    Ad, Bd, wd, Cd = _get_error_dynamics_state_matrix(mpc_dt, v_, ssA, ssB, ssC)
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
        
        # get discrete state matrix
        Ad, Bd, wd, Cd = _get_error_dynamics_state_matrix(mpc_dt, v_, ssA, ssB, ssC)
        
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
        T_ = np.kron(tmp, np.array([0,0,1,1])) / mpc_dt
        
        """
        print("T_")
        print(T_.shape)
        #print(T_)
        print("Bex")
        print(Bex.shape)
        #print(Bex)
        print("Wex")
        print(Wex.shape)
        #print(Wex)
        print("x0")
        print(x0.shape)
        #print(x0)
        print("Aex")
        print(Aex.shape)
        #print(x0)
        """
        
        dsteer_vec_tmp_   = T_ @ (Aex @ x0 + Wex)
        steering_rate_lim = config["mpc"]["constraint steer rate [rad]"]
        bnd_ =  config["mpc"]["constraint steering [rad]"] * np.ones([2 * mpc_n * DIM_U, 1]) # config["mpc"]["constraint steer rate [rad]"] * np.ones([mpc_n * DIM_U, 1])
        b_  = np.vstack([steering_rate_lim * np.ones([mpc_n-1, 1]) - dsteer_vec_tmp_,
                         steering_rate_lim * np.ones([mpc_n-1, 1]) + dsteer_vec_tmp_,
                         bnd_])
        
        A_ = np.vstack([ T_ @ Bex, 
                        -T_ @ Bex,
                         np.eye(mpc_n * DIM_U),
                        -np.eye(mpc_n * DIM_U)])
        
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
    predictd_states = np.zeros([len(input_vec), n_dim+2])
    for i in range(len(input_vec)):
        x_next, ssX = _calc_kinematics_model(x_, ssX, input_vec[i][0], mpc_dt, mpc_ref_v[i], ssA, ssB, ssC)
        predictd_states[i,:] = np.hstack([ x_next[IDX.Y], x_next[IDX.YAW], ssX[:,0] ])
        x_ = x_next.copy()
    
    predictd_states_vector = predictd_states.reshape(1,-1)[0]
    
    debug_ref_mat_no_delay_comp = debug_ref_mat[delay_step:, :]
    predicted_error = Aex @ x0 + Bex @ input_vec + Wex
    predicted_error = predicted_error.reshape(n_dim+2,-1).T
    
    
    predicted_state_ideal = debug_ref_mat_no_delay_comp[:,IDX.XY] + \
        np.vstack([-np.sin(debug_ref_mat_no_delay_comp[:,IDX.YAW])*predicted_error[:,0], np.cos(debug_ref_mat_no_delay_comp[:,IDX.YAW])*predicted_error[:,0]]).T
    predicted_state_ideal = predicted_state_ideal.reshape(1,-1)[0]
    
    debug_info = np.hstack([(input_vec.T)[0], predictd_states_vector.T, predicted_state_ideal.T, error_lat])
    
    ret = (delta_des_buffer, ssX, ssA, ssB, ssC, n_dim)
    
    return (u, debug_info, ret)

