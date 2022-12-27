"""
config_opt: 条件設定
arguments:  固定パラメータ(定数項やデータなど)
objfun:     目的関数
consfun:    制約式

"""
from scipy.optimize import minimize

def nonlinopt(config_opt, param_init, arguments, objfun, consfun): # 非線形最適化計算

    # 最適化計算の条件設定
    optmethod = config_opt["method"]           # 最適化計算のアルゴリズム
    tolerance = config_opt["tolerance"]        # 終了判定の許容誤差
    maxiter   = config_opt["max iteration"]    # 試行回数の上限
    disp      = config_opt["disp"]             # 表示
    if config_opt["boundary"]["apply"]:        # 境界
        bnds = config_opt["boundary"]["value"] # paramの各成分の下限と上限を並べる
    if config_opt["constraints"]["apply"]:     # 制約式
        cons = ({"type":"ineq",                # 等式(eq)か不等式(ineq)か
                 "fun" : consfun})             # 等式の場合はconsfun=0，不等式の場合はconsfun>0
    else: 
        cons = ({"type":"None"})    
        
    # 最適化の実行
    # 境界条件と制約式の有無によって場合分け
    if config_opt["boundary"]["apply"]:
        if config_opt["constraints"]["apply"]:
            result = minimize(fun = objfun, x0 = param_init, args = arguments,
                              method=optmethod, tol = tolerance, 
                              options={"disp":disp,"maxiter":maxiter},
                              bounds = bnds, constraints = cons )
        else:
            result = minimize(fun = objfun, x0 = param_init, args = arguments,
                              method=optmethod, tol = tolerance, 
                              options={"disp":disp,"maxiter":maxiter},
                              bounds = bnds )
    else:
        if config_opt["constraints"]["apply"]:
            result = minimize(fun = objfun, x0 = param_init, args = arguments,
                              method=optmethod, tol = tolerance, 
                              options={"disp":disp,"maxiter":maxiter},
                              constraints = cons )
        else:
            result = minimize(fun = objfun, x0 = param_init, args = arguments,
                              method=optmethod, tol = tolerance, 
                              options={"disp":disp,"maxiter":maxiter})
    
    param_opt = list(result["x"])
    fval = result["fun"]
    return param_opt, fval, cons["type"]