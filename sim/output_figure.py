import matplotlib.pyplot as plt
import cv2
import io
import numpy as np
from constant.set_constant import set_index


IDX = set_index()

def output_video(path, X, t_vec, lat_error_vec, L, title, config):
    # set format of output movie
    # 映像化処理の設定
    framerate = config["output video"]["frame rate"] #[/s]
    interval  = config["output video"]["thinning out interval (1: not thinning out)"] # 画像の間引き間隔(1: 間引きなし)
    savefmt   = config["output video"]["format (.mp4 or .gif)"] # 映像のファイル名
    save_name = "result_"+title+savefmt
    
    imgs = []
    for k in range(0,len(X),25):
        _plot_single_state(path, X, t_vec, lat_error_vec, L, title, k)
        
        # matplotlib -> cv2
        buf = io.BytesIO() # bufferを用意
        plt.savefig(buf, format='png', dpi=300) # bufferに保持
        plt.clf()
        plt.close()
        enc = np.frombuffer(buf.getvalue(), dtype=np.uint8) # bufferからの読み出し
        img = cv2.imdecode(enc, 1) # デコード
        
        imgs.append(img)
        if k == 0:
            height, width, _ = img.shape
            size = (width, height)
    
    if savefmt == '.mp4':
        movie = cv2.VideoWriter(save_name, cv2.VideoWriter_fourcc(*'MP4V'), framerate, size) # 映像の出力設定
        for i in range(len(imgs)):
            movie.write(imgs[i]) # 映像の書き出し
        movie.write(np.zeros_like(imgs[0])) # なぜか最後が切れるので追加
        movie.release() # 映像の書き出しの終了
        
    elif savefmt == '.gif':
        imgs_PIL = [Image.fromarray(cv2.cvtColor(imgs[i], cv2.COLOR_BGR2RGB)) for i in range(len(imgs))]
        imgs_PIL[0].save(save_name, save_all=True, append_images=imgs_PIL[1:], optimize=False, loop=0)
    else:
        print("save format: "+savefmt+" is wrong.")

def  plot_final_state(path, X, t_vec, lat_error_vec, L, title, config):
    _plot_single_state(path, X, t_vec, lat_error_vec, L, title, -1) # 厳密には一番最後は除かれている
    plt.savefig("final_state_"+title+".png", dpi=300)
    plt.clf()
    plt.close()


def _plot_single_state(path, X, t_vec, lat_error_vec, L, title, k):
    distance_tire = 1.5
    fig = plt.figure(facecolor="w",figsize=(6,8))
    ax = fig.add_subplot(2, 1, 1, aspect="equal")
    plt.plot(path[:,IDX.X], path[:,IDX.Y], color='r', linestyle = "dashed", linewidth=1) 
    plt.plot(X[:k,IDX.X],   X[:k,IDX.Y],   "None", color="blue", linewidth=1.2)
    
    rear_x  = X[k,IDX.X]
    rear_y  = X[k,IDX.Y]
    front_x = X[k,IDX.X] + L*np.cos(X[k,IDX.YAW])
    front_y = X[k,IDX.Y] + L*np.sin(X[k,IDX.YAW])
    
    rear_tire1_x  = rear_x  - distance_tire/2.* np.sin(X[k,IDX.YAW])
    rear_tire2_x  = rear_x  + distance_tire/2.* np.sin(X[k,IDX.YAW])
    front_tire1_x = front_x - distance_tire/2.* np.sin(X[k,IDX.YAW])
    front_tire2_x = front_x + distance_tire/2.* np.sin(X[k,IDX.YAW])
    rear_tire1_y  = rear_y  + distance_tire/2.* np.cos(X[k,IDX.YAW])
    rear_tire2_y  = rear_y  - distance_tire/2.* np.cos(X[k,IDX.YAW])
    front_tire1_y = front_y + distance_tire/2.* np.cos(X[k,IDX.YAW])
    front_tire2_y = front_y - distance_tire/2.* np.cos(X[k,IDX.YAW])
    
    plt.plot([rear_tire1_x, rear_tire2_x, front_tire2_x, front_tire1_x,rear_tire1_x], [rear_tire1_y, rear_tire2_y, front_tire2_y, front_tire1_y, rear_tire1_y], "None", color='k', linestyle="-", linewidth=1.2) # body
    plt.plot([rear_tire1_x, rear_tire2_x, front_tire2_x, front_tire1_x], [rear_tire1_y, rear_tire2_y, front_tire2_y, front_tire1_y], "o", color='k', linestyle="None", markersize=2) # rear_tire
    
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.xlim(-5, 118)
    plt.ylim(-5,  65)
    plt.title(title)
    plt.grid()
    
    # error plot
    ax2 = fig.add_subplot(2, 1, 2)
    plt.plot(t_vec[:k], abs(lat_error_vec[:k]), "None", color='r', linestyle="-", linewidth=0.8)
    plt.xlim(t_vec[0], t_vec[-1])
    plt.ylim(0, 2.5)
    plt.xlabel("time [s]")
    plt.ylabel("abs(lateral direction error) [m]")
    plt.grid()

# error plot of all cases
def plot_lat_error(t_vec, lat_error_vec_l):
    color_l = ["k", "b", "r", "orange"]
    N_mabiki = 1
    
    fig = plt.figure(facecolor="w",figsize=(10,4))
    for i in range(len(lat_error_vec_l)):
        plt.plot(t_vec[::N_mabiki], abs(lat_error_vec_l[i][::N_mabiki]), "None", color=color_l[i], linestyle="-", linewidth=3.6)
    
    plt.xlim(t_vec[0], t_vec[-1])
    plt.ylim(0,4)
    plt.xlabel("time [s]")
    plt.ylabel("abs(lateral direction error) [m]")
    plt.grid()
    plt.savefig("lateral_direction_error.png", dpi=300)
    plt.clf()
    plt.close()


def output_video_sample(path, X, t_vec, lat_error_vec, L, title, config):
    # set format of output movie
    # 映像化処理の設定
    framerate = config["output video"]["frame rate"] #[/s]
    interval  = config["output video"]["thinning out interval (1: not thinning out)"] # 画像の間引き間隔(1: 間引きなし)
    savefmt   = config["output video"]["format (.mp4 or .gif)"] # 映像のファイル名
    save_name = "result_"+title+savefmt
    
    imgs = []
    for k in range(0,len(X),25):
        _plot_single_state_sample(path, X, t_vec, lat_error_vec, L, title, k)
        
        # matplotlib -> cv2
        buf = io.BytesIO() # bufferを用意
        plt.savefig(buf, format='png', dpi=300) # bufferに保持
        plt.clf()
        plt.close()
        enc = np.frombuffer(buf.getvalue(), dtype=np.uint8) # bufferからの読み出し
        img = cv2.imdecode(enc, 1) # デコード
        
        imgs.append(img)
        if k == 0:
            height, width, _ = img.shape
            size = (width, height)
    
    if savefmt == '.mp4':
        movie = cv2.VideoWriter(save_name, cv2.VideoWriter_fourcc(*'MP4V'), framerate, size) # 映像の出力設定
        for i in range(len(imgs)):
            movie.write(imgs[i]) # 映像の書き出し
        movie.write(np.zeros_like(imgs[0])) # なぜか最後が切れるので追加
        movie.release() # 映像の書き出しの終了
        
    elif savefmt == '.gif':
        imgs_PIL = [Image.fromarray(cv2.cvtColor(imgs[i], cv2.COLOR_BGR2RGB)) for i in range(len(imgs))]
        imgs_PIL[0].save(save_name, save_all=True, append_images=imgs_PIL[1:], optimize=False, loop=0)
    else:
        print("save format: "+savefmt+" is wrong.")



def _plot_single_state_sample(path, X, t_vec, lat_error_vec, L, title, k):
    distance_tire = 1.5
    fig = plt.figure(facecolor="w",figsize=(6,4))
    ax = fig.add_subplot(1, 1, 1, aspect="equal")
    #plt.plot(path[:,IDX.X], path[:,IDX.Y], color='r', linestyle = "dashed", linewidth=1) 
    plt.plot(X[:k,IDX.X],   X[:k,IDX.Y],   "None", color="blue", linewidth=1.2)
    
    rear_x  = X[k,IDX.X]
    rear_y  = X[k,IDX.Y]
    front_x = X[k,IDX.X] + L*np.cos(X[k,IDX.YAW])
    front_y = X[k,IDX.Y] + L*np.sin(X[k,IDX.YAW])
    
    rear_tire1_x  = rear_x  - distance_tire/2.* np.sin(X[k,IDX.YAW])
    rear_tire2_x  = rear_x  + distance_tire/2.* np.sin(X[k,IDX.YAW])
    front_tire1_x = front_x - distance_tire/2.* np.sin(X[k,IDX.YAW])
    front_tire2_x = front_x + distance_tire/2.* np.sin(X[k,IDX.YAW])
    rear_tire1_y  = rear_y  + distance_tire/2.* np.cos(X[k,IDX.YAW])
    rear_tire2_y  = rear_y  - distance_tire/2.* np.cos(X[k,IDX.YAW])
    front_tire1_y = front_y + distance_tire/2.* np.cos(X[k,IDX.YAW])
    front_tire2_y = front_y - distance_tire/2.* np.cos(X[k,IDX.YAW])
    
    plt.plot([rear_tire1_x, rear_tire2_x, front_tire2_x, front_tire1_x,rear_tire1_x], [rear_tire1_y, rear_tire2_y, front_tire2_y, front_tire1_y, rear_tire1_y], "None", color='k', linestyle="-", linewidth=1.2) # body
    plt.plot([rear_tire1_x, rear_tire2_x, front_tire2_x, front_tire1_x], [rear_tire1_y, rear_tire2_y, front_tire2_y, front_tire1_y], "o", color='k', linestyle="None", markersize=2) # rear_tire
    
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.xlim(-5, 118)
    plt.ylim(-5,  65)
    plt.title(title)
    #plt.grid()
    
    """
    # error plot
    ax2 = fig.add_subplot(2, 1, 2)
    plt.plot(t_vec[:k], abs(lat_error_vec[:k]), "None", color='r', linestyle="-", linewidth=0.8)
    plt.xlim(t_vec[0], t_vec[-1])
    plt.ylim(0, 2.5)
    plt.xlabel("time [s]")
    plt.ylabel("abs(lateral direction error) [m]")
    plt.grid()
    """
