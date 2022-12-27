import matplotlib.pyplot as plt
import matplotlib.mathtext as mtxt
import numpy as np

# 散布図の作成と出力
# x_data: np.ndarray
# y_data: np.ndarray
def make_scatter(x_data, y_data, save_name, \
              color = ["k","red"], marker = ["None", "o"], marker_size = [], \
              line_style = ["-", "None"], line_width = [], \
              label = ["x","y"], xlim = [], ylim = [], title = "", \
              grid = True, plot_size = 2, axis_format = []):
    
    # プロットエリアの設定
    plot_area = [plot_size, plot_size]
    
    # 定義域と値域の設定
    min_ = min(min(y_data), min(x_data))
    max_ = max(max(y_data), max(x_data))
    delta_ = max_ - min_
    alpha_ = 0.1
    min_ = min_ - delta_*alpha_
    max_ = max_ + delta_*alpha_
    lim = [min_, max_]
    
    # データの型をリストに変更し、基準線を追加
    x_data = [lim]+[x_data]
    y_data = [lim]+[y_data]
    
    # グラフの作成と出力
    make_plot(x_data, y_data, save_name, \
              color = color, marker = marker, marker_size = marker_size, \
              line_style = line_style, line_width = line_width, \
              label = label, xlim = lim, ylim = lim, title = title, \
              grid = grid, plot_area = plot_area, axis_format = axis_format)

# グラフの作成と出力
# x_data: np.ndarray
# y_data: np.ndarray or list[np.ndarray]
def make_plot(x_data, y_data, save_name, \
              color = [], marker = [], marker_size = [], \
              line_style = [], line_width = [], \
              label = ["x","y"], xlim = [], ylim = [], title = "", \
              grid = False, plot_area = [2.4, 1.8], axis_format = []):
    
    # 色: https://matplotlib.org/2.0.2/examples/color/named_colors.html
    
    # 上付き文字，下付き文字の配置を適正化
    #mtxt.FontConstantsBase = mtxt.ComputerModernFontConstants
    
    # グラフエリア全体の書式設定
    plt.rcParams['font.family'] = 'times new roman'
    plt.rcParams['mathtext.default'] = 'regular'
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.rcParams['xtick.major.width'] = 1.0
    plt.rcParams['ytick.major.width'] = 1.0
    plt.rcParams['font.size'] = 8
    plt.rcParams['axes.linewidth'] = 1.0
    
    # プロットエリアの設定
    plt.figure(figsize = (plot_area[0], plot_area[1]))
    
    # 軸の値のフォーマット
    if len(axis_format)==2:
        plt.gca().xaxis.set_major_formatter(plt.FormatStrFormatter(axis_format[0]))
        plt.gca().yaxis.set_major_formatter(plt.FormatStrFormatter(axis_format[1]))
    
    # y_dataの型をリストに変更
    if type(y_data)==np.ndarray:
        y_data = [y_data]
    
    # 色の設定
    N_y_data = len(y_data)
    if len(color)<N_y_data:
        if N_y_data==1:
            color = ["red"]
        elif N_y_data==2:
            color = ["red", "blue"]
        elif N_y_data==3:
            color = ["red", "blue", "green"]
        elif N_y_data==4:
            color = ["red", "orange", "blue", "deepskyblue"]
        elif N_y_data==5:
            color = ["red", "blue", "green", "orange", "purple"]
        elif N_y_data==6:
            color = ["red", "orange", "blue", "deepskyblue", "purple", "magenta"]
        else:
            color = ["k" for i in range(N_y_data)]
    if len(marker)<N_y_data:
        marker    = ["o" for i in range(N_y_data)]
    if len(marker_size)<N_y_data:
        marker_size = [2 for i in range(N_y_data)]
    if len(line_style)<N_y_data:
        line_style = ["-" for i in range(N_y_data)]
    if len(line_width)<N_y_data:
        line_width  = [1 for i in range(N_y_data)]
    
    # プロット
    if type(x_data)==np.ndarray:
        for i in range(N_y_data):
            plt.plot(x_data, y_data[i],        # データ
                     color      = color[i],      # 色
                     marker     = marker[i],     # マーカーのスタイル
                     markersize = marker_size[i], # マーカーのサイズ
                     linestyle  = line_style[i],  # 線のスタイル
                     linewidth  = line_width[i])  # 線の太さ
    else:
        for i in range(N_y_data):
            plt.plot(x_data[i], y_data[i],        # データ
                     color      = color[i],      # 色
                     marker     = marker[i],     # マーカーのスタイル
                     markersize = marker_size[i], # マーカーのサイズ
                     linestyle  = line_style[i],  # 線のスタイル
                     linewidth  = line_width[i])  # 線の太さ
        
    # 軸ラベル
    plt.xlabel(label[0])
    plt.ylabel(label[1])
    
    # 定義域と値域
    if len(xlim)==2:
        plt.xlim(xlim[0], xlim[1])
    if len(ylim)==2:
        plt.ylim(ylim[0], ylim[1])
    
    # 目盛の最大個数
    plt.locator_params(axis='x', nbins=6)
    plt.locator_params(axis='y', nbins=6)
    
    # タイトル
    plt.title(title)
    
    # グリッド
    plt.grid(visible=grid, which='major')
    
    # 軸の値の表示にオフセット(+1.05e9 など)を利用しない
    #plt.gca().xaxis.get_major_formatter().set_useOffset(False)
    
    # 軸の数字が整数になるようにする
    #plt.gca().yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    
    # レイアウトの調整
    plt.tight_layout()
    
    # 出力
    plt.savefig(save_name+".png", transparent=True, dpi=600, format="png")
    plt.close()
