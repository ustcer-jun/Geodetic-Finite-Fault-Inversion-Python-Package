import numpy as np
import xarray as xr
import matplotlib as mpl
import matplotlib.pyplot as plt
from pyproj import CRS, Transformer
from typing import Sequence, Union
from matplotlib import cm
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator


def generate_grid(mode, patchi, dw, dl, inc=1,plt_flag = 1,ax = None):
    """
    construct finite fault model 

    """
    # ---  basic parameters ---
    X0, Y0, Z0 = patchi["x"], patchi["y"], patchi["z"]
    length, width = patchi["len"], patchi["wid"]
    dip = np.deg2rad(patchi["dip"])
    strike = np.deg2rad(patchi["strike"])

    # --- 层数计算（保证为 int） ---
    if inc == 1:
        nw = int(round(width / dw))
    else:
        nw = int(round(np.log(width / dw * (inc - 1) + 1) / np.log(inc)))
    nl = int(round(length / dl))

    nw = max(nw, 1)
    nl = max(nl, 1)

    # --- 每层尺寸与网格计数初始化 ---
    dW = np.zeros(nw, dtype=float)
    dL = np.zeros(nw, dtype=float)
    nL = np.zeros(nw, dtype=int)

    # 宽度方向（dW）
    if inc == 1:
        dW[:] = width / nw
    else:
        dW[-1] = width * (inc - 1) / (inc**nw - 1)
        if nw > 1:
            for j in range(nw - 1):
                dW[j] = dW[-1] * inc**(nw - j - 1)

    # 长度方向（最后一层均分）
    dL[-1] = length / nl
    nL[-1] = nl

    # 从下向上反推每层 dL 和 nL（保留原逻辑，但更稳健）
    for j in range(nw - 2, -1, -1):
        est = round(length / dL[j + 1] / inc)
        if est <= 1:
            dL[j] = length
            nL[j] = 1
        else:
            nL[j] = int(est)
            dL[j] = length / nL[j]

    # --- 总单元数检查 ---
    n_total = int(np.sum(nL))
    if n_total <= 0:
        raise ValueError("Computed zero total subfaults (n_total=0). Check dw, dl, len, wid, inc parameters.")

    # --- 预分配数组 ---
    XC = np.zeros(n_total, dtype=float)
    YC = np.zeros(n_total, dtype=float)
    ZC = np.zeros(n_total, dtype=float)
    ll = np.zeros(n_total, dtype=float)
    ww = np.zeros(n_total, dtype=float)
    XB = np.zeros(4 * n_total, dtype=float)
    YB = np.zeros(4 * n_total, dtype=float)
    ZB = np.zeros(4 * n_total, dtype=float)

    coss, sins = np.cos(strike), np.sin(strike)
    cosd, sind = np.cos(dip), np.sin(dip)

    # --- generate grid  ---
    offset = 0
    for j in range(nw):
        nj = int(nL[j])
        if nj == 0:
            continue

        # 中心位置（奇偶统一处理）
        if nj % 2 == 1:
            center = (nj - 1) / 2.0
        else:
            center = nj / 2.0 - 0.5

        # 计算当前层相对于原点在宽度方向的累计偏移（向下累计）
        sum_below = np.sum(dW[j+1:]) if j < nw - 1 else 0.0
        depth0 = -Z0 + sum_below * abs(sind)
        ex = sum_below * cosd

        for i in range(nj):
            k = offset + i  # 正确的线性索引（避免 -1）
            ey = (i - center) * dL[j]

            XC[k] = ex * coss + ey * sins + X0
            YC[k] = -ex * sins + ey * coss + Y0
            ZC[k] = -depth0
            ll[k] = dL[j]
            ww[k] = dW[j]

            Lx = 0.5 * dL[j] * sins
            Ly = 0.5 * dL[j] * coss
            W = dW[j] * cosd

            # 四个角（以中心为基准)
            XB[4 * k:4 * k + 4] = -np.array([Lx - W * coss, Lx, -Lx, -Lx - W * coss]) + XC[k]
            YB[4 * k:4 * k + 4] = -np.array([Ly + W * sins, Ly, -Ly, -Ly + W * sins]) + YC[k]
            ZB[4 * k:4 * k + 4] = np.array([-dW[j] * sind, 0.0, 0.0, -dW[j] * sind]) + ZC[k]

        offset += nj

    # --- 输出坐标（mode ）---
    if mode == 0:
        X, Y, Z = XC, YC, ZC
    elif mode == 1:
        X = XC - 0.5 * ll * np.sin(strike)
        Y = YC - 0.5 * ll * np.cos(strike)
        Z = ZC
    else:
        raise ValueError("mode must be 0 or 1")

    # --- plotting: km） ---
    if plt_flag == 1:
        # 如果没有传入 ax，则创建新的 figure
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection="3d")
            show_fig = True
        else:
            show_fig = False  # 不在这里 plt.show()，外部控制

        X_km, Y_km, Z_km = X / 1000.0, Y / 1000.0, Z / 1000.0
        XB_km, YB_km, ZB_km = XB / 1000.0, YB / 1000.0, ZB / 1000.0
        X0_km, Y0_km, Z0_km = X0 / 1000.0, Y0 / 1000.0, Z0 / 1000.0

        ax.scatter(X_km, Y_km, Z_km, marker="o", s=10, label="Grid Centers")
        ax.scatter([X0_km], [Y0_km], [Z0_km], marker="x", s=40, label="Fault Origin")

        for jj in range(n_total):
            idx = slice(4*jj, 4*jj+4)
            # 画闭合边界（把首点加到末尾）
            xline = np.concatenate([XB_km[idx], XB_km[idx.start:idx.start+1]])
            yline = np.concatenate([YB_km[idx], YB_km[idx.start:idx.start+1]])
            zline = np.concatenate([ZB_km[idx], ZB_km[idx.start:idx.start+1]])
            ax.plot(xline, yline, zline, '-', linewidth=0.6)

        ax.set_xlabel("W-E (km)")
        ax.set_ylabel("S-N (km)")
        ax.set_zlabel("Depth (km)")
        ax.set_title("3D - Fault Plane")
        ax.grid(False)
        ax.set_box_aspect([1, 1, 0.5])
        # ax.legend()
        # 仅在函数内部创建 figure 时才 show
        if show_fig:
            plt.show()

    return X, Y, Z, XB, YB, ZB, ll, ww, nw, nl, nL, dL, dW

def plot_data(xx, yy, zz, title=""):
    xx = xx / 1000
    yy = yy / 1000

    fig, ax = plt.subplots(figsize=(6, 5), constrained_layout=True)

    numcol = 200
    cmap = plt.get_cmap("jet", numcol)
    clim = [-np.max(np.abs(zz)), np.max(np.abs(zz))]
    
    sc = ax.scatter(
        xx, yy,
        c=zz, cmap=cmap,
        vmin=clim[0], vmax=clim[1],
        s=12, marker="o", edgecolor="k", linewidth=0.2
    )
    # 添加 colorbar
    fig.colorbar(sc, ax=ax, orientation="vertical", fraction=0.43, pad=0.03)

    ax.set_xlabel("UTM W - E (km)")
    ax.set_ylabel("UTM S - N (km)")
    ax.set_title(title)
    
    # 返回 axes, 对象，供外部函数使用
    return  fig, ax


def plot_gps(ax, xx, yy, dat):
    """
    绘制 GNSS 偏移 (水平 E/N 分量 + 垂直分量箭头)
    数据单位: mm → 自动转换为 cm

    参数
    ----
    ax : matplotlib Axes
        绘制的坐标轴
    xx, yy : array
        台站坐标 (米)
    dat : numpy array
        GPS 数据表，列要求:
        dat[:,2] = E (mm), dat[:,3] = N (mm), dat[:,4] = U (mm)

    返回
    ----
    qE, qU : quiver 句柄
    """

    # 单位转换: mm -> cm
    U_cm = dat[:, 2] / 10.0
    V_cm = dat[:, 3] / 10.0
    W_cm = dat[:, 4] / 10.0

    # 图幅范围 (km)
    x_min, x_max = np.nanmin(xx), np.nanmax(xx)
    y_min, y_max = np.nanmin(yy), np.nanmax(yy)
    axis_extent = max(x_max - x_min, y_max - y_min, 1.0) / 1000.0  # km

    # 水平位移模长
    mag_h = np.sqrt(U_cm**2 + V_cm**2)
    max_h = np.nanmax(mag_h) if mag_h.size else 1.0
    max_u = np.nanmax(np.abs(W_cm)) if W_cm.size else 1.0

    # 缩放因子: 最大箭头 ~ 图长度的 0.3
    scale_h = max_h / (0.5 * axis_extent)  # 100: cm/km
    scale_u = max_u / (0.3 * axis_extent)

    # 绘制水平箭头 (红色)
    qE = ax.quiver(xx/1000, yy/1000, U_cm, V_cm, color="r",
                   scale=scale_h, scale_units="xy", angles="xy",
                   width=0.004, headwidth=3.5, headlength=5, label="水平位移")

    # 绘制竖直箭头 (蓝色)
    qU = ax.quiver(xx/1000, yy/1000, np.zeros_like(W_cm), W_cm, color="b",
                   scale=scale_u, scale_units="xy", angles="xy",
                   width=0.004, headwidth=3.5, headlength=5, label="垂直位移")

    # 添加参考比例尺箭头
    ax.quiverkey(qE, X=0.85, Y=0.1, U=10,  # 10 cm 水平
                 label="10 cm horizontal", labelpos="E", coordinates="axes", color="r")
    ax.quiverkey(qU, X=0.85, Y=0.05, U=2,   # 2 cm 垂直
                 label="2 cm vertical", labelpos="E", coordinates="axes", color="b")

    # 轴美化
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("UTM W - E (km)")
    ax.set_ylabel("UTM S - N (km)")
    ax.grid(True)

    return qE, qU

def plot_geological_obs(xO, yO, dO, azO, scale_ref=100.0, ax=None):
    """
    绘制地质观测的位移箭头图 (单位 cm)

    参数
    ----
    xO, yO : array
        台站 UTM 坐标 (m)
    dO : array
        位移大小 (cm)
    azO : array
        位移方向 (azimuth, degree)
    patches : optional
        断层迹线 patch，用于 plot_trace
    scale_ref : float
        参考箭头大小 (cm)，默认 100 cm = 1 m
    ax : matplotlib.axes.Axes, optional
        如果传入则在现有坐标轴绘制，否则新建一个图
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 6))

    # 计算箭头分量
    Ux = dO * np.cos(np.deg2rad(90 - azO))  # 东向分量
    Uy = dO * np.sin(np.deg2rad(90 - azO))  # 北向分量

    # 绘制箭头
    q = ax.quiver(
        xO/1000, yO/1000, Ux/100, Uy/100,
        color="tab:green", scale=0.5, scale_units="xy", angles="xy",
        width=0.004, headwidth=3.5, headlength=5
    )

    # 添加比例尺箭头
    ax.quiverkey(
        q, X=0.8, Y=0.1, U=2,
        label=f"Offset {scale_ref/100:.1f} m",  # 转换为 m
        labelpos="E", coordinates="axes",
        fontproperties={"size": 12}
    )

    # 坐标轴设置
    ax.set_title("Geological observation, projected to trace", fontsize=16)
    ax.set_xlabel("UTM W - E (km)")
    ax.set_ylabel("UTM S - N (km)")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True)

    return ax

def plot_patches(filename: str, mode: int, ratio: float, title):

    a = np.loadtxt(filename)
    if a.ndim == 1:
        a = a[np.newaxis, :]

    xs = a[:, 3]/1000; ys = a[:, 4]/1000; zs = a[:, 5]/1000
    L = a[:, 6]/1000; W = a[:, 7]/1000
    dip = np.deg2rad(a[:, 8]); strike = np.deg2rad(a[:, 9])
    Us = a[:, 10]; Ud = a[:, 11]; Un = a[:, 12]

    # slip 大小选择
    if mode == 1:
        slip = Us
    elif mode == 2:
        slip = Ud
    elif mode == 3:
        slip = Un
    elif mode == 12:
        slip = np.sqrt(Us**2 + Ud**2)
    elif mode == 13:
        slip = np.sqrt(Us**2 + Un**2)
    elif mode == 23:
        slip = np.sqrt(Ud**2 + Un**2)
    else:
        raise ValueError("ERROR: Please choose correct mode (1,2,3,12,13,23)")

    n = xs.size

    # colormap setup
    smin, smax = np.nanmin(slip), np.nanmax(slip)
    if np.isclose(smin, smax):
        eps = abs(smin)*0.01 if smin != 0 else 1.0
        norm = mpl.colors.Normalize(vmin=smin-eps, vmax=smax+eps)
    else:
        norm = mpl.colors.Normalize(vmin=smin, vmax=smax)
    cmap = cm.get_cmap('jet')

    # ======= 使用一个 figure，左右两个子图 =======
    fig = plt.figure(figsize=(10,5), constrained_layout=True)
    gs = gridspec.GridSpec(1, 2, width_ratios=[1.5, 1])
    ax0 = fig.add_subplot(gs[0], projection='3d')
    ax1 = fig.add_subplot(gs[1])
    ax0.grid(True)

    # 绘制 patch 和箭头
    for j in range(n):
        tx = np.array([[xs[j], xs[j]+L[j]*np.sin(strike[j])],
                       [xs[j]+W[j]*np.cos(dip[j])*np.cos(strike[j]),
                        xs[j]+L[j]*np.sin(strike[j])+W[j]*np.cos(dip[j])*np.cos(strike[j])]])
        ty = np.array([[ys[j], ys[j]+L[j]*np.cos(strike[j])],
                       [ys[j]-W[j]*np.cos(dip[j])*np.sin(strike[j]),
                        ys[j]+L[j]*np.cos(strike[j])-W[j]*np.cos(dip[j])*np.sin(strike[j])]])
        tz = np.array([[zs[j], zs[j]],
                       [zs[j]-W[j]*np.sin(dip[j]), zs[j]-W[j]*np.sin(dip[j])]])

        color_rgba = cmap(norm(slip[j]))
        facecolors = np.tile(np.array(color_rgba)[None,None,:], (2,2,1))
        ax0.plot_surface(tx, ty, tz, rstride=1, cstride=1, facecolors=facecolors,
                         edgecolor='none', antialiased=True, shade=False)

        cx, cy, cz = np.mean(tx), np.mean(ty), np.mean(tz)
        x1 = Us[j]*np.sin(strike[j])-Ud[j]*np.cos(dip[j])*np.cos(strike[j])
        x2 = Us[j]*np.cos(strike[j])+Ud[j]*np.cos(dip[j])*np.sin(strike[j])
        x3 = Ud[j]*np.sin(dip[j])
        ax0.quiver(cx, cy, cz, x1*ratio, x2*ratio, x3*ratio,
                   color='k', linewidths=0.5, arrow_length_ratio=0.2, normalize=False)

    # colorbar 单独绑定左图底部
    mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
    mappable.set_array(slip)
    # 在 figure 底部单独创建 colorbar，避免压缩左图
    cax = fig.add_axes([0.2, 0.1, 0.3, 0.04])  # [left, bottom, width, height]
    cbar = fig.colorbar(mappable, cax=cax, orientation='horizontal')
    cbar.set_label("Slip (cm)")

    # 设置视角和坐标系
    ax0.view_init(elev=15, azim=80)
    ax0.set_xlabel('W - E (km)'); ax0.set_ylabel('S - N (km)'); ax0.set_zlabel('Depth (km)')
    ax0.set_title(title);
    ax0.tick_params(axis='x', direction='in', length=0, pad=4)
    ax0.tick_params(axis='y', direction='in', length=0, pad=4)
    ax0.tick_params(axis='z', direction='in', length=0, pad=6)
    ax0.set_box_aspect([1,1,0.5])
    ax0.zaxis.set_major_locator(MaxNLocator(nbins=4))  # 最多 5 个主刻度
    # slip vs depth 绘图
    dpth_col = a[:,2].astype(int)
    max_layer = int(np.max(dpth_col))
    sp = np.zeros(max_layer)
    dp = np.zeros(max_layer)
    for jj in range(1, max_layer+1):
        ii = np.where(dpth_col==jj)[0]
        if ii.size==0:
            sp[jj-1] = 0.0
            dp[jj-1] = np.nan
            continue
        if mode in [1,13]:
            sp[jj-1] = np.sum(np.abs(Us[ii])*L[ii])/100.0
        else:
            sp[jj-1] = np.sum(np.abs(Ud[ii])*L[ii])/100.0
        dp[jj-1] = np.mean(zs[ii]-W[ii]*np.sin(dip[ii])/2.0)

    ax1.plot(sp, dp, color='blue', linewidth=2, marker='o', markersize=6, markerfacecolor='white')
    ax1.set_xlabel('Cumulative Strike Slip (m*km)'); ax1.set_ylabel('Depth (km)')
    ax1.set_title('Slip vs Depth')
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.set_xlim(left=0)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.margins(x=0.05, y=0.05)

    # 整体 figure 调整，使两张图尽量紧凑
    plt.show()
    mu = 30e9  # 剪切模量，单位 Pa
    magnitude = 2/3 * np.log10(
        mu * np.sqrt(
            np.sum(Us/100 * (L*1000) * (W*1000))**2 + np.sum(Ud/100 * (L*1000) * (W*1000))**2
        )
    ) - 6.07

    print(f"Seismic Moment: MW {magnitude:.2f}")
    return sp, dp


import numpy as np

def proj_gov2trace(patch, xall, yall, dall, azall, tall, sall, dat_gov, interval, closeness):
    xxx, yyy, ddd, tpp, azOO, sOO = [], [], [], [], [], []

    dat_gov_out = dat_gov.copy()

    for i in range(len(dat_gov)):
        # 提取当前测线的观测点
        start = int(np.sum(dat_gov[:i]))
        end = int(np.sum(dat_gov[:i+1]))
        x = np.array(xall[start:end]).flatten()
        y = np.array(yall[start:end]).flatten()
        d = np.array(dall[start:end]).flatten()
        az = np.array(azall[start:end]).flatten()
        t = np.array(tall[start:end]).flatten()
        s = np.array(sall[start:end]).flatten()

        xx, yy, ux, uy, dd, tp, ss = [], [], [], [], [], [], []

        num = np.zeros(len(patch), dtype=int)
        dis = np.zeros(len(patch))
        num_num = np.zeros(len(patch)+1, dtype=int)

        # 计算每个断层片段的划分数和步长
        for j in range(len(patch)):
            num[j] = int(np.floor(patch[j]['len'] / interval))
            dis[j] = patch[j]['len'] / num[j]
            num_num[j+1] = num_num[j] + num[j]

        # 对每个断层片段
        for j in range(len(patch)):
            k = -1 / np.tan(np.radians(90 - patch[j]['strike']))
            x0 = patch[j]['x'] - 0.5 * patch[j]['len'] * np.sin(np.radians(patch[j]['strike']))
            y0 = patch[j]['y'] - 0.5 * patch[j]['len'] * np.cos(np.radians(patch[j]['strike']))

            for l in range(num[j]):
                x1 = x0 + dis[j] * np.sin(np.radians(patch[j]['strike']))
                y1 = y0 + dis[j] * np.cos(np.radians(patch[j]['strike']))

                # 定义左、右侧观测点区域
                jdl = (((y >= k*(x-x0)+y0) & (y <= k*(x-x1)+y1)) | ((y <= k*(x-x0)+y0) & (y >= k*(x-x1)+y1))) & (
                        ((y <= np.tan(np.radians(90 - patch[j]['strike']))*(x-x0)+y0) &
                         (y > np.tan(np.radians(90 - patch[j]['strike']))*(x-x0)+y0 - closeness/np.sin(np.radians(patch[j]['strike']))))
                        |
                        ((y >= np.tan(np.radians(90 - patch[j]['strike']))*(x-x0)+y0) &
                         (y < np.tan(np.radians(90 - patch[j]['strike']))*(x-x0)+y0 - closeness/np.sin(np.radians(patch[j]['strike']))))
                )

                jdr = (((y >= k*(x-x0)+y0) & (y <= k*(x-x1)+y1)) | ((y <= k*(x-x0)+y0) & (y >= k*(x-x1)+y1))) & (
                        ((y > np.tan(np.radians(90 - patch[j]['strike']))*(x-x0)+y0) &
                         (y < np.tan(np.radians(90 - patch[j]['strike']))*(x-x0)+y0 + closeness/np.sin(np.radians(patch[j]['strike']))))
                        |
                        ((y < np.tan(np.radians(90 - patch[j]['strike']))*(x-x0)+y0) &
                         (y > np.tan(np.radians(90 - patch[j]['strike']))*(x-x0)+y0 + closeness/np.sin(np.radians(patch[j]['strike']))))
                )

                if np.sum(jdl) > 0 or np.sum(jdr) > 0:
                    xtmp = (np.sum(x[jdl]) + np.sum(x[jdr])) / (np.sum(jdl) + np.sum(jdr))
                    ytmp = (np.sum(y[jdl]) + np.sum(y[jdr])) / (np.sum(jdl) + np.sum(jdr))
                    ttmp = (np.sum(t[jdl]) + np.sum(t[jdr])) / (np.sum(jdl) + np.sum(jdr))
                    stmp = (np.sum(s[jdl]) + np.sum(s[jdr])) / (np.sum(jdl) + np.sum(jdr))
                else:
                    xtmp, ytmp, ttmp, stmp = x0, y0, 0, 0

                dtl = np.sum(d[jdl]*np.cos(np.radians(az[jdl]-patch[j]['strike'])))/np.sum(jdl) if np.sum(jdl) > 0 else 0
                dtr = np.sum(d[jdr]*np.cos(np.radians(az[jdr]-patch[j]['strike'])))/np.sum(jdr) if np.sum(jdr) > 0 else 0

                strike = np.radians(patch[j]['strike'])
                vec = np.array([np.sin(strike), np.cos(strike)])
                dot_val = np.dot(np.array([xtmp - x0, ytmp - y0]), vec)

                xt = x0 + dot_val * np.sin(strike)
                yt = y0 + dot_val * np.cos(strike)
                st = patch[j]['strike']

                xx.append(xt)
                yy.append(yt)
                ux.append((dtl+dtr)*np.sin(np.radians(st)))
                uy.append((dtl+dtr)*np.cos(np.radians(st)))
                dd.append(dtl+dtr)
                tp.append(ttmp)
                ss.append(stmp)

                # 移动到下一个子段
                x0, y0 = x1, y1

                # 平滑当前片段区间
                seg_start = num_num[j]
                seg_end = num_num[j+1]
                if seg_end > seg_start:
                    dd_seg = np.convolve(dd[seg_start:seg_end], np.ones(5)/5, mode='same')
                    dd[seg_start:seg_end] = dd_seg

        # 删除 dd==0 的点
        djd = np.where(np.array(dd) == 0)[0]
        xx = np.delete(np.array(xx), djd)
        yy = np.delete(np.array(yy), djd)
        dd = np.delete(np.array(dd), djd)
        ux = np.delete(np.array(ux), djd)
        uy = np.delete(np.array(uy), djd)
        tp = np.delete(np.array(tp), djd)
        ss = np.delete(np.array(ss), djd)

        # 方位角
        azO = 90 - np.degrees(np.arctan2(uy, ux))
        azO[azO < 0] += 360

        dat_gov_out[i] = len(xx)

        # 累积
        xxx.extend(xx)
        yyy.extend(yy)
        ddd.extend(dd)
        tpp.extend(tp)
        azOO.extend(azO)
        sOO.extend(ss)

    return np.array(xxx), np.array(yyy), np.array(ddd), np.array(azOO), np.array(tpp), np.array(sOO), np.array(dat_gov_out)

def Read_grd_file(filename, data_name="displacement"):
    """
    Read netCDF fiel (.grd)

    Return
    ----
    X : 1D ndarray
    Y : 1D ndarray
    Data : 2D ndarray
    attrs : dict
    """
    ds = xr.open_dataset(filename)

    # 自动检测变量名
    if data_name not in ds.data_vars:
        data_name = list(ds.data_vars.keys())[0]
        print(f"⚠️ 未找到变量 '{data_name}'，自动使用 {data_name}")

    Data = ds[data_name].values

    # ----------- 坐标名判断（按你的要求）----------------
    # X coordinate
    if "x" in ds:
        X = ds["x"].values
    elif "lon" in ds:
        X = ds["lon"].values
    else:
        raise KeyError("❌ 未找到 X 坐标，期望 'x' 或 'lon'")

    # Y coordinate
    if "y" in ds:
        Y = ds["y"].values
    elif "lat" in ds:
        Y = ds["lat"].values
    else:
        raise KeyError("❌ 未找到 Y 坐标，期望 'y' 或 'lat'")
    # -----------------------------------------------------

    attrs = ds[data_name].attrs.copy()
    attrs.update(ds.attrs)

    ds.close()
    return X, Y, Data, attrs

def rednoise(N1, N2, r, show_plot=False, dx=1.0, dy=1.0):
    """
    Red noise simulator (2D) with optional visualization.

    Parameters
    ----------
    N1, N2 : int
        Size of the output array.
    r : float
        Spectral exponent (e.g., r=1 for red noise, r=2 for Brownian noise)
    show_plot : bool, optional
        If True, display a 2D plot of the generated red noise.
    dx, dy : float, optional
        Grid spacing in meters. Used for converting x/y axes to km.

    Returns
    -------
    y : 2D numpy array
        Normalized red noise with zero mean and unit RMS.
    """

    # Make dimensions even
    M1 = N1 + 1 if N1 % 2 else N1
    M2 = N2 + 1 if N2 % 2 else N2

    # Generate white noise 
    x = np.random.randn(M1, M2)

    # FFT and shift
    X = np.fft.fftshift(np.fft.fft2(x))

    # Frequency grids
    n1 = np.concatenate((np.arange(-M1//2, 0), np.arange(1, M1//2 + 1)))[:, np.newaxis]
    n2 = np.concatenate((np.arange(-M2//2, 0), np.arange(1, M2//2 + 1)))[np.newaxis, :]

    # Radial frequency magnitude
    kk = np.sqrt(n1**2 + n2**2)
    kk[kk == 0] = 1  # avoid division by zero

    # Apply spectral slope
    X = X / kk**r

    # Inverse FFT
    y = np.fft.ifft2(np.fft.ifftshift(X))
    y = np.real(y[:N1, :N2])

    # Remove mean and normalize RMS
    y -= np.mean(y)
    yrms = np.sqrt(np.mean(y**2))
    y /= yrms

    # Visualization
    if show_plot:
        extent = [0, N2*dx/1000, 0, N1*dy/1000]  # convert to km
        plt.figure(figsize=(6,5))
        im = plt.imshow(y, origin='lower', extent=extent, cmap='jet', aspect='auto')
        plt.colorbar(im, label='Normalized amplitude')
        plt.xlabel('UTM W - E (km)')
        plt.ylabel('UTM S - N (km)')
        plt.title(f'Red Noise Simulation (r={r})')
        plt.tight_layout()
        plt.show()

    return y


def get_patch_endpoints(patches):
    segments = []
    for patch in patches:
        x, y = patch["x"], patch["y"]
        length = patch["len"]
        strike = patch["strike"]

        # strike to degree
        strike_rad = np.radians(strike)

        # 计算端点
        x1 = x - 0.5 * length * np.sin(strike_rad)
        y1 = y - 0.5 * length * np.cos(strike_rad)
        x2 = x + 0.5 * length * np.sin(strike_rad)
        y2 = y + 0.5 * length * np.cos(strike_rad)

        # save to segmement
        segments.append((x1, y1, x2, y2))

    return segments

def plot_trace(patches, ax=None, if_show=False):
    """
    plot fault trace and patch mid-point (optional plotting)

    Parameters
    ----------
    patches : list of dict
        each dict should include fault parameters: x, y, len, strike, etc.
    ax : matplotlib.axes.Axes or None
        If None, a new figure and axes are created.
    if_show : bool
        If True (default), draw the trace (endpoints and connecting line) on ax.
        If False, no plotting is performed (but endpoints are still computed and returned).

    Returns
    -------
    x1_list, y1_list, x2_list, y2_list : list
        lists of endpoints for each patch
    """
    if if_show:
        if ax is None:
            fig, ax = plt.subplots()

    N = len(patches)
    x1_list, y1_list, x2_list, y2_list = [], [], [], []

    # extract patch parameters (center points)
    xs = np.array([p["x"] for p in patches])
    ys = np.array([p["y"] for p in patches])

    # (kept commented) plot patch mid-point if you want later
    # ax.plot(xs/1000, ys/1000, "bo", label="Patch center")

    for j in range(N):
        x, y = patches[j]["x"], patches[j]["y"]
        length = patches[j]["len"]
        strike = patches[j]["strike"]

        # strike degree to rad
        strike_rad = np.radians(strike)

        # calculate the endpoints
        x1 = x - 0.5 * length * np.sin(strike_rad)
        y1 = y - 0.5 * length * np.cos(strike_rad)
        x2 = x + 0.5 * length * np.sin(strike_rad)
        y2 = y + 0.5 * length * np.cos(strike_rad)
        print(x1,y1,y2,x2)
        x1_list.append(x1)
        y1_list.append(y1)
        x2_list.append(x2)
        y2_list.append(y2)

        # plot trace and mid-point only when requested
        if if_show:
            ax.plot(x1/1000, y1/1000, "r.")
            ax.plot(x2/1000, y2/1000, "b.")
            ax.plot([x1/1000, x2/1000], [y1/1000, y2/1000], "k-")

            ax.axis("equal")
    # ax.grid(True)

    return x1_list, y1_list, x2_list, y2_list

def trace2patch(lon, lat, xo, yo, default_wid=20e3, default_dip=90):
    """
    from geographic (fault trace) coordinates  to generate fault patch geometric parameters
    lon, lat : list/array, (endpoint of fault trace, total 2N, N is the num of fault)
    xo, yo   : trace (logitude, latitude coordinates)
    default_wid : float default width
    default_dip : float default dip

    return 
    ----
    patches : list of dict
        includs x, y, z, len, wid, dip, strike
        x,y,z the transformed relative UTM coordinates
    """

    # lon, lat -> UTM
    x0, y0, zone, hem = utm2ll(xo, yo, i_type=1)   # reference point
    X, Y, _, _ = utm2ll(np.array(lon), np.array(lat), i_type=1)  # total coordinates

    X = X - x0
    Y = Y - y0

    patches = []
    nseg = len(X) // 2
    for j in range(nseg):
        i1, i2 = 2 * j, 2 * j + 1
        xm = (X[i1] + X[i2]) / 2
        ym = (Y[i1] + Y[i2]) / 2
        length = np.sqrt((X[i2] - X[i1]) ** 2 + (Y[i2] - Y[i1]) ** 2)
        strike = 90 - np.degrees(np.arctan2(Y[i2] - Y[i1], X[i2] - X[i1]))

        patch = {
            "x": xm,
            "y": ym,
            "z": 0.0,
            "len": length,
            "wid": default_wid,
            "dip": default_dip,
            "strike": strike,
        }
        patches.append(patch)

        # print result
        print(f"        {{trace{j+1}}}")
        for k, v in patch.items():
            print(f"        {k:>5s} = {v:.6e}")
        print("")

    # plot the fault traces
    plt.plot(X, Y, "r-", label="Fault trace")
    for p in patches:
        plt.plot(p["x"], p["y"], "bo")
    plt.plot(0, 0, "b*", label="Origin")
    plt.axis("equal")
    plt.grid(True)
    plt.legend()
    plt.show()

    return patches

def utm2ll(xi, yi, i_zone=None, i_type=1, northern=True, centered=True):
    """
    Convert between lon/lat <-> UTM.
    - i_type == 1: lon,lat -> easting,northing (returns UTM in meters).
    - i_type == 2: easting,northing -> lon,lat.

    Parameters:
      xi, yi : scalar or array-like
      i_zone : UTM zone (int). If None and i_type==1, will be inferred from lon.
      northern: True if northern hemisphere (used for i_type==2 or to set EPSG).
      centered: If True, for i_type==1 will return easting centered (easting-500000).
                For i_type==2, if centered=True it will assume the passed easting is centered
                and add 500000 before converting. Default False (work with standard UTM).
    Returns:
      For i_type==1: (easting, northing)
      For i_type==2: (lon, lat)
    """
    # coerce to arrays for pyproj
    lonlat_input = (i_type == 1)
    if i_type == 1:
        lon = np.atleast_1d(xi).astype(float)
        lat = np.atleast_1d(yi).astype(float)

        if i_zone is None:
            i_zone = int(np.floor(((lon[0] + 180.0) % 360.0) / 6.0) + 1)

        epsg_code = 32600 + i_zone if northern else 32700 + i_zone
        crs_latlon = CRS.from_epsg(4326)
        crs_utm = CRS.from_epsg(epsg_code)
        transformer = Transformer.from_crs(crs_latlon, crs_utm, always_xy=True)
        eo, no = transformer.transform(lon, lat)  # standard UTM (includes 500000 false easting)
        if centered:
            eo = eo - 500000.0
        # return scalars if input scalars
        if np.isscalar(xi) and np.isscalar(yi):
            return float(eo[0]) if hasattr(eo, "__len__") else float(eo), float(no[0]) if hasattr(no, "__len__") else float(no)
        return eo, no

    elif i_type == 2:
        # UTM -> lon,lat
        easting = np.atleast_1d(xi).astype(float)
        northing = np.atleast_1d(yi).astype(float)

        if centered:
            easting = easting + 500000.0

        if i_zone is None:
            raise ValueError("When i_type==2, i_zone must be provided.")
        epsg_code = 32600 + i_zone if northern else 32700 + i_zone
        crs_utm = CRS.from_epsg(epsg_code)
        crs_latlon = CRS.from_epsg(4326)
        transformer = Transformer.from_crs(crs_utm, crs_latlon, always_xy=True)
        lon, lat = transformer.transform(easting, northing)
        # return scalar if input scalar
        if lon.size == 1:
            return float(lon[0]), float(lat[0])
        return lon, lat
    else:
        raise ValueError("i_type should be 1 or 2")

def write_inv(filename: str,
              xs: Sequence[float],
              ys: Sequence[float],
              zs: Sequence[float],
              lengths: Sequence[float],   # 原 matlab 名为 len，Python 中避开内建名
              wids: Sequence[float],
              dips: Sequence[float],
              strikes: Sequence[float],
              Us: Union[Sequence[float], float, None],
              Ud: Union[Sequence[float], float, None],
              Un: Union[Sequence[float], float, None],
              num_grid: Sequence[int]) -> None:
    """
    Python 版本的 write_inv。行为与原 MATLAB 函数等价（尽量保持原始格式与顺序）。
    filename: 输出文件名（字符串）
    xs, ys, zs, lengths, wids, dips, strikes: 一维序列 (总 patch 数)
    Us, Ud, Un: 可以传入数组，或传入 0 表示全 0(与 MATLAB 行为一致）
    num_grid: 每个 trace/段的 patch 数数组，其总和应等于 xs 的长度
    """
    # 转为 numpy 一维数组
    xs = np.asarray(xs).ravel()
    ys = np.asarray(ys).ravel()
    zs = np.asarray(zs).ravel()
    lengths = np.asarray(lengths).ravel()
    wids = np.asarray(wids).ravel()
    dips = np.asarray(dips).ravel()
    strikes = np.asarray(strikes).ravel()
    num_grid = np.asarray(num_grid).ravel().astype(int)

    n_patches = xs.size

    # 兼容 MATLAB 中 Us==0 的写法：若传入标量 0 或 None，则置为 zeros
    def _ensure_array(arr):
        if arr is None:
            return np.zeros(n_patches)
        if np.isscalar(arr):
            if arr == 0:
                return np.zeros(n_patches)
            else:
                # 如果传入单值但非 0，扩展为常数数组（保守处理）
                return np.full(n_patches, float(arr))
        a = np.asarray(arr).ravel()
        if a.size == 1:
            # 扩展单个值为整个长度（与 MATLAB 0 情形略不同，但更通用）
            return np.full(n_patches, float(a.item()))
        return a

    Us = _ensure_array(Us)
    Ud = _ensure_array(Ud)
    Un = _ensure_array(Un)

    # 检查长度一致性
    if not (ys.size == zs.size == lengths.size == wids.size == dips.size == strikes.size == Us.size == Ud.size == Un.size == n_patches):
        raise ValueError("输入数组长度不一致，请确保 xs,ys,zs,len,wid,dip,strike,Us,Ud,Un 大小相同。")

    if n_patches != num_grid.sum():
        raise ValueError(f"num_grid 的和 ({num_grid.sum()}) 应等于 xs 的长度 ({n_patches})。")

    # 构造 dpth，与 MATLAB 逻辑等价（使用 np.isclose 处理相等）
    Layer = 100
    dpth = np.empty(n_patches, dtype=int)
    dpth[0] = Layer
    for j in range(1, n_patches):
        if np.isclose(zs[j], zs[j-1]):
            dpth[j] = dpth[j-1]
        elif zs[j] > zs[j-1]:
            dpth[j] = dpth[j-1] + 1
        else:
            Layer += 100
            dpth[j] = Layer

    N = Layer // 100
    # 对每一层块做同样的变换： dpth(ii) = -(dpth(ii)-max(dpth(ii))-1)
    for jj in range(1, N + 1):
        mask = (dpth >= jj*100) & (dpth < (jj+1)*100)
        if np.any(mask):
            block = dpth[mask]
            max_block = int(block.max())
            # 保持与 MATLAB 完全等价的整数运算
            dpth[mask] = - (block - max_block - 1)

    # 写文件（保持 MATLAB fprintf 的格式）
    l = 0
    with open(filename, 'w') as fid:
        # num_grid 在 MATLAB 中是按 j=1..length(num_grid) 迭代
        for j_idx, ng in enumerate(num_grid, start=1):
            # ng 为当前 trace/段的格点数
            for k in range(int(ng)):
                l += 1
                idx = l - 1  # Python 的 0-based 索引
                # 与 MATLAB 中的格式串相匹配：
                # '%d\t%d\t%d\t%.6e\t%.6e\t%.6e\t%.6e\t%.6e\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f\t\n'
                fid.write(
                    f"{l}\t{j_idx}\t{int(dpth[idx])}\t"
                    f"{xs[idx]:.6e}\t{ys[idx]:.6e}\t{zs[idx]:.6e}\t"
                    f"{lengths[idx]:.6e}\t{wids[idx]:.6e}\t"
                    f"{dips[idx]:.2f}\t{strikes[idx]:.2f}\t"
                    f"{Us[idx]:.2f}\t{Ud[idx]:.2f}\t{Un[idx]:.2f}\t\n"
                )
# ---- 示例用法 ----
# write_inv('SSD_model.inv', xs, ys, zs, len_arr, wid_arr, dip_arr, strike_arr, Us_arr, Ud_arr, Un_arr, num_grid)
