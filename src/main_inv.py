# %%
import numpy as np
from Read_Config import parse_config
import matplotlib.pyplot as plt
from src import generate_grid, utm2ll, plot_data, plot_trace,plot_gps, plot_geological_obs,proj_gov2trace, write_inv,plot_patches
from Green_functions import generate_green_p, generate_green_g, generate_greenp_grd, generate_green_a, generate_green_o
from pathlib import Path
import os, sys
from Inversion import generate_smoothness, generate_green_ramp, check_bounds_and_columns, check_D_Green_segments,check_Greens_rmp_segments,plot_res
from scipy.optimize import lsq_linear   # fallback
import time
import scipy.io as sio
# %% ---  search path set ---
SRC_DIR = Path('/Users/junye/python/Geodetic_Inversion_Python/Geodetic-Finite-Fault-Inversion/src/').resolve()
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

# --- earthquake case  ---
earthquake = 'Ridgecrest'
base = Path('/Users/junye/python/Geodetic_Inversion_Python/Geodetic-Finite-Fault-Inversion/example')
input_path = base / earthquake / 'input'
model_path = base / earthquake / 'model'

# 切换当前工作目录到 model_path（等同于 MATLAB 的 cd）
os.chdir(model_path)

# read configuration file
config_inv = 'config_Ridgecrest.inv'
cfg = parse_config(str(model_path / config_inv))

# %% ---  Dat Container Initialization  ---

xP = np.array([]); yP = np.array([]); dP = np.array([]); sP = np.array([]); tpP = np.array([]); look = [];look = np.empty((0,3)) 
xgrd = np.array([]); ygrd = np.array([]); dGrd = np.array([]); tp_grd = np.array([]); look_grd = np.empty((0,3)); sGrd = np.array([]);
xA = np.array([]); yA = np.array([]); dA = np.array([]); sA = np.array([]); tpA = np.array([]); phi = [];
xG = np.array([]); yG = np.array([]); dG = np.array([]); sG = np.array([]); tpG = np.array([]);
xO = np.array([]); yO = np.array([]); dO = np.array([]); sO = np.array([]); uO = np.array([]); azO = np.array([]); tpO = np.array([]);
dat_ph = []; dat_az = []; dat_gps = []; dat_gov = []; dat_ph_grd = [];

# --- modify the config.inv file under model path to change the parameters ---
xo = float(cfg["model"]["origin"]["xo"])
yo = float(cfg["model"]["origin"]["yo"])

xo,yo = utm2ll(xo,yo,None,1)
Xo = float(cfg["model"]["origin"]["xo"])

# read the fault patch parameters
dw = float(cfg["inversion"]["inv_params"]["top_patch_width"])
dl = float(cfg["inversion"]["inv_params"]["top_patch_length"])
inc = float(cfg["inversion"]["inv_params"]["patch_increment_factor"])
ss = int(cfg["inversion"]["inv_params"]["strike_slip"])
ds = int(cfg["inversion"]["inv_params"]["dip_slip"])
ns = int(cfg["inversion"]["inv_params"]["normal_slip"])
fault_type = [ss, ds, ns]

# positivity constraints
PSC = int(cfg["inversion"]["inv_params"]["positivity_strike"])
PDC = int(cfg["inversion"]["inv_params"]["positivity_dip"])
PNC = int(cfg["inversion"]["inv_params"]["positivity_normal"])
PMAX = int(cfg["inversion"]["inv_params"]["positivity_max"])
BC0 = int(cfg["inversion"]["inv_params"]["bottom_zero_constraint"])

# smoothness constraints
SF = float(cfg["inversion"]["inv_params"]["smooth_factor"])
SSEG = int(cfg["inversion"]["inv_params"]["smooth_between_segments"])
SDF = int(cfg["inversion"]["inv_params"]["smooth_dip_over_strike"])
if SDF < 1:
        SF = SF/SDF;

#  zero edge constrains
BOT = int(cfg["model"]["edge_constraints"]["bot"])
SIDE = int(cfg["model"]["edge_constraints"]["side"])
TOP = int(cfg["model"]["edge_constraints"]["top"])
num_side = int(cfg["model"]["edge_constraints"]["num_side"])

SIDEID = np.vstack([
    np.array(cfg["model"]["edge_constraints"][f"side{j+1}"], dtype=int)
    for j in range(num_side)
]) if SIDE != 0 else np.empty((0,3), dtype=int)

# %% --- read the data weight for different datasets ---
PW = float(cfg["inversion"]["inv_params"]["weight_phase"])
PGW = int(cfg["inversion"]["inv_params"]["weight_ph_grd"])
AW = float(cfg["inversion"]["inv_params"]["weight_azi"])
GW = float(cfg["inversion"]["inv_params"]["weight_gps"])
OW = float(cfg["inversion"]["inv_params"]["weight_gov"])
SWP = int(cfg["inversion"]["inv_params"]["switch_phase"])

# ---  read the fault trace ---
num_src = int(cfg["model"]["model_params"]["num_of_sources"])

# --- read fault patch parameters --- 
patches = []
for i in range(1, num_src+1):
        trace = f"trace{i}"
        patch = {
            "x": float(cfg["model"][trace]["x"]),
            "y": float(cfg["model"][trace]["y"]),
            "z": float(cfg["model"][trace]["z"]),
            "len": float(cfg["model"][trace]["len"]),
            "wid": float(cfg["model"][trace]["wid"]),
            "dip": float(cfg["model"][trace]["dip"]),
            "strike": float(cfg["model"][trace]["strike"]),
        }
        if patch["dip"] == 90:
            patch["dip"] = 89.9
        patches.append(patch)
# --- load the model material parameters --- 
nu = float(cfg["model"]["model_params"]["poisson_ratio"])
# --- read the smoothness parametres between segmentselse ---
if SSEG != 0:
    SID = [np.atleast_1d(np.array(cfg["model"]["smooth"][f'smo{i+1}'], dtype=int))
           for i in range(int(cfg["model"]["smooth"]["num_seg_smooth"]))]
    SSID = [np.atleast_1d(np.array(cfg["model"]["smooth"][f'smoi{i+1}'], dtype=int))
            for i in range(int(cfg["model"]["smooth"]["num_inter_smooth"]))]
else:
    SID, SSID = [], []
print(f"SDF = {SDF},SSEG = {SSEG},SID = {SID},SSID = {SSID},TOP = {TOP},BOT = {BOT},SIDE = {SIDE},SIDEID = {SIDEID},")
# --- other parameters --- 
RMP = int(cfg["inversion"]["inv_params"]["remove_ramp"])
TP = int(cfg["inversion"]["inv_params"]["consider_topography"])

# %%--- load the data involved in inversion --- 
# --- load the descending LOS displacement --- 
num_des = int(cfg["data"]["data_params"]["num_des_sources"])
if num_des != 0:
    xx_list, yy_list, d_list, tp_list, s_list, look_list = [], [], [], [], [], []

    for j in range(num_des):
        des = f"des{j+1}"
        dat = np.loadtxt(input_path / cfg['data']["data_files"][des])

        xx, yy = utm2ll(dat[:,0]-Xo+3, dat[:,1], None, 1)
        yy = yy - yo

        xx_list.append(xx+500)
        yy_list.append(yy)
        d_list.append(dat[:,6]/10)
        tp_list.append(dat[:,2] if TP != 0 else dat[:,2]*0)
        s_list.append(dat[:,7])
        look_list.append(dat[:,3:6])

        fig, ax = plot_data(xx, yy, dat[:,6]/10,f"Descending{j} LOS, cm")
        plot_trace(patches, ax, True)
        plt.show()

        dat_ph.append(len(xx))

    xP = np.concatenate(xx_list)
    yP = np.concatenate(yy_list)
    dP = np.concatenate(d_list)
    tpP = np.concatenate(tp_list)
    sP = np.concatenate(s_list)
    look = np.vstack(look_list)

# --- ascending LOS displacement ---
num_asc = int(cfg["data"]["data_params"]["num_asc_sources"])
if num_asc != 0:
    xx_list, yy_list, d_list, tp_list, s_list, look_list = [], [], [], [], [], []

    for j in range(num_asc):
        asc = f"asc{j+1}"
        dat = np.loadtxt(input_path / cfg['data']["data_files"][asc])

        xx, yy = utm2ll(dat[:,0]-Xo+3, dat[:,1], None, 1)
        yy = yy - yo

        xx_list.append(xx+500)
        yy_list.append(yy)
        d_list.append(dat[:,6]/10)
        tp_list.append(dat[:,2] if TP != 0 else dat[:,2]*0)
        s_list.append(dat[:,7])
        look_list.append(dat[:,3:6])

        fig, ax = plot_data(xx, yy, dat[:,6]/10,f"Ascending{j} LOS, cm")
        plot_trace(patches, ax, True)
        plt.show()

        dat_ph.append(len(xx))

    xP = np.concatenate([xP, *xx_list])
    yP = np.concatenate([yP, *yy_list])
    dP = np.concatenate([dP, *d_list])
    tpP = np.concatenate([tpP, *tp_list])
    sP = np.concatenate([sP, *s_list])
    look = np.vstack([look, *look_list])

# --- phase gradient W-E ---
num_x_grd = int(cfg["data"]["data_params"]["num_x_grd_sources"])
if num_x_grd != 0:
    xx_list, yy_list, d_list, tp_list, s_list, look_list = [], [], [], [], [], []
    index = ['dec','asc']

    for j in range(num_x_grd):
        dat = np.loadtxt(input_path / cfg['data']["data_files"][f"{index[j]}_grdx"])

        xx, yy = utm2ll(dat[:,0]-Xo+3, dat[:,1], None, 1)
        yy = yy - yo

        xx_list.append(xx+500)
        yy_list.append(yy)
        d_list.append(dat[:,6])
        tp_list.append(dat[:,2] if TP != 0 else dat[:,2]*0)
        s_list.append(dat[:,7])
        look_list.append(dat[:,3:6])

        fig, ax = plot_data(xx, yy, dat[:,6],f"{index[j]}, phase gradient along W - E ")
        plot_trace(patches, ax)
        plt.show()

        dat_ph_grd.append(len(xx))

    xgrd = np.concatenate([xgrd, *xx_list])
    ygrd = np.concatenate([ygrd, *yy_list])
    dGrd = np.concatenate([dGrd, *d_list])
    tp_grd = np.concatenate([tp_grd, *tp_list])
    sGrd = np.concatenate([sGrd, *s_list])
    look_grd = np.vstack([look_grd, *look_list])


# --- phase gradient S-N ---
num_y_grd = int(cfg["data"]["data_params"]["num_y_grd_sources"])
if num_y_grd != 0:
    xx_list, yy_list, d_list, tp_list, s_list, look_list = [], [], [], [], [], []
    index = ['dec','asc']

    for j in range(num_y_grd):
        dat = np.loadtxt(input_path / cfg['data']["data_files"][f"{index[j]}_grdy"])

        xx, yy = utm2ll(dat[:,0]-Xo+3, dat[:,1], None, 1)
        yy = yy - yo

        xx_list.append(xx+500)
        yy_list.append(yy)
        d_list.append(dat[:,6])
        tp_list.append(dat[:,2] if TP != 0 else dat[:,2]*0)
        s_list.append(dat[:,7])
        look_list.append(dat[:,3:6])

        fig, ax = plot_data(xx, yy, dat[:,6],f"{index[j]}, phase gradient along S - N ")
        plot_trace(patches, ax)
        plt.show()

        dat_ph_grd.append(len(xx))

    xgrd = np.concatenate([xgrd, *xx_list])
    ygrd = np.concatenate([ygrd, *yy_list])
    dGrd = np.concatenate([dGrd, *d_list])
    tp_grd = np.concatenate([tp_grd, *tp_list])
    sGrd = np.concatenate([sGrd, *s_list])
    look_grd = np.vstack([look_grd, *look_list])

# --- load the azimuth data ---
num_azi = int(cfg["data"]["data_params"]["num_azi_sources"])

if num_azi != 0:
    xx_list, yy_list, d_list, tp_list, s_list = [], [], [], [], []
    phi_list = []

    for j in range(num_azi):
        dat = np.loadtxt(input_path / cfg['data']["data_files"][f"azi{j+1}"])

        # 坐标转换
        xx, yy = utm2ll(dat[:,0] - Xo + 3, dat[:,1], None, 1)
        yy = yy - yo

        # 保存到列表
        xx_list.append(xx+500)
        yy_list.append(yy)
        d_list.append(dat[:,2]*100)  # 注意原代码 dA = dat[:,2]*100
        tp_list.append(dat[:,2]*0 if TP != 0 else dat[:,2]*0)
        s_list.append(dat[:,3])
        phi_list.append(float(cfg["data"]["data_params"][f"phi{j+1}"]))

        # 绘图
        fig, ax = plot_data(xx, yy, dat[:,2]*100,f"MAI data, cm")
        plot_trace(patches, ax)
        plt.show()

        dat_az.append(len(xx))

    # 循环结束后一次性拼接
    xA = np.concatenate([xA, *xx_list]) 
    yA = np.concatenate([yA, *yy_list]) 
    dA = np.concatenate([dA, *d_list]) 
    tpA = np.concatenate([tpA, *tp_list]) 
    sA = np.concatenate([sA, *s_list]) 
    phi = phi + phi_list  # phi 原先是 list，用加法拼接即可
    
# --- load the GPS data (optimized, keep framework unchanged) ---
num_gps = int(cfg["data"]["data_params"]["num_gps_sources"])

# 初始化（保留原有变量名）
xG = np.array([]); yG = np.array([]); dG = np.array([]); sG = np.array([]); tpG = np.array([])

# 如果 dat_gps 不是 list（之前有时用 np.append），把它变成 list 以便高效 append
if not isinstance(dat_gps, list):
    try:
        dat_gps = list(dat_gps)
    except Exception:
        dat_gps = []

if num_gps != 0:
    # Figure 1: 第一组 GPS
    fig1, ax1 = plt.subplots(figsize=(7, 6))
    ax1.set_title("GPS observation Mainshock Mw 7.1", fontsize=14)
    ax1.set_xlabel("UTM W - E, km")
    ax1.set_ylabel("UTM S - N, km")
    ax1.grid(True)
    plot_trace(patches, ax1)


    # 预先读取 gps_h / gps_v (1D arrays)
    gps_h_arr = np.array([int(cfg["data"]["data_params"][f"gps{j+1}h"]) for j in range(num_gps)], dtype=int)
    gps_v_arr = np.array([int(cfg["data"]["data_params"][f"gps{j+1}v"]) for j in range(num_gps)], dtype=int)
    # 保持跟原来一致的形状（1 x N），便于后续使用
    gps_h = gps_h_arr.reshape(1, -1)
    gps_v = gps_v_arr.reshape(1, -1)

    # 用列表在 loop 中收集 block，循环结束后一次性 concat（更快）
    xG_blocks = []; yG_blocks = []; dG_blocks = []; sG_blocks = []; tpG_blocks = []

    for j in range(num_gps):
        gps_file = cfg["data"]["data_files"][f"gps{j+1}"]
        dat = np.loadtxt(input_path / gps_file)  # shape (npoints, ncols)

        # 坐标转换：只做两次 utm2ll（base + constant-x for wrap shift）
        npts = dat.shape[0]
        xx_base, yy_base = utm2ll(dat[:, 0] - Xo + 3, dat[:, 1], None, 1)
        # compute shift_x for wrap-around once for all rows (same constant X but different Y)
        shift_x_full, _ = utm2ll(np.full(npts, 5.999999999), dat[:, 1], None, 1)

        # apply wrap-around adjustment using vectorized masks
        mask_pos = (dat[:, 0] - Xo) > 3
        mask_neg = (dat[:, 0] - Xo) < -3

        xx = xx_base.copy()
        if mask_pos.any():
            xx[mask_pos] = xx_base[mask_pos] + 2.0 * shift_x_full[mask_pos]
        if mask_neg.any():
            xx[mask_neg] = xx_base[mask_neg] - 2.0 * shift_x_full[mask_neg]

        yy = yy_base - yo

        # 当前组的水平/垂直开关（取出 scalar）
        gps_h_j = int(gps_h_arr[j])
        gps_v_j = int(gps_v_arr[j])

        # 选择属于哪个组 (保持原始逻辑：j==0 -> group1, else group2)
        if j == 0:
            xb_list = xG_blocks; yb_list = yG_blocks; db_list = dG_blocks; sb_list = sG_blocks; tpb_list = tpG_blocks
            ax = ax1
            dat_gps.append(len(xx))

        # 拼接水平分量（注意：MATLAB/Python 原代码是 concat([xx, xx]) -> blockwise duplication）
        if gps_h_j != 0:
            xb_list.append(np.concatenate([xx, xx]))   # first all xx then again all xx
            yb_list.append(np.concatenate([yy, yy]))
            db_list.append(np.concatenate([dat[:, 2] / 10.0, dat[:, 3] / 10.0]))
            sb_list.append(np.concatenate([dat[:, 5] / 10.0, dat[:, 6] / 10.0]))
            if TP != 0:
                tpb_list.append(np.concatenate([dat[:, 8], dat[:, 8]]))
            else:
                tpb_list.append(np.concatenate([np.zeros_like(dat[:, 8]), np.zeros_like(dat[:, 8])]))

        # 拼接垂直分量
        if gps_v_j != 0:
            xb_list.append(xx.copy())
            yb_list.append(yy.copy())
            db_list.append(dat[:, 4] / 10.0)
            sb_list.append(dat[:, 7] / 10.0)
            if TP != 0:
                tpb_list.append(dat[:, 8].copy())
            else:
                tpb_list.append(np.zeros_like(dat[:, 8]))

        # 绘图（保持原来 per-file 绘图行为）
        plot_gps(ax, xx, yy, dat)

    # 循环结束后，把 blocks concat 回原来的 np.array 变量（保持名字不变）
    if len(xG_blocks) > 0:
        xG = np.concatenate(xG_blocks)
        yG = np.concatenate(yG_blocks)
        dG = np.concatenate(dG_blocks)
        sG = np.concatenate(sG_blocks)
        tpG = np.concatenate(tpG_blocks)
    else:
        xG = np.array([]); yG = np.array([]); dG = np.array([]); sG = np.array([]); tpG = np.array([])

    plt.show()

    # gps_type 保持原来形状 (1, N)
    gps_type = np.vstack((gps_h.reshape(1, -1), gps_v.reshape(1, -1)))

# --- load the Geological observation or optical offset data ---
num_gov = int(cfg["data"]["data_params"]["num_gov_sources"])
if num_gov != 0:
    fig, ax = plt.subplots(figsize=(7, 6))

    xx_list, yy_list, d_list, tp_list, s_list, az_list = [], [], [], [], [], []

    for j in range(num_gov):
        dat = np.loadtxt(input_path / cfg['data']["data_files"][f"gov{j+1}"])

        xx, yy = utm2ll(dat[:, 0] - Xo + 3, dat[:, 1], None, 1)
        yy = yy - yo

        # 保存到列表
        xx_list.append(xx+500)
        yy_list.append(yy)
        d_list.append(dat[:, 2]*100)  # 注意 dO 原来是 dat[:,2]*100
        tp_list.append(dat[:, 5] if TP != 0 else dat[:, 5]*0)
        s_list.append(dat[:, 3])
        az_list.append(dat[:, 4])

    # 循环结束后一次性拼接
    xO = np.concatenate([xO, *xx_list]) 
    yO = np.concatenate([yO, *yy_list])
    dO = np.concatenate([dO, *d_list]) 
    tpO = np.concatenate([tpO, *tp_list]) 
    sO = np.concatenate([sO, *s_list]) 
    azO = np.concatenate([azO, *az_list]) 

    # 绘制矢量图
    plot_geological_obs(xO, yO, dO, azO, scale_ref=100.0, ax=ax)
    plot_trace(patches, ax)
    plt.show()

#%% --- generate the fault model for inversion --- 
XS, YS, ZS = [], [], []
XB, YB, ZB = [], [], []
LL, WW, DIP, STRIKE = [], [], [], []
num_grid = []

fig = plt.figure(figsize=(10, 6), constrained_layout=True)
ax = fig.add_subplot(111, projection="3d")

for j in range(num_src):
    xs, ys, zs, xb, yb, zb, ll, ww, nw, _, _, _, _ = generate_grid(
        1, patches[j], dw, dl, inc, plt_flag=1, ax=ax
    )
    XS.extend(xs)
    YS.extend(ys)
    ZS.extend(zs)
    XB.extend(xb)
    YB.extend(yb)
    ZB.extend(zb)
    LL.extend(ll)
    WW.extend(ww)
    DIP.extend(np.zeros_like(xs) + patches[j]["dip"])
    STRIKE.extend(np.zeros_like(xs) + patches[j]["strike"])
    num_grid.append(len(xs))

# 转成 numpy 数组
XS = np.array(XS); YS = np.array(YS); ZS = np.array(ZS)
XB = np.array(XB); YB = np.array(YB); ZB = np.array(ZB)
LL = np.array(LL); WW = np.array(WW); DIP = np.array(DIP); STRIKE = np.array(STRIKE)
num_grid = np.array(num_grid)

plt.show()

# --- Generate the green's function --- #

GreenP = np.empty((0,0))
GreenA = np.empty((0,0))
GreenG = np.empty((0,0))
GreenO = np.empty((0,0))
GreenP_grd = np.empty((0,0))
if num_asc + num_des != 0:
    GreenP = generate_green_p(XS, YS, ZS, LL, WW, DIP, STRIKE,
                              xP, yP, tpP, look, nu, fault_type)

if num_x_grd + num_y_grd != 0:
    GreenP_grd = generate_greenp_grd(XS, YS, ZS, LL, WW, DIP, STRIKE,
                                     xgrd, ygrd, tp_grd, look_grd,
                                     nu, fault_type, dat_ph_grd, 50, 50)

if num_azi != 0:
    # print("XS's shape:",XS.shape,"\n",
    #   "phi's shape:",len(phi),"phi = ",dat_az,"\n",
    #   "fault_type's shape:",len(fault_type),"fault_type = ",fault_type,"\n",
    #   "dat_az's shape:",len(dat_az),"dat_az = ",dat_az,"\n",
    #   "xA's shape:",xA.shape)
    GreenA = generate_green_a(XS, YS, ZS, LL, WW, DIP, STRIKE,
                              xA, yA, tpA, phi, nu, fault_type, dat_az)

if num_gps != 0:
    GreenG = generate_green_g(XS, YS, ZS, LL, WW, DIP, STRIKE,
                              xG, yG, tpG, nu, fault_type,
                              gps_type, dat_gps)

xOO, yOO, dOO, azOO, tpOO, sOO = [], [], [], [], [], []
if num_gov != 0:
    xOO, yOO, dOO, azOO, tpOO, sOO, dat_gov = proj_gov2trace(
        patches, xO, yO, dO, azO, tpO, sO, dat_gov, 150, 500
    )
    GreenO = generate_green_o(XS, YS, ZS, LL, WW, DIP, STRIKE,
                              xOO, yOO, azOO, tpOO, nu, fault_type)
# 组合 Green's function 矩阵
print("GreenP's shape:",GreenP.shape,"\n",
      "GreenP_grd's shape:",GreenP_grd.shape,"\n",
      "GreenG's shape:",GreenG.shape,"\n",
      "GreenO's shape:",GreenO.shape,"\n",
      "GreenA's shape:",GreenA.shape)

# %%
blocks = []
if GreenP.size > 0:
    blocks.append(GreenP)
if GreenP_grd.size > 0:
    blocks.append(GreenP_grd)
if GreenA.size > 0:
    blocks.append(GreenA)
if GreenG.size > 0:
    blocks.append(GreenG)
if GreenO.size > 0:
    blocks.append(GreenO)

if not blocks:
    Greens = np.zeros((0, 0))
else:
    # 假设你希望在垂直方向拼接（行拼接）
    Greens = np.vstack(blocks)
print("Greens shape:", Greens.shape)

# %% create the smoothness matrix
Smooth = np.empty((0, 0))   # 初始化为空矩阵

if SF != 0:
    # 调用 generate_smoothness
    Smooth = generate_smoothness(
        XS, YS, ZS, LL, WW, STRIKE,
        num_grid, fault_type,
        SDF, SSEG, SID, SSID,
        TOP, BOT, SIDE, SIDEID,1)


#  add ramp to the dataset
# print("xP.shape:", xP.shape,"\n",
#       "yP.shape:", yP.shape,"\n",
#       "xA.shape:", xA.shape,"\n",
#       "xgrd.shape:", xgrd.shape,"\n",
#       "ygrd.shape:", ygrd.shape,"\n",
#       "dat_ph:", dat_ph,"\n",
#       "dat_az:", dat_az,"\n",
#       "dat_gps:", 2*dat_gps,"\n",)

rmp = np.empty((0, 0))  # 初始化为空矩阵
if RMP == 1:
    rmp = generate_green_ramp(
        xP, yP, xA, yA, xgrd, ygrd,
        dat_ph, dat_ph_grd, dat_az,
        dat_gps, dat_gov, 100
    )
    # % the last num should be typical value of rmp/ (ratio of cond number 
    # * typical value of GreenP), in order not to weigh the rmp too
    # much

# print("Smooth's shape:",Smooth.shape,"\n",
#       "rmp's shape:",rmp.shape)
# %% Prepare for the inversion matrix

# 初始化空数组
D = np.empty((0, 1), dtype=float)
W = np.empty((0, 1), dtype=float)

# Phase InSAR (descending + ascending)
if num_des + num_asc != 0 and PW != 0:
    nP = np.mean(sP)
    sP = sP / np.mean(sP)
    sP = sP / PW * len(sP)
    D = np.vstack([D, SWP * dP.reshape(-1,1)])   # 注意 dP reshape
    W = np.vstack([W, 1.0 / sP.reshape(-1,1)])
elif num_des + num_asc != 0 and PW == 0:
    D = np.vstack([D, SWP * dP.reshape(-1,1)])
    W = np.vstack([W, np.zeros_like(sP).reshape(-1,1)])

# Gridded InSAR
if num_x_grd + num_y_grd != 0 and PGW != 0:
    nGrd = np.mean(sGrd)
    sGrd = sGrd / np.mean(sGrd)
    sGrd = sGrd / PGW * len(sGrd)
    D = np.vstack([D, SWP * dGrd.reshape(-1,1)])
    W = np.vstack([W, 1.0 / sGrd.reshape(-1,1)])
elif num_x_grd + num_y_grd != 0 and PGW == 0:
    D = np.vstack([D, SWP * dGrd.reshape(-1,1)])
    W = np.vstack([W, np.zeros_like(sGrd).reshape(-1,1)])

# Azimuth offset
if num_azi != 0 and AW != 0:
    nA = np.mean(sA)
    sA = sA / np.mean(sA)
    sA = sA / AW * len(sA)
    D = np.vstack([D, dA.reshape(-1,1)])
    W = np.vstack([W, 1.0 / sA.reshape(-1,1)])
elif num_azi != 0 and AW == 0:
    D = np.vstack([D, dA.reshape(-1,1)])
    W = np.vstack([W, np.zeros_like(sA).reshape(-1,1)])

# GPS
if num_gps != 0 and GW != 0:
    sG = sG / np.mean(sG)
    sG = sG / GW * len(sG)
    D = np.vstack([D, dG.reshape(-1,1)])
    W = np.vstack([W, 1.0 / sG.reshape(-1,1)])
elif num_gps != 0 and GW == 0:
    D = np.vstack([D, dG.reshape(-1,1)])
    W = np.vstack([W, np.zeros_like(sG).reshape(-1,1)])

# Geological observations
if num_gov != 0 and OW != 0:
    nO = np.mean(sO)
    sOO = sOO / np.mean(sOO)
    sOO = sOO / OW * len(sOO)
    D = np.vstack([D, dOO.reshape(-1,1)])
    W = np.vstack([W, 1.0 / sOO.reshape(-1,1)])
elif num_gov != 0 and OW == 0:
    D = np.vstack([D, dOO.reshape(-1,1)])
    W = np.vstack([W, np.zeros_like(sOO).reshape(-1,1)])


# ---------- 修正版：计算平滑矩阵归一化 ----------
if np.all(Smooth == 0):
    raise ValueError("Smooth matrix is all zeros, cannot normalize.")

# row-wise max
smooth_rowmax = np.max(Smooth, axis=1)
mean_rowmax = np.mean(smooth_rowmax)  # 无需跳过零行
Smooth_norm = Smooth / mean_rowmax * (SF / Smooth.shape[0])
Smooth_norm = np.nan_to_num(Smooth_norm)  # 把 nan -> 0

# 确认维度
print("✅ Smooth_norm shape:", Smooth_norm.shape)

# ---------- 2) 用 hstack/vstack 构造 Green_all（你已有 block_upper/ block_lower，保留） ----------
block_upper = np.hstack([Greens, rmp])

print("block_upper shape: ", block_upper.shape)

block_lower = np.hstack([Smooth_norm, np.zeros((Smooth.shape[0], rmp.shape[1]))])
print("block_lower shape: ", block_lower.shape)
# 预分配并合并（vstack 本身就会复制一次，这里保留，但后面尽量避免重复复制）
Green_all = np.vstack([block_upper, block_lower])

# ---------- 3) 使用向量化权重（替换掉逐行 Python 循环） ----------
# 原逻辑只把前 len(W) 行乘权重，后面追加的平滑行不乘权重（你原来是这样）
n_rows = Green_all.shape[0]
w_vec = np.ones(n_rows, dtype=np.float64)
w_vec[:len(W)] = W.ravel()   # 如果 W 是 (n,1) 列向量，这样取到一维

# 确保 D_all 是一维并且被正确权重（你之前有 D_all = W * D）
# 这里重构 D_all：先把测量部分写好，再 pad
d_meas = (W.ravel() * D.ravel())  # shape (n_obs,)
if Green_all.shape[0] > d_meas.size:
    D_all = np.zeros(Green_all.shape[0], dtype=d_meas.dtype)
    D_all[:d_meas.size] = d_meas
else:
    D_all = d_meas[:Green_all.shape[0]]

# 应用权重到 Green_all 的前 len(W) 行（矢量化）
# 先确保连续和浮点类型，利于 BLAS 调用效率
Green_all = np.ascontiguousarray(Green_all, dtype=np.float64)
w_col = w_vec[:, None]          # shape (n_rows,1)
Green_all *= w_col              # 广播：每一行乘对应标量

print("D_all shape: ", D_all.shape)

# %%
# ---------- 4) 构造 bounds（保持你原有逻辑） ----------
lb = -PMAX * np.ones(Green_all.shape[1], dtype=np.float64)
ub =  PMAX * np.ones(Green_all.shape[1], dtype=np.float64)
# PSC / PDC 等约束不变（保持你原有 slice 操作）
fault_sum = int(np.sum(fault_type))
rmp_cols = rmp.shape[1] if rmp is not None else 0
if PSC > 0:
    lb[0:Green_all.shape[1] - rmp_cols:fault_sum] = 0
elif PSC < 0:
    ub[0:Green_all.shape[1] - rmp_cols:fault_sum] = 0
if PDC > 0:
    lb[1:Green_all.shape[1] - rmp_cols:fault_sum] = 0
elif PDC < 0:
    ub[1:Green_all.shape[1] - rmp_cols:fault_sum] = 0

# ---------- 5) 列尺度归一化（强烈推荐：改善条件数，常显著加速迭代） ----------
col_norms = np.linalg.norm(Green_all, axis=0)
# 防止 0 除法
col_norms_safe = col_norms.copy()
col_norms_safe[col_norms_safe == 0] = 1.0

G_scaled = Green_all / col_norms_safe  # 广播除法
lb_scaled = lb * col_norms_safe
ub_scaled = ub * col_norms_safe

dat_sizes = [np.sum(dat_ph), np.sum(dat_ph_grd), np.sum(dat_az), np.sum(dat_gps), np.sum(dat_gov)]
dataset_names = ["phase", "phase_grad", "azimuth", "gps", "gov"]

# check_bounds_and_columns(Greens, rmp, lb, ub, dat_sizes, dataset_names)

# check_D_Green_segments(D_all, Greens, rmp, dat_ph, dat_ph_grd, dat_az, dat_gps, dat_gov)

# ---------- 6) 调用求解器（对缩放后的矩阵求解，求得 x_scaled，再反向缩放） ----------
t0 = time.perf_counter()
res = lsq_linear(G_scaled, D_all.ravel(), bounds=(lb_scaled, ub_scaled),
                 lsmr_tol='auto', max_iter = int(1e9))
t1 = time.perf_counter()
print("lsq_linear time (scaled):", t1 - t0)

x_scaled = res.x
U = x_scaled / col_norms_safe   # 反缩放得到真实解
resnorm = np.sum(res.fun**2)
residual = res.fun
exitflag = res.status

# ---------- 7) RMP 分支：保留你原有结构，但也应用相同的优化策略 ----------
if RMP == 1:
    rmp0 = rmp.copy()
    Urmp0 = U[-rmp0.shape[1]:]
    # 重建 D_all 为一维数组（权重与 pad）
    D_meas2 = (W.ravel() * (D.ravel() - (rmp0 @ Urmp0).ravel()))
    D_all = np.zeros(Green_all.shape[0], dtype=D_meas2.dtype)
    D_all[:D_meas2.size] = D_meas2

    # regenerate ramp (你原来的函数调用)
    rmp = generate_green_ramp(xP, yP, xA, yA, xgrd, ygrd,
                              dat_ph, dat_ph_grd, dat_az,
                              dat_gps, dat_gov, 1e6)

    # 重新拼接 Greens & Smooth / norm_factor （与前面相同的合并逻辑）
    Greens_rows, Greens_cols = Greens.shape
    Smooth_rows, Smooth_cols = Smooth.shape
    # reuse mean_rowmax 和 mean_rowmax * ... 的逻辑
    norm_factor = mean_rowmax * Smooth.shape[0] / SF
    block_upper = np.hstack([Greens, rmp])
    block_lower = np.hstack([Smooth / norm_factor, np.zeros((Smooth_rows, rmp.shape[1]))])
    Green_all = np.vstack([block_upper, block_lower])

    # 确保连续与浮点，再 vectorized 权重
    Green_all = np.ascontiguousarray(Green_all, dtype=np.float64)
    w_vec2 = np.ones(Green_all.shape[0], dtype=np.float64)
    w_vec2[:len(W)] = W.ravel()
    Green_all *= w_vec2[:, None]

    # 列归一化、缩放 bounds（与上面相同）
    col_norms = np.linalg.norm(Green_all, axis=0)
    col_norms_safe = col_norms.copy()
    col_norms_safe[col_norms_safe == 0] = 1.0
    G_scaled = Green_all / col_norms_safe
    lb_scaled = lb * col_norms_safe
    ub_scaled = ub * col_norms_safe

    res = lsq_linear(G_scaled, D_all.ravel(), bounds=(lb_scaled, ub_scaled),
                     lsmr_tol='auto', max_iter= int(1e9))
    x_scaled = res.x
    U = x_scaled / col_norms_safe
    resnorm = np.sum(res.fun**2)
    residual = res.fun
    exitflag = res.status

else:
    rmp0 = 0
    Urmp0 = 0

# ---------- 8) 提取 Us, Ud, Urmp （保持原逻辑） ----------
Urmp = U[-rmp.shape[1]:] if rmp is not None else np.array([])
Us = U[0:-rmp.shape[1]:2] if rmp is not None else U[0::2]
Ud = U[1:-rmp.shape[1]:2] if rmp is not None else U[1::2]
print("W's shape:",W.shape,"\n",
      "Green_all's shape:",Green_all.shape,"\n",
      "D_all's shape:",D_all.shape)
# Save fault slip files
write_inv('test.inv',XS,YS,ZS,LL,WW,DIP,STRIKE,Us,Ud,0,num_grid);

plot_patches('test.inv',13,0.002,"Ridgecrest Mw7.1")


print("Greens.shape:", Greens.shape)
print("rmp.shape:", rmp.shape)
print("U.shape:", U.shape)
print("D.shape:", D.shape)
print("rmp0.shape:", rmp0.shape if rmp0 is not None else None)
print("Urmp0.shape:", Urmp0.shape if Urmp0 is not None else None)

# check_Greens_rmp_segments(Greens, rmp, D_all, dat_ph, dat_ph_grd, dat_az, dat_gps, dat_gov)
# %% --- compute residuals ---
# 保证向量为 1D
D_vec = D.flatten()

# 预测值与残差
pred = np.hstack([Greens, rmp]) @ U
residual = pred - (D_vec - rmp0 @ Urmp0)

# 平方误差
ms = residual ** 2
ms0 = D_vec ** 2

# 各数据段的累积边界
num_dat = np.array([
    0,
    np.sum(dat_ph),
    np.sum(dat_ph_grd),
    np.sum(dat_az),
    np.sum(dat_gps),
    np.sum(dat_gov)
], dtype=int)

# 初始化存储每段结果
ms_i = np.zeros(len(num_dat) - 1)
ms0_i = np.zeros(len(num_dat) - 1)

# --- 按段计算每种数据的拟合误差 ---
for j in range(len(num_dat) - 1):
    start = np.sum(num_dat[:j+1])
    end   = np.sum(num_dat[:j+2])

    if num_dat[j+1] != 0:
        ms_i[j]  = np.sum(ms[start:end])
        ms0_i[j] = np.sum(ms0[start:end])
    else:
        ms_i[j]  = -1
        ms0_i[j] = -1

# --- 计算每段拟合改进百分比 ---
pct = np.zeros_like(ms_i)
valid_mask = ms0_i > 0
pct[valid_mask] = 100 * (ms0_i[valid_mask] - ms_i[valid_mask]) / ms0_i[valid_mask]
pct[~valid_mask] = 0

# --- 打印与 MATLAB 完全一致的输出格式 ---
print(f"rms misfit (dat., res.) and reduction percentage = "
      f"{np.sum(ms0):.6e} {np.sum(ms):.6e} "
      f"({round(100*(np.sum(ms0)-np.sum(ms))/np.sum(ms0))}%)")

print(f"reduction percentage for each dataset = "
      f"{pct[0]:.1f}% {round(pct[1])}% {round(pct[2])}% "
      f"{round(pct[3])}% {round(pct[4])}%, "
      f"only {500 - round(np.sum(pct))}% to go!")

print(f"exitflag = {exitflag}")

# --- 刚度模量 mu ---
mu = 30e9  # 若不使用分层，可直接用常数

# （可选）根据深度分层刚度
# dp_list = ZS    # 深度中心（单位 m），如果有 WW, DIP 可改为: dp_list = ZS - WW * np.sin(np.deg2rad(DIP))
# mu_list = np.zeros_like(dp_list)
# mu_list[dp_list > -5e3] = 21.4e9
# mu_list[(dp_list <= -5e3) & (dp_list > -20e3)] = 36.3e9
# mu_list[(dp_list <= -20e3) & (dp_list > -35e3)] = 43.2e9
# mu_list[(dp_list <= -35e3) & (dp_list > -45e3)] = 47.5e9
# mu_list[(dp_list <= -45e3) & (dp_list > -55e3)] = 68.3e9
# mu_list[(dp_list <= -55e3) & (dp_list > -90e3)] = 68.3e9
# mu_list[dp_list < -90e3] = 75.1e9

# --- 计算地震矩 ---
# Us: strike-slip 方向滑动
# Ud: dip-slip 方向滑动
# LL: 断层走向长度（单位 m）
# WW: 断层倾向宽度（单位 m）
# ub: 上下界约束（用于符号判断）

# 获取滑动方向符号（MATLAB 的 sign(ub(1:2:end)) - 0.5)*2
sn = (np.sign(ub[0:len(Us)*2:2]) - 0.5) * 2  # shape 与 Us 对应

# 计算地震矩 M0
# 1/100 因为 Us 和 Ud 以 cm 为单位，需要转为 m
M0 = mu * np.sqrt(
    np.sum((Us * sn / 100.0 * LL * WW) ** 2) +
    np.sum((Ud / 100.0 * LL * WW) ** 2)
)

# 计算震级 Mw
m = 2.0 / 3.0 * np.log10(M0) - 6.07

print(f"Seismic moment magnitude: Mw = {m:.3f}")
plot_res(
    Greens, rmp, U, D, Urmp0, rmp0,
    dat_ph, dat_ph_grd, dat_az, dat_gps, dat_gov,
    xP, yP, xgrd, ygrd, xA, yA, xG, yG, xO, yO,
    0, 11, xo, yo,patches)
plt.show()