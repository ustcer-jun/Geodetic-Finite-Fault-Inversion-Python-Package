from src import plot_patches, rednoise,plot_trace, utm2ll
from Green_functions import generate_green_g,calc_okada
from multiprocessing import Pool, cpu_count
import psutil,os,shutil,tempfile
from scipy.interpolate import RectBivariateSpline
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
from Read_Config import  parse_config
from scipy.interpolate import RectBivariateSpline, RegularGridInterpolator
import pygmt as pg
from pygmt.params import Box

def wrap_to_pi(LOS_dis, wavelength=0.056):
    """
    Convert line-of-sight displacement (cm) to wrapped radar phase (radians)
    and wrap to [-pi, pi].
    
    Parameters
    ----------
    LOS_dis : array_like
        LOS displacement in cm
    wavelength : float, optional
        Radar wavelength in meters, default 0.056 m
    
    Returns
    -------
        Wrapped phase in radians, range [-pi, pi]
    """
    # cm -> m
    LOS_m = np.array(LOS_dis) * 0.01
    # convert to phase: 4*pi/lambda * LOS
    phase = LOS_m * (4 * np.pi / wavelength)
    # wrap to [-pi, pi]
    return (phase + np.pi) % (2 * np.pi) - np.pi

def plot_data(ax, X, Y, Data, title="Data", clim=None, cbar_label="Displacement (cm)"):
    """
    plot the 2-D data 
    X, Y : grid coordinates
    Data : data 
    title : default "Data"
    """
    im = ax.imshow(Data, cmap="jet", origin="lower",
                   extent=[X.min()/1000, X.max()/1000,
                           Y.min()/1000, Y.max()/1000])
    if clim is not None:
        im.set_clim(clim[0], clim[1])

    # 调整 colorbar 大小与间距，避免子图重叠
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(cbar_label)

    ax.set_title(title, fontsize=12)
    ax.set_xlabel("UTM W - E (km)")
    ax.set_ylabel("UTM S - N (km)")
    return im


def Save2grd(filename,X,Y,Data,units,Annot,X_name = "lon",Y_name = "lon",data_name = "displacement"):
   
    """
    Save 2D grid data to netCDF file compatible with GMT/GMTSAR.
    X, Y: 1D coordinates
    Data: 2D array (len(Y) x len(X))
    units: variable units string
    Annot: description string
    filename: output netCDF filename
    """
    ny, nx = Data.shape
    if X.size != nx or Y.size != ny:
        raise ValueError(f"坐标与数据尺寸不匹配: got X.size={X.size}, Y.size={Y.size}, Data.shape={Data.shape}")
    
    ds = xr.Dataset(
    {
        data_name : (["y", "x"], Data)
    },
    coords={
        "x": X,
        "y": Y
    }
    )
    # 添加一些元数据（可选）
    ds[data_name].attrs["units"] = units
    ds.x.attrs["name"] = X_name;
    ds.y.attrs["name"] = Y_name;
    ds.attrs["description"] = Annot;

    # 写入 netCDF 文件
    ds.to_netcdf(filename);
    print(f"Saved `{filename}` (shape={Data.shape}, x={X.size}, y={Y.size})")

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

def forward_point(args):
    """
    Calculate single grid coordinate 
    """
    i, xi, yi, xs, ys, zs, L, W, DIP, STRIKE, slips, nu, fault_types, gps_type = args
    tpi = 0.0
    green = generate_green_g(xs, ys, zs, L, W, DIP, STRIKE,
                             xi, yi, tpi, nu, fault_types, gps_type, 1)
    return i, green @ slips


def forward_all_parallel(xs, ys, zs, L, W, DIP, STRIKE,
                         xG, yG, slips, nu, fault_types, gps_type, n_process=None):
    """
    parallel Calculate Sensitivity Matrix
    """
    if n_process is None:
        n_process = max(1, cpu_count() - 2)  # Multiprocessing 

    N = xG.size
    dd = np.zeros(3 * N)

    # 构建参数列表，每个元素对应一个网格点
    args_list = [(i, float(xG[i]), float(yG[i]), xs, ys, zs, L, W, DIP, STRIKE, slips, nu, fault_types, gps_type)
                 for i in range(N)]

    print(f"Starting parallel computation with {n_process} processes ...")
    with Pool(processes=n_process) as pool:
        for i, result in pool.map(forward_point, args_list):
            dd[i*3:(i+1)*3] = result

    return dd


def forward_3D_dis(file_name, show_plot ,dx,dy,xlim,ylim):
    if show_plot == 1:
        plot_patches(file_name,13,5e-3,"Synthetic Slip model")
    """
    Parallel calculation
    1、 load the fault geometry and slip parameter file
    2、 forward the 3-D displacment field and proj to satellite LOS direction
    3、 Save to netcdf file and plot file
    """
    ## loading parameters
    fault_paras =np.loadtxt(file_name);
    if fault_paras.ndim == 1:
        fault_paras = fault_paras[np.newaxis, :]

    xs = fault_paras[:, 3]; ys = fault_paras[:, 4]; zs = fault_paras[:, 5];
    L = fault_paras[:, 6]; W = fault_paras[:, 7];
    dip = np.deg2rad(fault_paras[:, 8]); strike = np.deg2rad(fault_paras[:, 9])
    DIP = fault_paras[:, 8];STRIKE = fault_paras[:, 9];
    Us = fault_paras[:, 10].ravel()
    Ud = fault_paras[:, 11].ravel()
    Un = fault_paras[:, 12].ravel()

    ## meshing surface grid
    UTM_X = np.arange(-xlim,xlim+1,dx);UTM_Y = np.arange(-ylim,ylim+1,dy);
    UTM_X_grid, UTM_Y_grid = np.meshgrid(UTM_X,UTM_Y);
    ## flatten grid
    xG = UTM_X_grid.ravel()   # 1D array, length N
    yG = UTM_Y_grid.ravel()
    N = xG.size
    nu = 0.25;
    fault_types = np.array([1,1,0]); gps_type = np.array([[1],[1]]);
    slips = np.zeros(2*Us.size);slips[0::2] = Us;slips[1::2] = Ud;

    # ===== Parallel Calculation =====
    print(f"Computing displacement for {N} grid points ...")
    dd = forward_all_parallel(xs, ys, zs, L, W, DIP, STRIKE,
                          xG, yG, slips, nu, fault_types, gps_type)
    nx = UTM_X.size; ny = UTM_Y.size
    De = dd[0::3]; Dn = dd[1::3]; Dz = dd[2::3];
    de = De.reshape(ny, nx)
    dn = Dn.reshape(ny, nx)
    dz = Dz.reshape(ny, nx)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)
    plot_data(axes[0], UTM_X, UTM_Y, de, "West - East Component, cm",clim=(-50, 50))
    plot_data(axes[1], UTM_X, UTM_Y, dn, "South - North Component, cm",clim=(-50, 50))
    plot_data(axes[2], UTM_X, UTM_Y, dz, "Vertical Component, cm",clim=(-50, 50))
    fig.suptitle("Forward 3-D Displacement Field", fontsize=16, weight="bold")
    plt.show()

    ### Project into LOS direction
    track=np.array([190,-10]);
    look= np.array([30,30]);

    # project vector
    proj_vec1 = np.array([
        np.sin(np.deg2rad(look[0])) * np.cos(np.deg2rad(track[0])),
        np.sin(np.deg2rad(look[0])) * np.sin(np.deg2rad(track[0])),
        np.cos(np.deg2rad(look[0]))
    ])
    proj_vec2 = np.array([
        np.sin(np.deg2rad(look[1])) * np.cos(np.deg2rad(track[1])),
        np.sin(np.deg2rad(look[1])) * np.sin(np.deg2rad(track[1])),
        np.cos(np.deg2rad(look[1]))
    ])

    # 将 de, dn, dv 展平并组合成 N x 3 矩阵
    data_matrix = np.column_stack((De, Dn, Dz));

    # Project into Satellite LOS direcetion
    data_dec = (data_matrix @ proj_vec1).reshape(ny,nx)  # N x 1
    data_asc = (data_matrix @ proj_vec2).reshape(ny,nx)  # N x 1
    Save2grd("data_dec.grd", UTM_X, UTM_Y, data_dec, "cm", "Simulated displacement field")
    Save2grd("data_asc.grd", UTM_X, UTM_Y, data_asc, "cm", "Simulated displacement field")
    # data_dec_noise = data_dec + noise;
    # data_asc_noise = data_asc + noise;
    phase_dec = wrap_to_pi(data_dec);
    phase_asc = wrap_to_pi(data_asc);
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), constrained_layout=True)
    plot_data(axes[0,0], UTM_X, UTM_Y, data_asc, "Ascending LOS, cm",clim=(-50, 50))
    plot_data(axes[0,1], UTM_X, UTM_Y, data_dec, "Descending LOS, cm",clim=(-50, 50))
    plot_data(axes[1,0], UTM_X, UTM_Y, phase_asc, "Ascending Wrapped Phase, rad", clim=None, cbar_label="rad")
    plot_data(axes[1,1], UTM_X, UTM_Y, phase_dec, "Descending Wrapped Phase, rad", clim=None, cbar_label="rad")
    fig.suptitle("Forward LOS / Phase Field", fontsize=16, weight="bold")
    plt.show()

def slip2gps_okada_vectorized(xmin, xmax, ymin, ymax, inc, slip_model_in, nu=0.25):
    """
    Vectorized Okada forward: compute displacement on regular grid using calc_okada vectorized over points.

    slip_model_in columns expected (per your reference/typical format):
    [fault, patch, layer, x, y, z, length, width, strike, dip, coeff, strike_slip, dip_slip]

    Returns:
      ue, un, uz : 2D arrays (same shape as meshgrid)
      x_grid_km, y_grid_km : 2D arrays of coordinates in km
    """
    d2r = np.pi/180.0

    # parse patch parameters
    xp = slip_model_in[:, 3].astype(float)   # patch center x (m)
    yp = slip_model_in[:, 4].astype(float)
    zp = slip_model_in[:, 5].astype(float)
    lp = slip_model_in[:, 6].astype(float)
    wp = slip_model_in[:, 7].astype(float)
    strkp = slip_model_in[:, 8].astype(float)
    dip0 = slip_model_in[:, 9].astype(float)
    s_strike = slip_model_in[:, 11].astype(float)  # strike-slip amount
    s_dip = slip_model_in[:, 12].astype(float)     # dip-slip amount

    Npatch = xp.size

    # build grid (in meters)
    x = np.arange(xmin, xmax + inc, inc, dtype=float)
    y = np.arange(ymin, ymax + inc, inc, dtype=float)
    x_grid, y_grid = np.meshgrid(x, y)   # shapes (ny, nx)
    xgps = x_grid.ravel()
    ygps = y_grid.ravel()
    npts = xgps.size

    # accumulators (flattened)
    ue_tot = np.zeros(npts, dtype=float)
    un_tot = np.zeros(npts, dtype=float)
    uz_tot = np.zeros(npts, dtype=float)

    HF = 1  # as in your reference

    # compute patch centers offset along strike by half-length (same as your ref)
    dxf = lp / 2.0
    # theta = (90 - strike) in radians for xy2XY used in your ref
    theta = (90.0 - strkp) * d2r

    # helper: rotate offsets (reuse your existing xy2XY if present, else implement small routine)
    try:
        from src import xy2XY  # if your project provides xy2XY
        dx_all, dy_all = xy2XY(dxf, np.zeros_like(dxf), -theta)
    except Exception:
        # simple rotation: rotate (dxf,0) by angle -theta
        dx_all = dxf * np.cos(-theta)  # but original xy2XY may do different convention; keep fallback simple
        dy_all = dxf * np.sin(-theta)

    xcenters = xp + dx_all
    ycenters = yp + dy_all
    zcenters = zp  # unchanged

    # try vectorized calc_okada call inside loop over patches (calc_okada should accept array xpt,ypt)
    # If calc_okada supports vectorized xpt,ypt, each call returns arrays of length npts -> fast
    # If not, we'll fallback to chunked/loop computation.
    vector_ok = True
    for k in range(Npatch):
        # compute arrays relative to patch center
        xpt = xgps - xcenters[k]
        ypt = ygps - ycenters[k]
        U1 = s_strike[k]
        U2 = s_dip[k]
        delta = dip0[k] * d2r
        d = -zcenters[k]   # as in your ref (depth sign)
        len_val = lp[k]
        W = wp[k]
        strike_rad = strkp[k] * d2r
        tp = np.zeros_like(xpt)

        # attempt vectorized evaluation
        try:
            ue1, un1, uz1 = calc_okada(HF, U1, xpt, ypt, nu, delta, d, len_val, W,
                                       1, strike_rad, tp)
            ue2, un2, uz2 = calc_okada(HF, U2, xpt, ypt, nu, delta, d, len_val, W,
                                       2, strike_rad, tp)
        except TypeError:
            # calc_okada did not accept vector arrays; mark fallback and break
            vector_ok = False
            break
        except Exception:
            # some implementations raise other errors for vector case; treat as fallback too
            vector_ok = False
            break

        ue_tot += np.asarray(ue1).ravel() + np.asarray(ue2).ravel()
        un_tot += np.asarray(un1).ravel() + np.asarray(un2).ravel()
        uz_tot += np.asarray(uz1).ravel() + np.asarray(uz2).ravel()

    if not vector_ok:
        # fallback: chunked or patch-loop but still faster than single-point if calc_okada supports per-array points
        print("[slip2gps_okada_vectorized] calc_okada is not vectorized: falling back to chunked per-patch computation.")
        ue_tot.fill(0.0); un_tot.fill(0.0); uz_tot.fill(0.0)
        for k in range(Npatch):
            xpt = xgps - xcenters[k]
            ypt = ygps - ycenters[k]
            U1 = s_strike[k]
            U2 = s_dip[k]
            delta = dip0[k] * d2r
            d = -zcenters[k]
            len_val = lp[k]
            W = wp[k]
            strike_rad = strkp[k] * d2r
            tp = np.zeros_like(xpt)

            # compute in smaller chunks to avoid huge memory blow if needed
            chunk = 1000000  # adjust if memory limited
            for i0 in range(0, npts, chunk):
                i1 = min(npts, i0 + chunk)
                xpt_chunk = xpt[i0:i1]; ypt_chunk = ypt[i0:i1]; tp_chunk = tp[i0:i1]
                ue1, un1, uz1 = calc_okada(HF, U1, xpt_chunk, ypt_chunk, nu, delta, d, len_val, W,
                                           1, strike_rad, tp_chunk)
                ue2, un2, uz2 = calc_okada(HF, U2, xpt_chunk, ypt_chunk, nu, delta, d, len_val, W,
                                           2, strike_rad, tp_chunk)
                ue_tot[i0:i1] += np.asarray(ue1).ravel() + np.asarray(ue2).ravel()
                un_tot[i0:i1] += np.asarray(un1).ravel() + np.asarray(un2).ravel()
                uz_tot[i0:i1] += np.asarray(uz1).ravel() + np.asarray(uz2).ravel()

    # reshape to grid
    ue_grid = ue_tot.reshape(x_grid.shape)
    un_grid = un_tot.reshape(x_grid.shape)
    uz_grid = uz_tot.reshape(x_grid.shape)

    # convert coords to km for plotting consistency (like your original)
    x_grid_km = x_grid / 1000.0
    y_grid_km = y_grid / 1000.0

    return ue_grid, un_grid, uz_grid, x_grid_km, y_grid_km

def generate_look_grids(X, Y, track_deg, look_deg, sat_alt_m=700000.0,
                        center=(0.0, 0.0), center_z=0.0, debug=False):
    """
    Generate per-pixel LOS unit vectors (E,N,Up) for a given grid X,Y based on
    a center-provided track & look angle.

    Assumptions / conventions:
      - X, Y are 2D arrays (meshgrid) or 1D arrays (then a meshgrid is made).
      - Coordinates units: meters (same units used for satellite altitude sat_alt_m).
      - track_deg: single scalar azimuth/heading in degrees (measure from North toward East).
      - look_deg: single scalar incidence in degrees.
      - The center point (default (0,0)) is where the provided formula for proj_vec holds.
      - proj_vec_center (as per your formula) is interpreted as **ground -> satellite** unit vector.
      - We place a satellite at sat0 = center + proj_vec_center * R, where R satisfies sat0_z = center_z + R * proj_vec_center_z = sat_alt_m.
        => R = (sat_alt_m - center_z) / proj_vec_center_z
      - For each pixel position p=(x,y,0) (ground z=0, or adjusted by center_z), LOS vector is ground->sat0 = sat0 - p; we normalize it to unit length.
      - Returns look_grid with shape (ny, nx, 3), columns = [lx, ly, lz] (E,N,Up).
      - NOTE: check sign convention with your projection: if you need the vector pointing from satellite->ground instead, flip sign.

    Parameters
    ----------
    X, Y : 2D arrays (ny, nx) OR 1D arrays
        If 1D arrays are provided, they are interpreted as x and y axes and meshgrid will be formed.
    track_deg : float
        heading / track angle at center (deg). E.g. 190 for descending.
    look_deg : float
        incidence angle at center (deg). E.g. 40.
    sat_alt_m : float
        satellite altitude above ground, meters (default 700 km).
    center : tuple (xc, yc)
        coordinate position where given track/look apply (default (0,0)).
    center_z : float
        ground elevation at center (default 0.0).
    debug : bool
        if True prints checks (center proj vectors etc.)

    Returns
    -------
    look_grid : ndarray, shape (ny, nx, 3)
        per-pixel LOS unit vectors oriented as ground -> satellite (E, N, Up).
    sat0 : ndarray shape (3,) the computed satellite position (meters)
    """
    # prepare X,Y as full 2D arrays
    if X is None or Y is None:
        raise ValueError("X and Y must be provided (1D or 2D arrays).")
    X = np.asarray(X)
    Y = np.asarray(Y)
    if X.ndim == 1 and Y.ndim == 1:
        Xg, Yg = np.meshgrid(X, Y)
    elif X.ndim == 2 and Y.ndim == 2:
        Xg, Yg = X, Y
    else:
        raise ValueError("X and Y must be both 1D arrays or both 2D arrays (meshgrid).")

    ny, nx = Xg.shape

    # compute center-proj vector using your formula:
    t = float(track_deg)
    lk = float(look_deg)
    # user formula (note signs): proj_vec_center = [-sin(look)*cos(track),
    #                                              sin(look)*sin(track),
    #                                              cos(look)]
    proj_c = np.array([
        -np.sin(np.deg2rad(lk)) * np.cos(np.deg2rad(t)),
         np.sin(np.deg2rad(lk)) * np.sin(np.deg2rad(t)),
         np.cos(np.deg2rad(lk))
    ], dtype=float)

    # ensure proj_c is unit (numerical safety)
    proj_c = proj_c / (np.linalg.norm(proj_c) + 1e-16)

    # compute R so satellite z = sat_alt_m (satellite position along proj_c direction from center)
    if abs(proj_c[2]) < 1e-12:
        raise ValueError("Center proj_vec has nearly zero vertical component; cannot place satellite for given look angle.")
    R = (sat_alt_m - center_z) / proj_c[2]
    sat0 = np.array([center[0], center[1], center_z], dtype=float) + proj_c * R

    # compute vectors ground->sat0 for each pixel
    # pixel ground z coordinate assumed 0. If your ground has topography, add that as an argument.
    px = sat0[0] - Xg  # east component
    py = sat0[1] - Yg  # north component
    pz = sat0[2] - 0.0  # up component relative to ground plane (if ground z != 0, pass that)

    # vector lengths
    denom = np.sqrt(px * px + py * py + pz * pz)
    # avoid division by zero
    denom[denom == 0] = 1e-16

    lx = px / denom
    ly = py / denom
    lz = pz / denom

    look_grid = np.stack([lx, ly, lz], axis=-1)  # shape (ny, nx, 3)

    if debug:
        # check center cell closest to (center) — find index of center coordinates
        # find nearest pixel to center:
        ix = (np.abs(Xg[0, :] - center[0])).argmin()
        iy = (np.abs(Yg[:, 0] - center[1])).argmin()
        pv = look_grid[iy, ix, :]
        print("proj_c (center formula)   :", proj_c)
        print("look_grid at nearest pix  :", pv)
        print("sat0 (m)                  :", sat0)
        print("R (m)                     :", R)
        # difference:
        print("center diff norm (should be ~0):", np.linalg.norm(pv - proj_c))

    return look_grid, sat0

def forward_3D_dis_fast(file_name, show_plot, dx, dy, xlim, ylim, nu=0.25):
    """
    Fast forward: replace the old pointwise forward with vectorized slip2gps_okada to compute displacement field.
    file_name: input slip parameter file (same as your original; will be parsed to required slip_model_in format)
    show_plot: 1 to plot, else skip plotting
    dx,dy: grid spacing in meters
    xlim,ylim: extents in meters (positive) -> grid from -xlim..xlim, -ylim..ylim
    """
    # load fault parameters (same as your original)
    fault_paras = np.loadtxt(file_name)
    if fault_paras.ndim == 1:
        fault_paras = fault_paras[np.newaxis, :]

    # EXPECTED column mapping in original code:
    # columns (0-index): 0..?
    # from your previous code: xs = fault_paras[:,3]; ys = [:,4]; zs = [:,5];
    #                           L = [:,6]; W = [:,7]; dip = [:,8]; strike = [:,9];
    #                           Us = [:,10]; Ud = [:,11]; Un = [:,12]
    xs = fault_paras[:, 3].astype(float)
    ys = fault_paras[:, 4].astype(float)
    zs = fault_paras[:, 5].astype(float)
    L = fault_paras[:, 6].astype(float)
    W = fault_paras[:, 7].astype(float)
    strike = fault_paras[:, 9].astype(float)
    dip = fault_paras[:, 8].astype(float)  # degrees

    # slip components (use strike/dip fields; if Un present, ignore or include as zero)
    Us = fault_paras[:, 10].astype(float) if fault_paras.shape[1] > 10 else np.zeros(xs.size)
    Ud = fault_paras[:, 11].astype(float) if fault_paras.shape[1] > 11 else np.zeros(xs.size)
    # unify to slip_model_in layout expected by slip2gps_okada_vectorized
    # slip_model_in columns: [fault, patch, layer, x, y, z, length, width, strike, dip, coeff, strike_slip, dip_slip]
    Npatch = xs.size
    slip_model_in = np.zeros((Npatch, 13), dtype=float)
    slip_model_in[:, 0] = np.arange(1, Npatch + 1)      # fault id (dummy)
    slip_model_in[:, 1] = np.arange(1, Npatch + 1)      # patch id
    slip_model_in[:, 2] = 1                             # layer id (dummy)
    slip_model_in[:, 3] = xs
    slip_model_in[:, 4] = ys
    slip_model_in[:, 5] = zs
    slip_model_in[:, 6] = L
    slip_model_in[:, 7] = W
    slip_model_in[:, 8] = strike
    slip_model_in[:, 9] = dip
    slip_model_in[:, 10] = 1.0   # coeff placeholder
    slip_model_in[:, 11] = Us
    slip_model_in[:, 12] = Ud

    xmin, xmax = -xlim, xlim
    ymin, ymax = -ylim, ylim

    print(f"Vectorized forward: computing displacement for grid within x=[{xmin},{xmax}] y=[{ymin},{ymax}] spacing dx={dx},dy={dy}")
    ue, un, uz, Xkm, Ykm = slip2gps_okada_vectorized(xmin, xmax, ymin, ymax, dx, slip_model_in, nu=nu)

    # unit: assume calc_okada returns same units as slip (keep same as original code)
    # Flatten to arrays De, Dn, Dz as before (in cm if slips were in cm)
    De = ue.ravel(); Dn = un.ravel(); Dz = uz.ravel()

    nx = ue.shape[1]; ny = ue.shape[0]  # careful: meshgrid shape is (ny,nx)
    UTM_X = np.arange(xmin, xmax + dx, dx)
    UTM_Y = np.arange(ymin, ymax + dy, dy)
    # plot / save as in your original forward_3D_dis

    # build 2D meshgrid coordinates for pixels (centers in meters)
    Xg, Yg = np.meshgrid(UTM_X, UTM_Y)      # shapes (ny, nx)

    # generate per-pixel LOS grids for descending and ascending tracks
    # user-specified center track/look (example you gave):
    # track = [190 (desc), 350 (asc)], look = [40,40]
    track = np.array([190.0, 350.0])
    look  = np.array([40.0, 40.0])
    # choose satellite altitude (meters). Typical LEO ~ 700 km. Tune if needed.
    sat_alt_m = 700000.0

    look_des_grid, sat0_desc = generate_look_grids(Xg, Yg, track_deg=track[0], look_deg=look[0],
                                                    sat_alt_m=sat_alt_m, center=(0.0, 0.0), center_z=0.0)
    look_asc_grid,  sat0_asc  = generate_look_grids(Xg, Yg, track_deg=track[1], look_deg=look[1],
                                                    sat_alt_m=sat_alt_m, center=(0.0, 0.0), center_z=0.0)
    if show_plot == 1:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)
        # your original plot_data accepted (axes[i], UTM_X, UTM_Y, arr,...)
        # build UTM_X and UTM_Y arrays in meters as centers
        plot_data(axes[0], UTM_X, UTM_Y, ue, "West - East Component, cm", clim=(-50, 50))
        plot_data(axes[1], UTM_X, UTM_Y, un, "South - North Component, cm", clim=(-50, 50))
        plot_data(axes[2], UTM_X, UTM_Y, uz, "Vertical Component, cm", clim=(-50, 50))
        fig.suptitle("Forward 3-D Displacement Field", fontsize=16, weight="bold")
        fig, axes = plt.subplots(2, 3, figsize=(14,10), constrained_layout=True)
        components = ["East (x)", "North (y)", "Up (z)"]

        for i in range(3):
            im = axes[0,i].imshow(look_des_grid[:,:,i], extent=[UTM_X.min(), UTM_X.max(),
                                                                UTM_Y.min(), UTM_Y.max()],
                                                                origin='lower', cmap='jet', vmin=-1, vmax=1)
            axes[0,i].set_title(f"Descending – {components[i]}")
            fig.colorbar(im, ax=axes[0,i])

        for i in range(3):
            im = axes[1,i].imshow(look_asc_grid[:,:,i], extent=[UTM_X.min(), UTM_X.max(),
                                                                UTM_Y.min(), UTM_Y.max()],
                                                                origin='lower', cmap='jet', vmin=-1, vmax=1)
            axes[1,i].set_title(f"Ascending – {components[i]}")
            fig.colorbar(im, ax=axes[1,i])
        plt.show()

    # flatten data and look grids for vectorized dot product
    De = ue.ravel()
    Dn = un.ravel()
    Dz = uz.ravel()
    data_stack = np.vstack([De, Dn, Dz]).T   # shape (N, 3)
    N = data_stack.shape[0]

    # flatten look arrays
    look_desc_flat = look_des_grid.reshape(-1, 3)
    look_asc_flat  = look_asc_grid.reshape(-1, 3)

    # project: elementwise dot product (fast)
    data_dec_flat = np.sum(data_stack * look_desc_flat, axis=1)   # (N,)
    data_asc_flat  = np.sum(data_stack * look_asc_flat, axis=1)

    # reshape back to grid shapes
    data_dec = data_dec_flat.reshape(ue.shape)
    data_asc  = data_asc_flat.reshape(ue.shape)

# (then Save2grd / wrap_to_pi / plotting as before)

    Save2grd("data_dec.grd", UTM_X, UTM_Y, data_dec, "cm", "Simulated displacement field")
    Save2grd("data_asc.grd", UTM_X, UTM_Y, data_asc, "cm", "Simulated displacement field")

    phase_dec = wrap_to_pi(data_dec)
    phase_asc = wrap_to_pi(data_asc)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10), constrained_layout=True)
    plot_data(axes[0,0], UTM_X, UTM_Y, data_asc, "Ascending LOS, cm", clim=(-50, 50))
    plot_data(axes[0,1], UTM_X, UTM_Y, data_dec, "Descending LOS, cm", clim=(-50, 50))
    plot_data(axes[1,0], UTM_X, UTM_Y, phase_asc, "Ascending Wrapped Phase, rad", clim=None, cbar_label="rad")
    plot_data(axes[1,1], UTM_X, UTM_Y, phase_dec, "Descending Wrapped Phase, rad", clim=None, cbar_label="rad")
    fig.suptitle("Forward LOS / Phase Field", fontsize=16, weight="bold")
    plt.show()

    # return arrays similar to original if needed
    return ue, un, uz, data_asc, data_dec

def Calculate_Ph_grd(filename, target_spacing, if_interploate, prefix,
                     save_interpolated=True, if_show=False):
    """
    Memory-optimized phase interpolation + gradient calculation.
    Uses chunked cubic spline interpolation to avoid memory explosion.
    """
    # === Step 1: Read input data ===
    x, y, data, _ = Read_grd_file(filename, "displacement")
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    data = np.asarray(data, dtype=np.float32)

    if x[0] > x[-1]:
        x = x[::-1]; data = data[:, ::-1]
    if y[0] > y[-1]:
        y = y[::-1]; data = data[::-1, :]

    # === Step 2: Prepare new grid ===
    if if_interploate:
        nx_new = int(np.round((x.max() - x.min()) / target_spacing)) + 1
        ny_new = int(np.round((y.max() - y.min()) / target_spacing)) + 1
        x_new = np.linspace(x.min(), x.max(), nx_new)
        y_new = np.linspace(y.min(), y.max(), ny_new)
    else:
        x_new, y_new = x, y
        nx_new, ny_new = len(x_new), len(y_new)

    # === Step 3: 内存检查 ===
    avail_mem = psutil.virtual_memory().available / (1024 ** 3)
    est_bytes = nx_new * ny_new * 4 / (1024 ** 3)
    print(f"[INFO] New grid: {nx_new}x{ny_new}, est memory {est_bytes:.2f} GB, available {avail_mem:.2f} GB")

    # === Step 4: 使用 RectBivariateSpline 分块 cubic 插值 ===
    if if_interploate:
        print("[INFO] Using chunked RectBivariateSpline cubic interpolation...")
        spline = RectBivariateSpline(y, x, data, kx=3, ky=3, s=0)

        # 使用 memmap 存储插值结果
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".npy")
        data_in = np.memmap(temp_file.name, dtype=np.float32, mode="w+", shape=(ny_new, nx_new))

        # 动态选择分块大小
        max_gb = avail_mem * 0.3
        row_chunk = int(max(100, min(ny_new, (max_gb * 1e9) // (nx_new * 4))))
        print(f"[INFO] Interpolating in chunks of {row_chunk} rows...")

        for i0 in range(0, ny_new, row_chunk):
            i1 = min(i0 + row_chunk, ny_new)
            data_in[i0:i1, :] = spline(y_new[i0:i1], x_new).astype(np.float32)

        print(f"[INFO] Interpolation done → {temp_file.name}")

    else:
        data_in = np.asarray(data, dtype=np.float32)

    # === Step 5: Compute gradient (central diff) ===
    dx = float(x_new[1] - x_new[0])
    print(dx)
    dy = float(y_new[1] - y_new[0])
    print(dy)

    f_ip1 = data_in[1:-1, 2:]
    f_im1 = data_in[1:-1, :-2]
    f_jp1 = data_in[2:, 1:-1]
    f_jm1 = data_in[:-2, 1:-1]

    ph_grd_x = (f_ip1 - f_im1) / (2.0 * dx)
    ph_grd_y = (f_jp1 - f_jm1) / (2.0 * dy)

    # NaN mask
    mask_nan = np.isnan(f_ip1) | np.isnan(f_im1) | np.isnan(f_jp1) | np.isnan(f_jm1)
    ph_grd_x[mask_nan] = np.nan
    ph_grd_y[mask_nan] = np.nan

    x_grd = x_new[1:-1]
    y_grd = y_new[1:-1]

    # === Step 6: Save results ===
    if save_interpolated:
        Save2grd(f"{prefix}_interp.grd", x_new, y_new, np.array(data_in), "None",
                 "Interpolated Phase (LOS)", "UTM-X", "UTM-Y", "Phase")
    Save2grd(f"{prefix}_grd_x.grd", x_grd, y_grd, ph_grd_x, "None", "Phase Gradient (E-W)", "UTM-X", "UTM-Y", "Grad X")
    Save2grd(f"{prefix}_grd_y.grd", x_grd, y_grd, ph_grd_y, "None", "Phase Gradient (N-S)", "UTM-X", "UTM-Y", "Grad Y")

    # === Step 7: Optional plot ===
    if if_show:
        fig, axes = plt.subplots(1, 3, figsize=(13, 5), constrained_layout=True)
        plot_data(axes[0], x_new, y_new, data_in, "Interpolated LOS", clim=(-50, 50))
        plot_data(axes[1], x_grd, y_grd, ph_grd_x, "Grad X", clim=(-1e-2, 1e-2),cbar_label = "Phase gradient")
        plot_data(axes[2], x_grd, y_grd, ph_grd_y, "Grad Y", clim=(-1e-2, 1e-2),cbar_label = "Phase gradient")
        plt.show()

    return x_grd, y_grd, ph_grd_x, ph_grd_y

def plot_profile(config_file, data_file,
                 ax_map=None, ax_profile=None,
                 prefix="Ascending",
                 show_map=True, show_profile=True,
                 clear_profile_ax=True,
                 profile_plot_kwargs=None,
                 if_show_trace=True):
    """
    改进版 plot_profile：修复子图拥挤、limits 被覆盖、suptitle 空白过大等问题。
    主要策略：
      - 关闭 constrained_layout 自动布局，改用 fig.subplots_adjust 手工微调；
      - 在设置 xlim/ylim 后禁用自动缩放，确保 limits 不被后续元素覆盖；
      - 明确 map 坐标以 km 为单位（绘图与 limits 一致）；
      - 使用 indexing='xy' 创建 meshgrid 避免转置歧义。
    """

    cfg = parse_config(config_file)
    num_of_faults = int(cfg["model"]["model_params"]["num_of_faults"])

    # --- 读取断层参数 (patches list) ---
    patches = []
    for i in range(1, num_of_faults + 1):
        trace = f"trace{i}"
        patch = {
            "x": float(cfg["seis_fault1"][trace]["x"]),
            "y": float(cfg["seis_fault1"][trace]["y"]),
            "z": float(cfg["seis_fault1"][trace]["z"]),
            "len": float(cfg["seis_fault1"][trace]["len"]),
            "wid": float(cfg["seis_fault1"][trace]["wid"]),
            "dip": float(cfg["seis_fault1"][trace]["dip"]),
            "strike": float(cfg["seis_fault1"][trace]["strike"]),
        }
        if patch["strike"] == 90:
            patch["strike"] = 89.9
        patches.append(patch)

    # --- 读取数据 ---
    x, y, data, _ = Read_grd_file(data_file)  # 假设 x,y 单位为 meters

    # --- map 轴准备（注意：关闭 constrained_layout，使用 subplots_adjust） ---
    if show_map:
        if ax_map is None:
            fig_map, ax_map = plt.subplots(figsize=(7, 6), constrained_layout=False)
            # 手工调整边距以避免 suptitle 与子图冲突
            fig_map.subplots_adjust(left=0.08, right=0.98, top=0.88, bottom=0.08)
        # 若 plot_data 内部使用 x/1000 进行了 km 转换，则这里传入原始 x,y 即可
        plot_data(ax_map, x, y, data, data_file)

    # --- 断层 trace 计算 & 绘图 ---
    x1_list, y1_list, x2_list, y2_list = plot_trace(patches, ax=ax_map, if_show=if_show_trace)
    if len(x1_list) == 0 or len(x2_list) == 0:
        raise RuntimeError("plot_trace 未返回有效端点。")
    x1, y1, x2, y2 = (np.mean(x1_list), np.mean(y1_list), np.mean(x2_list), np.mean(y2_list))
    x1, y1, x2, y2 = float(x1), float(y1), float(x2), float(y2)

    # --- 参数 ---
    cross_distance = 300_000.0   # ±300 km in meters
    profile_halfwidth = 1_000.0  # ±1 km in meters

    # --- fault vectors (meters) ---
    fault_vec = np.array([x2 - x1, y2 - y1], dtype=float)
    fault_len = np.hypot(fault_vec[0], fault_vec[1])
    if fault_len == 0:
        raise ValueError("Fault endpoints identical or zero-length calculated.")
    fault_dir = fault_vec / fault_len
    fault_norm = np.array([-fault_dir[1], fault_dir[0]])
    x_mid, y_mid = 0.5 * (x1 + x2), 0.5 * (y1 + y2)

    # --- construct meshgrid 时使用 indexing='xy' 以避免转置歧义 ---
    X, Y = np.meshgrid(x, y, indexing='xy')
    pts = np.column_stack((X.ravel(), Y.ravel()))
    vals = data.ravel()
    vecs = pts - np.array([x_mid, y_mid])
    dist_normal = np.dot(vecs, fault_norm)
    dist_along = np.dot(vecs, fault_dir)

    # select strip ±1km and ±300km along normal (units: meters)
    mask = (np.abs(dist_along) <= profile_halfwidth) & (np.abs(dist_normal) <= cross_distance)
    dist_normal_sel = dist_normal[mask]
    vals_sel = vals[mask]

    # --- profile axis 准备 ---
    if show_profile:
        if ax_profile is None:
            fig_prof, ax_profile = plt.subplots(figsize=(10, 5), constrained_layout=False)
            fig_prof.subplots_adjust(left=0.08, right=0.98, top=0.90, bottom=0.10, hspace=0.25)
        else:
            if clear_profile_ax:
                ax_profile.cla()

        lower_name = data_file.lower()
        is_gradient = any(s in lower_name for s in ['_grd_x', '_grd_y', '_grad', 'gradient', '_dx', '_dy'])
        if is_gradient:
            y_min, y_max = -8e-3, 8e-3
            ylabel = "Phase gradient"
        else:
            y_min, y_max = -250.0, 250.0
            ylabel = "LOS displacement (cm)"

        # mask out central ±50 km (meters)
        mask_zone_m = 50_000.0
        not_in_mask_zone = np.abs(dist_normal_sel) > mask_zone_m
        plot_x = dist_normal_sel[not_in_mask_zone] / 1000.0  # 转为 km for plotting
        plot_y = vals_sel[not_in_mask_zone]

        # 绘制灰色矩形（中间 ±50 km）
        ax_profile.axvspan(-mask_zone_m/1000.0, mask_zone_m/1000.0,
                           color='gray', alpha=0.35, zorder=0, label='_nolegend_')

        pkw = {"marker": "o", "linestyle": "None", "markersize": 3, "alpha": 0.6, "color": "k"}
        if profile_plot_kwargs:
            pkw.update(profile_plot_kwargs)
        ax_profile.plot(plot_x, plot_y, **pkw)

        # fault center vertical line
        ax_profile.axvline(0.0, color='r', linestyle='--', lw=1.5, label='_nolegend_')

        # labels, title, limits
        ax_profile.set_xlabel("Cross-fault distance (km)")
        ax_profile.set_ylabel(ylabel)
        # 给 title 较小的 pad 避免与 suptitle 大空白
        ax_profile.set_title(f"{prefix} Profile (±1 km width) — {data_file}", pad=8)
        # 先设置 xlim/ylim，再禁用 autoscale，防止后面绘图动作改变 limits
        ax_profile.set_xlim(-cross_distance/1000.0, cross_distance/1000.0)
        ax_profile.set_ylim(y_min, y_max)
        ax_profile.set_autoscale_on(False)  # 关键：关闭自动缩放，避免后续图层改变 limits
        ax_profile.grid(True, linestyle='--', alpha=0.4)
        # legend only if necessary
        try:
            ax_profile.legend()
        except Exception:
            pass

    # --- map overlays: profile line and fault trace (统一以 km 显示) ---
    if show_map:
        a_start = np.array([x_mid, y_mid]) - cross_distance * fault_norm
        a_end   = np.array([x_mid, y_mid]) + cross_distance * fault_norm

        # convert to km when plotting on map because plot_data likely used km or you want km axes
        ax_map.plot([a_start[0]/1000, a_end[0]/1000],
                    [a_start[1]/1000, a_end[1]/1000],
                    'k-', lw=2, label="Profile A–A'")
        if if_show_trace:
            ax_map.plot([x1/1000, x2/1000],
                        [y1/1000, y2/1000],
                        'r-', lw=2, label="Fault trace")

        # put labels near the ends (offsets in meters -> convert to km offset)
        ax_map.text(a_start[0]/1000 + 20.0, a_start[1]/1000 + 20.0, "A", color='k', fontsize=10, weight='bold')
        ax_map.text(a_end[0]/1000 - 20.0, a_end[1]/1000 - 20.0, "A'", color='k', fontsize=10, weight='bold')

        # Set map limits in km (use appropriate region; here we set +/- 600 km as before)
        ax_map.set_xlim([-600, 600])
        ax_map.set_ylim([-600, 600])
        ax_map.set_aspect("equal", adjustable="box")

        # set clim on map images/collections (attempt safely)
        lower_name = data_file.lower()
        if any(s in lower_name for s in ['_grd_x', '_grd_y', '_grad', 'gradient', '_dx', '_dy']):
            vmin, vmax = -5e-3, 5e-3
        else:
            vmin, vmax = -150.0, 150.0

        for im in ax_map.get_images():
            try:
                im.set_clim(vmin, vmax)
            except Exception:
                pass
        for coll in ax_map.collections:
            try:
                coll.set_clim(vmin, vmax)
            except Exception:
                pass

        try:
            ax_map.legend()
        except Exception:
            pass

    return ax_map, (ax_profile if show_profile else None)

def Proj_Geo_coor(file_name, lon_trench):
    """
    将UTM网格数据转换为地理坐标（经纬度）网格并保存。
    假定输入的UTM坐标是中心化的（easting 已减 500000）。
    仅需指定数据文件名和参考经度 lon_trench。
    """

    # ===== 读取数据 =====
    x, y, data, _ = Read_grd_file(file_name)
    print(f"UTM X range: {x.min()} to {x.max()}")
    print(f"UTM Y range: {y.min()} to {y.max()}")

    # ===== 推算UTM分区号 =====
    i_zone = int(np.floor(((lon_trench + 180.0) % 360.0) / 6.0) + 1)
    print(f"UTM Zone: {i_zone}")

    # ===== 计算参考点 (UTM原点对应的经纬度) =====
    lon0, lat0 = utm2ll(0, 0, i_zone, i_type=2, northern=True)
    dlon = lon0 - lon_trench
    print(f"Reference lon0={lon0:.6f}, lat0={lat0:.6f}, dlon={dlon:.6f}")

    # ===== 创建输出数组 =====
    Lon = np.empty((len(y), len(x)))
    Lat = np.empty((len(y), len(x)))

    # ===== 网格化计算 =====
    X, Y = np.meshgrid(x, y)
    X_flat, Y_flat = X.ravel(), Y.ravel()

    lon_flat, lat_flat = utm2ll(X_flat, Y_flat, i_zone, i_type=2, northern=True)
    lon_flat -= dlon  # 对齐到参考经度
    Lon[:, :] = lon_flat.reshape(Y.shape)
    Lat[:, :] = lat_flat.reshape(Y.shape)

    # ===== 保留4位有效数字 =====
    Lon = np.round(Lon, 4)
    Lat = np.round(Lat, 4)

    # ===== 输出范围检查 =====
    print(f"Longitude range: {Lon.min()} to {Lon.max()}")
    print(f"Latitude range:  {Lat.min()} to {Lat.max()}")

    # ===== 保存新文件 =====
    out_name = file_name.replace(".grd", "_lonlat.grd")
    Save2grd(out_name, Lon[0, :], Lat[:, 0], data,
             "None", "Converted to geographic coordinates", "Longitude", "Latitude", "Data")

    print(f"✅ Saved converted grid → {out_name}")
    return Lon, Lat, data

# def plot_gmt_map(
#     grid_file: str,
#     map_type: str = "displacement",
#     fault_file: str = "gem_active_faults.gmt",
#     projection: str = "M10c",
#     transparency: int = 50,
#     if_wrap: bool = False,
#     if_mask: bool = False,
#     if_save: bool = False,
#     show: bool = True
# ):
#     """
#     绘制PyGMT地图，可选择掩膜、包裹数据，并可保存处理后的数据。
    
#     参数
#     ----
#     grid_file : str
#         输入grd文件
#     map_type : str
#         "displacement" 或 "gradient"，决定色条标签
#     fault_file : str
#         断层文件
#     projection : str
#         地图投影
#     transparency : int
#         grdimage透明度
#     if_wrap : bool
#         是否调用 wrap_to_pi 包裹数据
#     if_mask : bool
#         是否生成 landmask 掩膜
#     if_save : bool
#         是否保存处理后的数据
#     show : bool
#         是否显示绘图
#     """
#     # === 打开数据 ===
#     data = xr.open_dataset(grid_file)
#     data_array = data['Data'].values if 'Data' in data else data['z'].values
#     lon, lat = data['x'].values, data['y'].values
#     region = [lon.min(), lon.max(), lat.min(), lat.max()]

#     # === 掩膜处理（优先） ===
#     if if_mask:
#         inc = pg.grdinfo(grid_file, spacing=True)
#         inc_values = inc.strip().replace('-I', '')
#         landmask = pg.grdlandmask(region=region, spacing=inc_values, maskvalues=[np.nan, 1], resolution='f')
#         data_array = np.where(np.isnan(landmask), np.nan, data_array)
#         if if_save:
#             new_file = os.path.splitext(grid_file)[0] + "_mask.grd"
#             xr.DataArray(data_array, coords=[lat, lon], dims=['y','x']).to_dataset(name='Data').to_netcdf(new_file)
#             print(f"Masked data saved to {new_file}")

#     # === 包裹处理 ===
#     if if_wrap:
#         data_array = wrap_to_pi(data_array)
#         if if_save:
#             new_file = os.path.splitext(grid_file)[0] + "_wraped.grd"
#             xr.DataArray(data_array, coords=[lat, lon], dims=['y','x']).to_dataset(name='Data').to_netcdf(new_file)
#             print(f"Wrapped data saved to {new_file}")

#     # === 绘图 ===
#     fig = pg.Figure()
#     fig.grdimage(grid='@earth_relief_01m', projection=projection, region=region, shading=True, cmap='gray95')
#     fig.coast(water='lightblue', shorelines='0.3p,gray', resolution='f', frame=True)

#     # 临时文件用于绘图
#     tmp_file = "tmp_data.grd"
#     xr.DataArray(data_array, coords=[lat, lon], dims=['y','x']).to_dataset(name='Data').to_netcdf(tmp_file)
#     if map_type == "displacement":
#         pg.makecpt(cmap="rainbow", series=[-50, 50])
#     elif map_type == "Wrapped Phase":
#         pg.makecpt(cmap="rainbow", series=[-np.pi, -np.pi])
#     else:
#         pg.makecpt(cmap="rainbow", series=[-1e-2, 1e-2])
        
#     fig.grdimage(grid=tmp_file, cmap=True, transparency=transparency)
#     # 绘制断层和震中
#     fig.plot(data=fault_file, pen="0.5p,yellow")
#     fig.plot(x=[98.6869, 95.6712], y=[-2.2465, 2.2465], pen="3p,red", label="Fault trace")
#     fig.plot(x=97.6553, y=0, style="a0.4c", fill="red", label="Epicenter")
#     fig.legend(position="JTL+jTL+o0.1c+w2.5c", box=True)

#     # 色条
#     if map_type.lower() in ["displacement", "los", "phase", "phasemap"]:
#         cbar_label = "LOS displacement (cm)"
#         print("Plotting unwrapped phase / LOS displacement map...")
#     elif map_type.lower() in ["gradient", "phase_gradient"]:
#         cbar_label = "Phase Gradient"
#         print("Plotting phase gradient map...")
#     else:
#         cbar_label = "Data value"
#         print(f"Plotting generic data map for type: {map_type}")

#     fig.colorbar(
#         position="jBR+o0.7c/0.8c+h+w5c/0.3c+ml",
#         box=Box(pen="0.8p,black", fill="white@30"),
#         frame=[f"x+l{cbar_label}"]
#     )

#     if show:
#         fig.show()

#     os.remove(tmp_file)


def plot_gmt_map(
    grid_file: str,
    map_type: str = "displacement",
    fault_file: str = "gem_active_faults.gmt",
    projection: str = "M10c",
    transparency: int = 50,
    if_wrap: bool = True,
    if_mask: bool = True,
    if_save: bool = False,
    show: bool = True
):
    """
    绘制PyGMT地图，可选择掩膜、包裹数据，并可保存处理后的数据。
    
    参数
    ----
    grid_file : str
        输入grd文件
    map_type : str
        "displacement" 或 "gradient"，决定色条标签
    fault_file : str
        断层文件
    projection : str
        地图投影
    transparency : int
        grdimage透明度
    if_wrap : bool
        是否调用 wrap_to_pi 包裹数据
    if_mask : bool
        是否生成 landmask 掩膜
    if_save : bool
        是否保存处理后的数据
    show : bool
        是否显示绘图
    """
    # === 打开数据 ===
    data = xr.open_dataset(grid_file)
    data_array = data['Data'].values if 'Data' in data else data['z'].values
    lon, lat = data['x'].values, data['y'].values
    region = [lon.min(), lon.max(), lat.min(), lat.max()]

    # === 掩膜处理（优先） ===
    if if_mask:
        inc = pg.grdinfo(grid_file, spacing=True)
        inc_values = inc.strip().replace('-I', '')
        landmask = pg.grdlandmask(region=region, spacing=inc_values, maskvalues=[np.nan, 1], resolution='f')
        data_array = np.where(np.isnan(landmask), np.nan, data_array)
        # 临时文件用于绘图
        xr.DataArray(data_array, coords=[lat, lon], dims=['y','x']).to_dataset(name='Data').to_netcdf("tmp1_data.grd")
        if if_save:
            new_file = os.path.splitext(grid_file)[0] + "_mask.grd"
            if map_type == "displacement":
                unit = "cm"
                Annot = "Phase"
                data_name = "displacement"
            else:
                unit = "None"
                Annot = "Grad"
                data_name = "Gradient"
            Save2grd(new_file,lon,lat,data_array,unit,Annot,X_name = "Lon",Y_name = "Lat",data_name = data_name)
            print(f"Masked data saved to {new_file}")

    # === 包裹处理 ===
    if if_wrap:
        data_array = wrap_to_pi(data_array)
        xr.DataArray(data_array, coords=[lat, lon], dims=['y','x']).to_dataset(name='Data').to_netcdf("tmp2_data.grd")
        if if_save:
            new_file = os.path.splitext(grid_file)[0] + "_wraped.grd"
            if map_type == "displacement":
                unit = "cm"
                Annot = "Phase"
                data_name = "displacement"
            else:
                unit = "None"
                Annot = "Grad"
                data_name = "Gradient"
            Save2grd(new_file,lon,lat,data_array,unit,Annot,X_name = "Lon",Y_name = "Lat",data_name = data_name)
            print(f"Wrapped data saved to {new_file}")

    if map_type == "displacement":
        # === 绘图 ===
        fig = pg.Figure()
        with fig.subplot(
        nrows=1,
        ncols=3,
        figsize=("22c", "9c"),  # width of 15 cm, height of 6 cm
        autolabel=True,
        margins="0.3c",  # horizontal 0.3 cm and vertical 0.2 cm margins
        sharex="b",  # shared x-axis on the bottom side
        sharey="l",  # shared y-axis on the left side
        ):
            with fig.set_panel(panel=0):
                #fig.grdimage(grid='@earth_relief_01m', projection=projection, region=region, shading=True, cmap='gray95')
                fig.coast(water='lightblue', shorelines='0.3p,gray', resolution='f')
                cbar_label = "LOS displacement (cm)"
                pg.makecpt(cmap="rainbow", series=[-150, 50])
                fig.grdimage(grid=grid_file, cmap=True, transparency=transparency)
                    # 绘制断层和震中
                fig.plot(data=fault_file, pen="0.5p,yellow")
                fig.plot(x=[98.6869, 95.6712], y=[-2.2465, 2.2465], pen="3p,red", label="Fault trace")
                fig.plot(x=97.6553, y=0, style="a0.4c", fill="red", label="Epicenter")
                fig.legend(position="JTL+jTL+o0.2c/1.2c+w2.5c", box=False)
                print("Plotting unwrapped phase / LOS displacement map...")
                fig.colorbar(
                position="jBR+o0.5c/0.5c+h+w3c/0.3c+ml",
                box=Box(pen="0.8p,black", fill="white@30"),
                frame=[f"x+l{cbar_label}"])
            with fig.set_panel(panel=1):
                # fig.grdimage(grid='@earth_relief_01m', projection=projection, region=region, shading=True, cmap='gray95')
                fig.coast(water='lightblue', shorelines='0.3p,gray', resolution='f')
                cbar_label = "LOS displacement (cm)"
                pg.makecpt(cmap="rainbow", series=[-150, 50])
                fig.grdimage(grid="tmp1_data.grd", cmap=True, transparency=transparency)
                    # 绘制断层和震中
                fig.plot(data=fault_file, pen="0.5p,yellow")
                fig.plot(x=[98.6869, 95.6712], y=[-2.2465, 2.2465], pen="3p,red", label="Fault trace")
                fig.plot(x=97.6553, y=0, style="a0.4c", fill="red", label="Epicenter")
                fig.legend(position="JTL+jTL+o0.2c/1.2c+w2.5c", box=False)
                print("Plotting unwrapped phase / LOS displacement map...")
                pg.makecpt(cmap="rainbow", series=[-150, 50])
                fig.colorbar(
                position="jBR+o0.5c/0.5c+h+w3c/0.3c+ml",
                box=Box(pen="0.8p,black", fill="white@30"),
                frame=[f"x+l{cbar_label}"])
            with fig.set_panel(panel=2):
                # fig.grdimage(grid='@earth_relief_01m', projection=projection, region=region, shading=True, cmap='gray95')
                fig.coast(water='lightblue', shorelines='0.3p,gray', resolution='f')
                pg.makecpt(cmap="rainbow", series=[-np.pi, np.pi])
                fig.grdimage(grid="tmp2_data.grd", cmap=True, transparency=transparency)
                    # 绘制断层和震中
                fig.plot(data=fault_file, pen="0.5p,yellow")
                fig.plot(x=[98.6869, 95.6712], y=[-2.2465, 2.2465], pen="3p,red", label="Fault trace")
                fig.plot(x=97.6553, y=0, style="a0.4c", fill="red", label="Epicenter")
                fig.legend(position="JTL+jTL+o0.2c/1.2c+w2.5c", box=False)
                print("Plotting wrapped phase map...")
                cbar_label = "rad"
                fig.colorbar(
                    position="jBR+o0.5c/0.5c+h+w3c/0.3c+ml",
                    box=Box(pen="0.8p,black", fill="white@30"),
                    frame=[f"x+l{cbar_label}"])
                pg.makecpt(cmap="rainbow", series=[-np.pi, np.pi])

    else:
        fig = pg.Figure()
        with fig.subplot(
        nrows=1,
        ncols=3,
        figsize=("15c", "8c"),  # width of 15 cm, height of 6 cm
        autolabel=True,
        margins="0.3c",  # horizontal 0.3 cm and vertical 0.2 cm margins
        sharex="b",  # shared x-axis on the bottom side
        sharey="l",  # shared y-axis on the left side
        ):
            with fig.set_panel(panel=0):
                # fig.grdimage(grid='@earth_relief_01m', projection=projection, region=region, shading=True, cmap='gray95')
                fig.coast(water='lightblue', shorelines='0.3p,gray', resolution='f', frame=True)
                cbar_label = "Phase gradient "
                pg.makecpt(cmap="rainbow", series=[-1e-2, 1e-2])
                fig.grdimage(grid=grid_file, cmap=True, transparency=transparency)
                    # 绘制断层和震中
                fig.plot(data=fault_file, pen="0.5p,yellow")
                fig.plot(x=[98.6869, 95.6712], y=[-2.2465, 2.2465], pen="3p,red", label="Fault trace")
                fig.plot(x=97.6553, y=0, style="a0.4c", fill="red", label="Epicenter")
                fig.legend(position="JTL+jTL+o0.2c/1.2c+w2.5c", box=False)
                print("Plotting Phase gradient map ...")
                fig.colorbar(
                position="jBR+o0.5c/0.5c+h+w3c/0.3c+ml",
                box=Box(pen="0.8p,black", fill="white@30"),
                frame=[f"x+l{cbar_label}"])
            with fig.set_panel(panel=1):
                # fig.grdimage(grid='@earth_relief_01m', projection=projection, region=region, shading=True, cmap='gray95')
                fig.coast(water='lightblue', shorelines='0.3p,gray', resolution='f', frame=True)
                cbar_label = "Phase gradient "
                pg.makecpt(cmap="rainbow", series=[-1e-2, 1e-2])
                fig.grdimage(grid="tmp1_data.grd", cmap=True, transparency=transparency)
                    # 绘制断层和震中
                fig.plot(data=fault_file, pen="0.5p,yellow")
                fig.plot(x=[98.6869, 95.6712], y=[-2.2465, 2.2465], pen="3p,red", label="Fault trace")
                fig.plot(x=97.6553, y=0, style="a0.4c", fill="red", label="Epicenter")
                fig.legend(position="JTL+jTL+o0.2c/1.2c+w2.5c", box=False)
                print("Plotting Phase gradient ...")
                pg.makecpt(cmap="rainbow", series=[-1e-2, 1e-2])
                fig.colorbar(
                position="jBR+o0.5c/0.5c+h+w3c/0.3c+ml",
                box=Box(pen="0.8p,black", fill="white@30"),
                frame=[f"x+l{cbar_label}"])

    # if map_type == "displacement":
    #     pg.makecpt(cmap="rainbow", series=[-50, 50])
    # elif map_type == "Wrapped Phase":
    #     pg.makecpt(cmap="rainbow", series=[-np.pi, -np.pi])
    # else:
    #     pg.makecpt(cmap="rainbow", series=[-1e-2, 1e-2])

    if show:
        fig.show()

    os.remove("tmp1_data.grd")
    os.remove("tmp2_data.grd")


def rewrite_llde_with_los(file_path, proj_vec, overwrite=True, fmt="%.9f", verbose=True):
    """
    读取一个文本数据文件（每行为：x y val samp_rate ...），
    将其转换为 8 列格式并保存（覆盖原文件或写入新文件）。

    输出列含义 (1-based):
      1: x (原第1列)
      2: y (原第2列)
      3: zeros
      4: LOS_x (proj_vec[0])
      5: LOS_y (proj_vec[1])
      6: LOS_z (proj_vec[2])
      7: original value (原第3列)
      8: original sampling rate (原第4列)

    参数
    ----
    file_path : str
        要处理的文件路径（例如 "/path/to/file.llde"）。
    proj_vec : array-like of length 3
        LOS 方向余弦向量 (vx, vy, vz)，例如 proj_vec1 或 proj_vec2。
    overwrite : bool
        是否覆盖原文件（True，先写临时文件再替换）；若 False 则写入 file_path + ".out"。
    fmt : str
        写文件时每个数值的格式（适用于 np.savetxt）。
    verbose : bool
        是否打印处理信息。

    返回
    ----
    out_path : str
        最终写入的文件路径。
    """
    # --- 参数/输入检查 ---
    proj_vec = np.asarray(proj_vec, dtype=float).ravel()
    if proj_vec.size != 3:
        raise ValueError("proj_vec 必须是长度为3的一维数组 (vx, vy, vz)。")

    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"输入文件不存在: {file_path}")

    # 读取数据：允许文件中有 >4 列，我们只使用前4列（如存在）
    try:
        data = np.loadtxt(file_path, dtype=float)
    except Exception as e:
        raise RuntimeError(f"读取文件失败: {file_path}\n{e}")

    if data.ndim == 1:
        # 单行文件：把它变成 (1, ncols)
        data = data.reshape(1, -1)

    nrows, ncols = data.shape
    if ncols < 4:
        raise ValueError(f"输入文件至少应有 4 列 (x y value samp_rate)。检测到列数 = {ncols}")

    # --- 截取必要列并构造新矩阵 ---
    x = data[:, 0]
    y = data[:, 1]
    val = data[:, 2]      # 原第3列 -> 新第7列
    samp = data[:, 3]     # 原第4列 -> 新第8列
    samp = np.full(nrows, 1, dtype=float)
    tp = np.zeros(nrows, dtype=float)
    los_x = np.full(nrows, proj_vec[0], dtype=float)
    los_y = np.full(nrows, proj_vec[1], dtype=float)
    los_z = np.full(nrows, proj_vec[2], dtype=float)

    # 组合为 (nrows, 8)
    out_arr = np.column_stack((x, y, tp, los_x, los_y, los_z, val, samp))

    # --- 写入临时文件然后替换（更安全） ---
    if overwrite:
        dirn, base = os.path.split(file_path)
        tf = None
        try:
            fd, temp_path = tempfile.mkstemp(prefix=base + ".", dir=dirn)
            os.close(fd)
            np.savetxt(temp_path, out_arr, fmt=fmt, delimiter="\t")
            # 替换原文件
            shutil.move(temp_path, file_path)
            out_path = file_path
        except Exception as e:
            # 若写入临时文件失败，尝试直接写出 .out 文件并抛错
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except Exception:
                    pass
            raise RuntimeError(f"写入/替换文件失败: {e}")
    else:
        out_path = file_path + ".out"
        np.savetxt(out_path, out_arr, fmt=fmt, delimiter="\t")

    if verbose:
        print(f"Processed '{file_path}' -> '{out_path}'")
        print(f" rows: {nrows}, input cols: {ncols} -> output cols: 8")
        print(f" LOS used: [{proj_vec[0]:.6g}, {proj_vec[1]:.6g}, {proj_vec[2]:.6g}]")

    return out_path


# ===== 使用示例 =====

if __name__ == "__main__":
    file_name = "test_model.inv";
    # forward_3D_dis(file_name, 1,500,500,1e5,1e5);
    forward_3D_dis_fast(file_name, 1,1000,1000,1e5,1e5);
    # Calculate_Ph_grd("data_asc.grd",50,1,"Asc",True);
    # Calculate_Ph_grd("data_dec.grd",50,1,"Des",True);
