import xarray as xr
from Read_Config import parse_config
import numpy as np
import matplotlib.pyplot as plt
from src import Read_grd_file
from src import get_patch_endpoints

def quatree(a, b, c, r, up, down, rects=None):
    """
    Quadtree subsampling with region visualization support.
    
    a, b : 2D grid coordinates (same shape as c)
    c    : data matrix
    r    : threshold controlling subdivision
    up   : upper bound of subdivision
    down : lower bound of subdivision
    rects: list of rectangles [xmin, xmax, ymin, ymax] for visualization
    """
    if rects is None:
        rects = []

    x, y, z, nn = [], [], [], []
    m, n = c.shape

    # 如果该区域几乎全是 NaN，则直接返回
    if np.isnan(c).sum() >= m * n * 0.999999:
        return np.array(x), np.array(y), np.array(z), np.array(nn), rects

    # 如果块太小，停止递归，用平均值表示
    if m <= down or n <= down:
        x.append(np.nanmean(a))
        y.append(np.nanmean(b))
        z.append(np.nanmean(c))
        nn.append(np.count_nonzero(~np.isnan(c)))
        rects.append([a.min(), a.max(), b.min(), b.max()])
        return np.array(x), np.array(y), np.array(z), np.array(nn), rects

    # 四叉树划分
    half_m, half_n = m // 2, n // 2
    for j in [0, half_m]:
        for k in [0, half_n]:
            sub_a = a[j:j+half_m, k:k+half_n]
            sub_b = b[j:j+half_m, k:k+half_n]
            sub_c = c[j:j+half_m, k:k+half_n]

            # 去均值后的子块
            q = sub_c - np.nanmean(sub_c)
            q[np.isnan(q)] = 0

            if (np.sqrt(np.mean(q**2)) <= r) and m <= up and n <= up:
                # Smoothed area
                if np.isnan(sub_c).sum() >= sub_c.size * 0.75:
                    continue
                sub_a = sub_a.astype(float).copy()
                sub_b = sub_b.astype(float).copy()
                sub_a[np.isnan(sub_c)] = np.nan
                sub_b[np.isnan(sub_c)] = np.nan

                x.append(np.nanmean(sub_a))
                y.append(np.nanmean(sub_b))
                z.append(np.nanmean(sub_c))
                nn.append(np.count_nonzero(~np.isnan(sub_c)))
                rects.append([sub_a.min(), sub_a.max(), sub_b.min(), sub_b.max()])
            else:
                # with short-wavelength signal 
                x0, y0, z0, n0, rects = quatree(sub_a, sub_b, sub_c, r, up, down, rects)
                x.extend(x0)
                y.extend(y0)
                z.extend(z0)
                nn.extend(n0)

    return np.array(x), np.array(y), np.array(z), np.array(nn), rects


def quatree_median(a, b, c, r, up, down, rects=None):
    """
    Quadtree subsampling using median instead of mean.

    Parameters
    ----------
    a : 2D array x coordinates
    b : 2D array y coordinates
    c : 2D array data values
    r : float
        threshold for subsampling (smoothness control)
    up : int upper bound on block size
    down : int lower bound on block size

    Returns
    -------
    x, y, z, nn : 1D arrays
        subsampled coordinates, values, and counts of valid points
    """
    if rects is None:
        rects = []

    x, y, z, nn = [], [], [], []
    m, n = c.shape

    # 如果该区域几乎全是 NaN，则返回空
    if np.isnan(c).sum() >= m * n * 0.99:
        return np.array(x), np.array(y), np.array(z), np.array(nn)

    # 如果块太小，停止递归，用均值表示
    if m <= down or n <= down:
        x.append(np.nanmean(a.reshape(-1)))
        y.append(np.nanmean(b.reshape(-1)))
        z.append(np.nanmean(c.reshape(-1)))
        nn.append(np.count_nonzero(~np.isnan(c)))
        rects.append([a.min(), a.max(), b.min(), b.max()])
        return np.array(x), np.array(y), np.array(z), np.array(nn), rects

    half_m = m // 2
    half_n = n // 2
    for j in [0, half_m]:
        for k in [0, half_n]:
            sub_a = a[j:j+half_m, k:k+half_n]
            sub_b = b[j:j+half_m, k:k+half_n]
            sub_c = c[j:j+half_m, k:k+half_n]

            # 去均值后的子块
            q = sub_c - np.nanmean(sub_c.reshape(-1))
            if np.all(np.isnan(q)):
                continue
            q[np.isnan(q)] = np.nanmedian(q)

            # 判断是否平滑
            if (np.nanmax(q) - np.nanmin(q) <= r) and m <= up and n <= up:
                pz = sub_c.copy()
                px = sub_a.copy()
                py = sub_b.copy()

                # 如果 NaN 太多，跳过
                if np.isnan(pz).sum() >= pz.size * 0.5:
                    continue

                # 拉直为 1D 向量
                pz = pz.reshape(-1)
                px = px.reshape(-1)
                py = py.reshape(-1)

                # 找到中位数及对应点
                ztmp = np.nanmedian(pz)
                iztmp = np.nanargmin(np.abs(pz - ztmp))
                xtmp, ytmp = px[iztmp], py[iztmp]

                x.append(xtmp)
                y.append(ytmp)
                z.append(ztmp)
                nn.append(np.count_nonzero(~np.isnan(pz)))
                rects.append([sub_a.min(), sub_a.max(), sub_b.min(), sub_b.max()])
            else:
                # 递归继续划分
                x0, y0, z0, n0, rects = quatree_median(sub_a, sub_b, sub_c, r, up, down, rects)
                x.extend(x0)
                y.extend(y0)
                z.extend(z0)
                nn.extend(n0)

    return np.array(x), np.array(y), np.array(z), np.array(nn), rects

def subsample(
    path: str,
    file: str,
    strain: float,
    uplimit: int,
    downlimit: int,
    method: Optional[int] = None,   # 1 mean, 2 median, 3 distance-based
    segments=None,
    epicenter=None,
    write_or_not: bool = True,
    plot_or_not: bool = True,
    verbose: bool = True
):
    """
    Subsample a .grd file using quadtree variants.

    Parameters
    ----------
    path, file : str
        directory and filename (without .grd)
    strain : float
        quadtree 'r' parameter (controls subdivision)
    uplimit, downlimit : int
        quadtree up/down limits
    method : int
        1 -> quatree (mean)
        2 -> quatree_median
        3 -> quatree_distance (requires segments)
        4 -> quatree_distance (require epicenter)
    segments : list or None
    required when method==3
    write_or_not : bool
    plot_or_not : bool
    verbose : bool
    """
    # compatibility: some calls pass 'type' named arg
    if method is None:
        # try to accept 'type' from caller if present in locals (compat hint)
        method = 1

    fname = f"{path}/{file}.grd"
    xvec, yvec, zz, other = Read_grd_file(fname)

    # meshgrid: note shapes (ny, nx)
    xx, yy = np.meshgrid(xvec, yvec)   # xx.shape == (len(yvec), len(xvec))

    # ensure zz orientation matches xx/yy:
    if zz.shape == xx.shape:
        Z = zz.copy()
    elif zz.T.shape == xx.shape:
        Z = zz.T.copy()
        if verbose:
            print("Info: transposed 'zz' to match meshgrid orientation.")
    else:
        # try to broadcast/reshape: raise informative error if mismatch
        raise ValueError(
            f"Shape mismatch: meshgrid xx/yy shape {xx.shape} but zz shape {zz.shape}. "
            "Check Read_grd_file output ordering (nx vs ny)."
        )

    # call the chosen quadtree routine (note the correct argument orders)
    if verbose:
        print("Quatree sampling: method =", method)

    if int(method) == 1:
        # quatree returns (x_out,y_out,z_out,nn_out,rects)
        x_out, y_out, z_out, nn_out, rects = quatree(xx, yy, Z, r=strain, up=uplimit, down=downlimit)
    elif int(method) == 2:
        x_out, y_out, z_out, nn_out, rects = quatree_median(xx, yy, Z, r=strain, up=uplimit, down=downlimit)
    elif int(method) == 3:
        if segments is None:
            raise ValueError("segments must be provided when using distance-based quatree (method == 3).")
        # quatree_distance signature: (a,b,c,segments,up,down,...)
        x_out, y_out, z_out, nn_out, rects = quatree_distance(xx, yy, Z, segments, uplimit, downlimit,strain,
                                                dist_method="blocked", block_size=200_000, res=1.0, rects=None,
                                                dist_mode="fault", epicenter=None, hybrid_mode="min")
    elif int(method) == 4:
        if epicenter is None:
            raise ValueError("epicenter must be provided when using distance-based quatree (method == 4).")
        x_out, y_out, z_out, nn_out, rects = quatree_distance(xx, yy, Z, None, uplimit, downlimit,strain,
                                                dist_method="blocked", block_size=200_000, res=1.0, rects=None,
                                                dist_mode="epicenter", epicenter=epicenter, hybrid_mode="min")
    else:
        raise ValueError("Unknown method: choose 1(mean),2(median),3(distance)")

    # ensure numpy arrays
    x_out = np.asarray(x_out, dtype=float)
    y_out = np.asarray(y_out, dtype=float)
    z_out = np.asarray(z_out, dtype=float)
    nn_out = np.asarray(nn_out, dtype=float)

    # safe weight: 1/sqrt(nn), but avoid division by zero
    with np.errstate(divide="ignore", invalid="ignore"):
        n_out = np.where(nn_out > 0, 1.0 / np.sqrt(nn_out), np.inf)

    if plot_or_not:
        fig, ax = plt.subplots(figsize=(8, 6))
        numcol = 200
        cmap = plt.get_cmap("jet", numcol)
        clim = [-np.max(np.abs(z_out)), np.max(np.abs(z_out))]
        sc = ax.scatter(
            x_out, y_out,
            c=z_out, cmap=cmap,
            vmin=clim[0], vmax=clim[1],
            s=12, marker="o", edgecolors='k', linewidth=0.2
        )
        ax.set_aspect("equal")
        ax.set_title("Quadtree Subsampled Data", fontsize=14)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        cbar = plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("z value")
        plt.show()

    if verbose:
        print(f"Number of subsampled data: {len(x_out)}")

    # write to file (llde format in your code)
    if write_or_not:
        outfile = f"{path}/{file}.llde"
        # ensure same length
        L = min(len(x_out), len(y_out), len(z_out), len(n_out))
        with open(outfile, "w") as f:
            for j in range(L):
                f.write(f"{x_out[j]:.9f}\t{y_out[j]:.9f}\t{z_out[j]:.9f}\t{n_out[j]:.9f}\n")
        if verbose:
            print(f"Subsampled data saved to {outfile}")

    return x_out, y_out, z_out, n_out, rects

from scipy.ndimage import distance_transform_edt

try:
    from shapely.geometry import LineString, Point
    from shapely.strtree import STRtree
    _has_shapely = True
except ImportError:
    _has_shapely = False


# ------------------------
# 底层工具函数
# ------------------------
# ------------  Calculate the distance grid (regarding to epicenter) ----------
def compute_distance_grid_epicenter(X, Y, epicenter):
    """
    计算每个网格点到震中的欧氏距离
    epicenter : (x0, y0)
    """
    x0, y0 = epicenter
    return np.sqrt((X - x0)**2 + (Y - y0)**2)

def _point_to_segments_min_distance(px, py, segments):
    """暴力计算每个点到所有断层段的最小距离 (向量化)。"""
    d_min = np.full(px.shape, np.inf)
    for (x1, y1, x2, y2) in segments:
        vx, vy = x2 - x1, y2 - y1
        wx, wy = px - x1, py - y1
        c1 = vx * wx + vy * wy
        c2 = vx * vx + vy * vy
        t = np.clip(c1 / c2, 0.0, 1.0)
        proj_x = x1 + t * vx
        proj_y = y1 + t * vy
        d = np.sqrt((px - proj_x) ** 2 + (py - proj_y) ** 2)
        d_min = np.minimum(d_min, d)
    return d_min


def compute_distance_grid_blocked(X, Y, segments, block_size=200_000):
    """分块计算，适合大网格，纯 numpy 实现"""
    px = X.ravel().astype(float)
    py = Y.ravel().astype(float)
    N = len(px)
    dist_min = np.full(N, np.inf, dtype=float)

    for i in range(0, N, block_size):
        px_block = px[i:i + block_size]
        py_block = py[i:i + block_size]
        d_block = _point_to_segments_min_distance(px_block, py_block, segments)
        dist_min[i:i + block_size] = d_block
    return dist_min.reshape(X.shape)


def compute_distance_grid_shapely(X, Y, segments):
    """基于 shapely 的空间索引 (STRtree)"""
    if not _has_shapely:
        raise ImportError("Shapely 未安装，请使用 `pip install shapely`")
    lines = [LineString([(x1, y1), (x2, y2)]) for (x1, y1, x2, y2) in segments]
    tree = STRtree(lines)

    px = X.ravel().astype(float)
    py = Y.ravel().astype(float)
    dist_min = np.zeros_like(px, dtype=float)

    for i, (xi, yi) in enumerate(zip(px, py)):
        p = Point(xi, yi)
        nearest = tree.nearest(p)
        dist_min[i] = p.distance(nearest)

    return dist_min.reshape(X.shape)


def compute_distance_grid_raster(X, Y, segments, res=1.0):
    """基于栅格 + 距离变换的快速近似法"""
    nx, ny = X.shape[1], X.shape[0]
    mask = np.zeros((ny, nx), dtype=bool)

    for (x1, y1, x2, y2) in segments:
        ix1 = int((x1 - X.min()) / res)
        iy1 = int((y1 - Y.min()) / res)
        ix2 = int((x2 - X.min()) / res)
        iy2 = int((y2 - Y.min()) / res)
        if 0 <= ix1 < nx and 0 <= iy1 < ny:
            mask[iy1, ix1] = True
        if 0 <= ix2 < nx and 0 <= iy2 < ny:
            mask[iy2, ix2] = True

    dist_pix = distance_transform_edt(~mask) * res
    return dist_pix


# ------------------------
# 主函数
# ------------------------
def quatree_distance(
    a, b, c, segments,
    up, down, p_thresh,
    dist_method="blocked",
    block_size=200_000,
    res=1.0,
    rects=None,
    dist_mode="fault",         
    epicenter=None,           
    hybrid_mode="min"        
):
    """
    基于距断层的高斯函数的 Quadtree 降采样.

    参数
    ----
    a, b : 2D 网格坐标 (meshgrid)
    c    : 2D 数据
    segments : [(x1,y1,x2,y2), ...] 断层段
    p_thresh : 0-1，越小，越聚集于近断层。
    up, down : 与原 quatree 相同的上限、下限
    dist_method : {"blocked", "shapely", "raster"}
        blocked : 分块 brute-force (默认, 稳健)
        shapely : 空间索引加速 (需安装 shapely)
        raster  : 二值栅格 + 距离变换 (近似最快)
    block_size : blocked 方法的块大小
    res : raster 方法的分辨率
    dist_mode : "fault" | "epicenter" | "hybrid"
    epicenter : Lon/lat, UTM local
    hybrid_mode : "min" | "product"
    返回
    ----
    x, y, z, nn : 采样结果
    """
    if rects is None:
        rects = []

    x_list, y_list, z_list, nn_list = [], [], [], []
    m, n = c.shape

    if np.isnan(c).sum() >= m * n * 0.99:
        return np.array(x_list), np.array(y_list), np.array(z_list), np.array(nn_list), rects

    if m <= down or n <= down:
        x_list.append(np.nanmean(a.reshape(-1)))
        y_list.append(np.nanmean(b.reshape(-1)))
        z_list.append(np.nanmean(c.reshape(-1)))
        nn_list.append(np.count_nonzero(~np.isnan(c)))
        rects.append([a.min(), a.max(), b.min(), b.max()])
        return np.array(x_list), np.array(y_list), np.array(z_list), np.array(nn_list), rects

    # ---- 原始：断层距离 ----
    if dist_mode == "fault":
        if dist_method == "blocked":
            dist_grid = compute_distance_grid_blocked(a, b, segments, block_size)
        elif dist_method == "shapely":
            dist_grid = compute_distance_grid_shapely(a, b, segments)
        elif dist_method == "raster":
            dist_grid = compute_distance_grid_raster(a, b, segments, res=res)
        else:
            raise ValueError(f"未知的 dist_method: {dist_method}")

    # ---- 新增：震中距离 ----
    elif dist_mode == "epicenter":
        if epicenter is None:
            raise ValueError("dist_mode='epicenter' 时必须提供 epicenter=(x0,y0)")
        dist_grid = compute_distance_grid_epicenter(a, b, epicenter)

    # ---- 混合模式（可选，但很有用） ----
    elif dist_mode == "hybrid":
        if epicenter is None:
            raise ValueError("dist_mode='hybrid' 时必须提供 epicenter")

        dist_fault = compute_distance_grid_blocked(a, b, segments, block_size)
        dist_epi   = compute_distance_grid_epicenter(a, b, epicenter)

        if hybrid_mode == "min":
            dist_grid = np.minimum(dist_fault, dist_epi)
        elif hybrid_mode == "product":
            dist_grid = np.sqrt(dist_fault * dist_epi)
        else:
            raise ValueError(f"未知的 hybrid_mode: {hybrid_mode}")

    else:
        raise ValueError(f"未知的 dist_mode: {dist_mode}")

        # 距离归一化 0~1
    dref = float(np.nanpercentile(dist_grid, 99))  # 99% 数据的参考距离，去除一些outlier
    dist_norm = np.clip(dist_grid / (dref + 1e-12), 0, 1)
    # 高斯映射到采样概率 0~1
    # 这里 0.5 是归一化尺度，
    prob = np.exp(-0.5 * (dist_norm / 0.5)**2)

    # 用块内 90% percentile 判断是否靠近断层
    block_score = float(np.nanpercentile(prob, 90))


    # 判断逻辑：远场 -> 合并，近场 -> 细分
    if block_score < p_thresh :  # 阈值，可调，判断是否继续采样
        # 如果太多 NaN，跳过
        if np.isnan(c).sum() >= c.size * 0.75:
            return np.array(x_list), np.array(y_list), np.array(z_list), np.array(nn_list), rects
        # 作为叶节点输出平均值
        a_copy = a.astype(float).copy()
        b_copy = b.astype(float).copy()
        a_copy[np.isnan(c)] = np.nan
        b_copy[np.isnan(c)] = np.nan
        x_list.append(float(np.nanmean(a_copy.reshape(-1))))
        y_list.append(float(np.nanmean(b_copy.reshape(-1))))
        z_list.append(float(np.nanmean(c.reshape(-1))))
        nn_list.append(int(np.count_nonzero(~np.isnan(c))))
        rects.append([float(np.nanmin(a)), float(np.nanmax(a)), float(np.nanmin(b)), float(np.nanmax(b))])
        return np.array(x_list), np.array(y_list), np.array(z_list), np.array(nn_list), rects

    # 否则 mean_prob >= p_thresh => 说明该块“靠近断层”，需要细分（如果可能）
    # 划分为4个子块：使用 mid 切分，保证覆盖所有元素（适用于奇偶维度）
    mid_m = m // 2
    mid_n = n // 2
    # 子块索引对： (row_start,row_end), (col_start,col_end)
    row_slices = [(0, mid_m), (mid_m, m)]
    col_slices = [(0, mid_n), (mid_n, n)]

    for (rs, re) in row_slices:
        for (cs, ce) in col_slices:
            # 子块可能为空（当 mid==0 或其它边界时），跳过
            if rs >= re or cs >= ce:
                continue
            sub_a = a[rs:re, cs:ce]
            sub_b = b[rs:re, cs:ce]
            sub_c = c[rs:re, cs:ce]
            sub_dist = dist_grid[rs:re, cs:ce] if dist_grid is not None else None

            x0, y0, z0, n0, rects = quatree_distance(
                sub_a, sub_b, sub_c, segments, up, down,p_thresh,
                dist_method=dist_method, block_size=block_size, res=res,
                rects=rects,dist_mode=dist_mode,epicenter=epicenter,hybrid_mode= hybrid_mode)
                
            # 合并子结果
            if x0.size:
                x_list.extend(x0.tolist())
                y_list.extend(y0.tolist())
                z_list.extend(z0.tolist())
                nn_list.extend(n0.tolist())

    return np.array(x_list), np.array(y_list), np.array(z_list), np.array(nn_list), rects

def plot_quatree_process(x, y, data, x_out, y_out, z_out, rects, segments=None, epicenter = None):
    """
    visualize the process of Quadtree subsampling:
    - left: figure: Origin data matrix data with local coordinates x and y;
    - right: subsampling (colorful scatter) + quatree partition subsampling
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)

    # left : origin data distribution
    im = axes[0].imshow(
        data, origin="lower", cmap="jet",
        extent=[x.min(), x.max(), y.min(), y.max()]
    )
    axes[0].set_title("Original Data")
    axes[0].set_xlabel("X")
    axes[0].set_ylabel("Y")
    fig.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04, label="z value")

    # right : rectangle partition + plot scatter
    numcol = 200
    cmap = plt.get_cmap("jet", numcol)
    clim = [-np.max(np.abs(z_out)), np.max(np.abs(z_out))]

    sc = axes[1].scatter(
        x_out, y_out,
        c=z_out, cmap=cmap,
        vmin=clim[0], vmax=clim[1],
        s=12, marker="o", edgecolor="k", linewidth=0.2
    )

    # rectangle partition
    for rect in rects:
        xmin, xmax, ymin, ymax = rect
        axes[1].plot(
            [xmin, xmax, xmax, xmin, xmin],
            [ymin, ymin, ymax, ymax, ymin],
            "k-", lw=0.5
        )
    # plot fault trace 
    if segments is not None:
        for (x1, y1, x2, y2) in segments:
            axes[1].plot([x1, x2], [y1, y2], "r-", lw=1.2, label="Fault Trace")

    if epicenter is not None:
        axes[1].plot(
            epicenter[0],
            epicenter[1],
            marker="*",
            color="r",
            markersize=12,   # ← 正确
            zorder=10)
    
    axes[1].set_aspect("equal")
    axes[1].set_title("Quadtree Sampling Process", fontsize=14)
    axes[1].set_xlabel("Longitude")
    axes[1].set_ylabel("Latitude")
    fig.colorbar(sc, ax=axes[1], fraction=0.046, pad=0.04, label="z value")

    plt.show()

## How to subsample data
if __name__ == "__main__":
    cfg = parse_config("../config_ridgecrest.inv")
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
            if patch["strike"] == 90:
                patch["strike"] = 89.9
            patches.append(patch)
    # segments = get_patch_endpoints(patches);
    # print(segments)
    # path = "./"
    # x_out, y_out, z_out, n_out,rects = subsample("./", "data_asc", strain=5, uplimit=1000, downlimit=2,method = 1,
    #                                     segments = segments, write_or_not=False, plot_or_not=True)

