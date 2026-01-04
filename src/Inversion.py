
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.ticker import PercentFormatter
from scipy.stats import norm
from src import plot_data, utm2ll,plot_trace

import numpy as np

def generate_smoothness(xs, ys, zs, lens, wid, strike,
                           num_grid, fault_type, fdip,
                           sseg, sid, ssid, top, bot, side, sideid,
                           plt=False):
    """
    Python translation of MATLAB generate_smoothness (full).
    Inputs:
      xs, ys, zs, lens, wid, strike : arrays length = total_patches (may be column vectors)
      num_grid : array length = number_of_segments (counts patches per segment)
      fault_type : array-like (e.g. [1,1,0])
      fdip, sseg, sid, ssid, top, bot, side, sideid : same meaning as MATLAB inputs
    Returns:
      smooth : numpy.ndarray of shape (n_rows, n_cols) exactly matching MATLAB row ordering.
    """
    # --- normalize inputs to 1D arrays ---
    xs = np.asarray(xs).ravel()
    ys = np.asarray(ys).ravel()
    zs = np.asarray(zs).ravel()
    lens = np.asarray(lens).ravel()   # 'len' in MATLAB
    wid = np.asarray(wid).ravel()
    strike = np.asarray(strike).ravel()
    num_grid = np.asarray(num_grid).ravel().astype(int)
    fault_type = np.asarray(fault_type).ravel().astype(int)

    total_patches = int(np.sum(num_grid))
    if xs.size != total_patches:
        # allow xs to be provided in different orientation; but require length match
        assert xs.size == total_patches, "xs length must equal sum(num_grid)"
    ncomp = int(np.sum(fault_type))   # e.g. 2 for [1,1,0]
    ncols = total_patches * ncomp

    # If no components, return empty
    if ncomp == 0:
        return np.empty((0, 0), dtype=float)

    # --- 1) compute dpth as MATLAB does ---
    Layer = 100
    dpth = np.zeros(total_patches, dtype=float)
    dpth[0] = Layer
    for j in range(1, total_patches):
        if zs[j] == zs[j-1]:
            dpth[j] = dpth[j-1]
        elif zs[j] > zs[j-1]:
            dpth[j] = dpth[j-1] + 1
        else:
            Layer += 100
            dpth[j] = Layer

    # --- renumber per fault group as MATLAB ---
    N_fault_groups = int(Layer // 100)
    for g in range(1, N_fault_groups + 1):
        mask = (dpth >= g*100) & (dpth < (g+1)*100)
        if np.any(mask):
            maxv = np.max(dpth[mask])
            dpth[mask] = -(dpth[mask] - maxv - 1)
    id_arr = dpth.astype(int)   # this is MATLAB's 'id'

    # --- create in (1..N) and ip (segment index 1..nseg) exactly as MATLAB ---
    in_list = []
    ip_list = []
    lcount = 0
    for seg_idx in range(1, len(num_grid) + 1):   # seg_idx is 1-based like MATLAB
        ngrid = int(num_grid[seg_idx-1])
        for _ in range(ngrid):
            lcount += 1
            in_list.append(lcount)    # 1-based patch id
            ip_list.append(seg_idx)   # 1-based segment id
    in_arr = np.array(in_list, dtype=int)   # length total_patches, values 1..total_patches
    ip_arr = np.array(ip_list, dtype=int)   # length total_patches, values 1..n_segments

    layer_max = int(np.max(id_arr))

    # container for rows
    rows = []

    # small helper to append a dense row
    def append_row_at(entries):
        # entries: sequence of (col_index_zero_based, value)
        row = np.zeros(ncols, dtype=float)
        for cidx, val in entries:
            row[cidx] = val
        rows.append(row)

    # --- A. horizontal smoothness within a segment (along strike) ---
    if fault_type.size > 0 and fault_type[0] != 0:
        for j in range(total_patches - 1):   # j is python 0-based, corresponds to MATLAB j
            if id_arr[j+1] == id_arr[j]:
                denom = lens[j] + lens[j+1]
                if denom == 0:
                    continue
                # strike component (component index 0)
                append_row_at([
                    ((j+1) * ncomp + 0, -2.0 / denom),
                    (j * ncomp + 0,  2.0 / denom)
                ])
                # dip component (component index 1) if exists
                if ncomp >= 2:
                    append_row_at([
                        ((j+1) * ncomp + 1, -2.0 * fdip / denom),
                        ( j    * ncomp + 1,  2.0 * fdip / denom)
                    ])
    # print("smoothness's shape:",np.shape(rows))
                # plotting omitted (plt)
    # print intermediate shape
    # print("after horizontal within: rows =", len(rows))

    # --- B. vertical smoothness within a segment (layer contacts) ---
    if fault_type.size > 1 and fault_type[1] != 0 and fdip != 0:
        for seg_j in range(1, len(num_grid) + 1):   # seg_j is 1-based like MATLAB
            for k_layer in range(layer_max, 1, -1):  # k from layer down to 2
                i1 = np.where((id_arr == k_layer) & (ip_arr == seg_j))[0]   # python indices (0-based)
                i2 = np.where((id_arr == (k_layer - 1)) & (ip_arr == seg_j))[0]
                if i1.size == 0 or i2.size == 0:
                    continue
                # lens[i1[0]] is the base length for dividing along strike
                L1 = float(lens[i1[0]])
                L2 = float(lens[i2[0]])
                for l_idx, ind1 in enumerate(i1):
                    x11 = L1 * (l_idx)      # (l-1) in MATLAB where l starts from 1
                    x12 = L1 * (l_idx + 1)
                    for m_idx, ind2 in enumerate(i2):
                        x21 = L2 * (m_idx)
                        x22 = L2 * (m_idx + 1)
                        # overlap test
                        if (x21 < x12) and (x22 > x11):
                            denom = wid[ind2] + wid[ind1]
                            if denom == 0:
                                continue
                            # strike comp
                            append_row_at([
                                (ind2 * ncomp + 0, -2.0 / denom),
                                (ind1 * ncomp + 0,  2.0 / denom)
                            ])
                            # dip comp
                            if ncomp >= 2:
                                append_row_at([
                                    (ind2 * ncomp + 1, -2.0 * fdip / denom),
                                    (ind1 * ncomp + 1,  2.0 * fdip / denom)
                                ])
    # print("after vertical within: rows =", len(rows))
    # print("smoothness's shape:",np.shape(rows))

    # --- C. horizontal smoothness between segments (sid) ---
    if (sseg != 0) and (sid is not None) and len(sid) > 0:
        sid_arr = np.asarray(sid, dtype=int)
        for row_idx in range(sid_arr.shape[0]):
            seg1 = int(sid_arr[row_idx, 0])   # MATLAB 1-based
            seg2 = int(sid_arr[row_idx, 1])   # 1-based
            for j_layer in range(1, layer_max + 1):  # j from 1..layer
                i1_cand = np.where((ip_arr == seg1) & (id_arr == j_layer))[0]
                i2_cand = np.where((ip_arr == seg2) & (id_arr == j_layer))[0]
                if i1_cand.size == 0 or i2_cand.size == 0:
                    continue
                i1 = int(i1_cand[-1])   # last
                i2 = int(i2_cand[0])    # first
                dist = np.hypot(xs[i1] - xs[i2], ys[i1] - ys[i2])
                if dist == 0:
                    continue
                append_row_at([
                    (i1 * ncomp + 0, -2.0 / dist),
                    (i2 * ncomp + 0,  2.0 / dist)
                ])
                if ncomp >= 2:
                    append_row_at([
                        (i1 * ncomp + 1, -2.0 * fdip / dist),
                        (i2 * ncomp + 1,  2.0 * fdip / dist)
                    ])
    # print("after sid: rows =", len(rows))
    # print("smoothness's shape:",np.shape(rows))

    # --- D. ssid special intersections ---
    if (ssid is not None) and len(ssid) > 0:
        ssid_arr = np.asarray(ssid, dtype=int)
        for row_idx in range(ssid_arr.shape[0]):
            a_seg = int(ssid_arr[row_idx, 0])
            b_seg = int(ssid_arr[row_idx, 1])
            where_type = int(ssid_arr[row_idx, 2])   # 1 means start, 2 means end
            for j_layer in range(1, layer_max + 1):
                # select i1
                cand1 = np.where((ip_arr == a_seg) & (id_arr == j_layer))[0]
                if cand1.size == 0:
                    continue
                if where_type == 2:
                    # last
                    i1_idx = int(cand1[-1])
                elif where_type == 1:
                    i1_idx = int(cand1[0])
                else:
                    continue
                cand2 = np.where((ip_arr == b_seg) & (id_arr == j_layer))[0]
                if cand2.size == 0:
                    continue
                # choose nearest according to MATLAB logic
                if where_type == 2:
                    # distance to the *end* of i1 patch: subtract len* sin/cos components
                    dx = xs[cand2] - (xs[i1_idx] + lens[i1_idx] * np.sin(strike[i1_idx]))
                    dy = ys[cand2] - (ys[i1_idx] + lens[i1_idx] * np.cos(strike[i1_idx]))
                    arr = dx*dx + dy*dy
                else:
                    dx = xs[cand2] - xs[i1_idx]
                    dy = ys[cand2] - ys[i1_idx]
                    arr = dx*dx + dy*dy
                i2_rel = int(np.argmin(arr))
                i2_idx = int(cand2[i2_rel])
                dist = np.hypot(xs[i1_idx] - xs[i2_idx], ys[i1_idx] - ys[i2_idx])
                if dist == 0:
                    continue
                append_row_at([
                    (i1_idx * ncomp + 0, -2.0 / dist),
                    (i2_idx * ncomp + 0,  2.0 / dist)
                ])
    # print("after ssid: rows =", len(rows))
    # print("smoothness's shape:",np.shape(rows))
    # --- E. edge constraints (top) ---
    ed_weigh = 10.0
    if top != 0:
        ii = np.where(id_arr == 1)[0]   # patches at top
        for idx in ii:
            denom = wid[int(idx)]
            if denom == 0:
                continue
            append_row_at([(int(idx) * ncomp + 0,  2.0 / denom * ed_weigh)])
            if ncomp >= 2:
                append_row_at([(int(idx) * ncomp + 1,  2.0 / denom * ed_weigh)])
    # print("smoothness's shape:",np.shape(rows))
    # --- F. edge constraints (bottom) ---
    if bot != 0:
        ii = np.where(id_arr == layer_max)[0]
        for idx in ii:
            denom = wid[int(idx)]
            if denom == 0:
                continue
            append_row_at([(int(idx) * ncomp + 0,  2.0 / denom * ed_weigh)])
            if ncomp >= 2:
                append_row_at([(int(idx) * ncomp + 1,  2.0 / denom * ed_weigh)])
    # print("smoothness's shape:",np.shape(rows))
    # --- G. side constraints (sideid) ---
    if side != 0 and (sideid is not None) and len(sideid) > 0:
        sideid_arr = np.asarray(sideid, dtype=int)
        for j_layer in range(1, layer_max + 1):
            for row_idx in range(sideid_arr.shape[0]):
                seg = int(sideid_arr[row_idx, 0])
                first_flag = int(sideid_arr[row_idx, 1])
                last_flag  = int(sideid_arr[row_idx, 2])

                # MATLAB 原逻辑中第三列为0时有个 else continue
                if last_flag == 0:
                    continue

                ii = []  # **每个 k 都重置**
                cand = np.where((ip_arr == seg) & (id_arr == j_layer))[0]
                if cand.size == 0:
                    continue

                if first_flag != 0:
                    ii.append(int(cand[0]))
                if last_flag != 0:
                    ii.append(int(cand[-1]))

                # print([i+1 for i in ii])
                ii.append(int(cand[-1]))
                # Print like MATLAB (1-based indices) for direct comparison
    #             print([i + 1 for i in ii])
                # Debug print similar to MATLAB disp(ii) but convert to 1-based for readability if desired
                # print([ii+1 for ii in idxs])  # optional: show 1-based indices
                for ii_idx in ii:
                    # matlab used 2/len(ii)*ed_weigh without zero-check => we mimic that but safer:
                    denom_len = float(lens[ii_idx]) if lens[ii_idx] != 0 else np.nan
                    if np.isnan(denom_len) or denom_len == 0.0:
                        # emulate MATLAB's potential Inf or raise — here we skip to be safe
                        # If you want exact MATLAB behavior, remove the continue and let ZeroDivisionError happen.
                        continue

                    # --- FIRST component: column index = (ii-1)*ncomp + 0  (zero-based: ii_idx*ncomp + 0)
                    first_col = ii_idx * ncomp + 0
                    append_row_at([(first_col, 2.0 / denom_len * ed_weigh)])

                    # --- LAST component: column index = ii*ncomp - 1   (zero-based: ii_idx*ncomp + (ncomp-1))
                    last_col = ii_idx * ncomp + (ncomp - 1)
                    append_row_at([(last_col, 2.0 / denom_len * ed_weigh)])
    # finalize matrix just like you had
    if len(rows) == 0:
        smooth = np.empty((0, ncols), dtype=float)
    else:
        smooth = np.vstack(rows).astype(float)

        # final shape
        # print("Smoothness's shape:", smooth.shape)
    return smooth

def smoothness_upgraded(
    xs, ys, zs, len_arr, wid, strike, num_grid,
    fault_type, fdip, sseg, sid, ssid, top, bot, side, sideid,
    plt_flag=0, mode=1
):
    """
    Python / numpy translation of your MATLAB smoothness_upgraded.
    Inputs:
      xs, ys, zs, len_arr, wid, strike : 1D array-like, length = total_patches
      num_grid : iterable of ints, length = number of segments, sum(num_grid) = total_patches
      fault_type : iterable (e.g. [1,1,0]) -> sum gives ncomp (components per patch)
      fdip : scalar
      sseg : scalar (0 or 1)
      sid : array-like shape (k,2) or None  (MATLAB used 1-based indices)
      ssid: array-like shape (k,3) or None  (MATLAB used 1-based indices)
      top, bot, side : scalars (0/1)
      sideid: array-like shape (m,3) or None (MATLAB style)
      plt_flag : if 1 draw 3D lines (requires matplotlib)
      mode : 1 or 2 (first or second order)
    Returns:
      smooth : numpy array shape (M, total_vars) where total_vars = ncomp * total_patches
    Notes:
      - The function assumes sid/ssid/sideid indices are MATLAB 1-based. It converts them to 0-based.
      - Keeps the coefficient formulas exactly as in your MATLAB code.
    """
    xs = np.asarray(xs).ravel()
    ys = np.asarray(ys).ravel()
    zs = np.asarray(zs).ravel()
    len_arr = np.asarray(len_arr).ravel()
    wid = np.asarray(wid).ravel()
    strike = np.asarray(strike).ravel()
    num_grid = np.asarray(num_grid).ravel().astype(int)
    fault_type = np.asarray(fault_type).ravel()
    ncomp = int(np.sum(fault_type))  # components per patch (e.g., 2)
    if ncomp < 1:
        raise ValueError("fault_type must indicate at least one slip component")

    total_patches = int(np.sum(num_grid))
    if xs.size != total_patches:
        raise ValueError("xs/ys/zs/len/wid/strike must have length = sum(num_grid)")

    # --- build depth mapping (dpth/id) same as MATLAB ---
    Layer = 100
    dpth = np.empty(total_patches, dtype=float)
    dpth[0] = Layer
    for j in range(1, total_patches):
        if zs[j] == zs[j-1]:
            dpth[j] = dpth[j-1]
        elif zs[j] > zs[j-1]:
            dpth[j] = dpth[j-1] + 1
        else:
            Layer += 100
            dpth[j] = Layer

    Nfaults = Layer // 100
    # remap each group to small positive integer indices (as MATLAB did)
    for j in range(1, Nfaults + 1):
        mask = (dpth >= j*100) & (dpth < (j+1)*100)
        if np.any(mask):
            group = dpth[mask]
            dpth[mask] = - (group - np.max(group) - 1)

    id_arr = dpth.astype(int)  # id for each patch (1..layer)
    layer = np.max(id_arr)     # number of vertical layers

    # --- build ip (segment index for each patch), zero-based patch indices 0..N-1 ---
    ip = np.empty(total_patches, dtype=int)
    idx = 0
    for seg_idx, ng in enumerate(num_grid):
        for _ in range(int(ng)):
            ip[idx] = seg_idx  # zero-based segment index
            idx += 1

    # prepare container for rows (we'll collect rows as lists of numpy arrays)
    smooth_rows = []
    total_vars = total_patches * ncomp
    ed_weigh = 10.0

    # optional plotting
    if plt_flag == 1:
        try:
            from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
            import matplotlib.pyplot as plt
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
        except Exception:
            ax = None
    else:
        ax = None

    # --- horizontal smoothness within segment ---
    if int(fault_type[0]) != 0:  # MATLAB: fault_type(1) ~= 0
        # mode 1 (first-order) between adjacent patches
        if mode == 1:
            for j in range(total_patches - 1):
                if id_arr[j+1] == id_arr[j]:
                    denom = (len_arr[j] + len_arr[j+1])
                    if denom == 0:
                        continue
                    col1 = np.zeros(total_vars, dtype=float)
                    # patch j+1 strike comp -> negative
                    if ncomp >= 1:
                        col1[(j+1)*ncomp + 0] = -2.0 / denom
                        col1[j*ncomp + 0]     =  2.0 / denom
                        smooth_rows.append(col1)
                    if ncomp >= 2:
                        col2 = np.zeros(total_vars, dtype=float)
                        col2[(j+1)*ncomp + 1] = -2.0 * fdip / denom
                        col2[j*ncomp + 1]     =  2.0 * fdip / denom
                        smooth_rows.append(col2)
                    if ax is not None:
                        ax.plot([xs[j+1], xs[j]], [ys[j+1], ys[j]], [zs[j+1], zs[j]], color='k')

        # mode 2 (second-order) across three consecutive patches
        if mode == 2:
            for j in range(total_patches - 2):
                if id_arr[j+2] == id_arr[j]:
                    denom1 = (len_arr[j+2] + len_arr[j+1])
                    denom2 = (len_arr[j+2] + 2*len_arr[j+1] + len_arr[j])
                    denom3 = (len_arr[j+1] + len_arr[j])
                    # strike-like row (NOTE: original matlab multiplies by fdip here; we keep it)
                    if ncomp >= 1:
                        col = np.zeros(total_vars, dtype=float)
                        col[(j+2)*ncomp + 0] = -2.0 * fdip / denom1
                        col[(j+1)*ncomp + 0] =  8.0 * fdip / denom2
                        col[(j)*ncomp     + 0] = -2.0 * fdip / denom3
                        smooth_rows.append(col)
                    # dip row
                    if ncomp >= 2:
                        col = np.zeros(total_vars, dtype=float)
                        col[(j+2)*ncomp + 1] = -2.0 * fdip / denom1
                        col[(j+1)*ncomp + 1] =  8.0 * fdip / denom2
                        col[(j)*ncomp     + 1] = -2.0 * fdip / denom3
                        smooth_rows.append(col)
                    if ax is not None:
                        ax.plot([xs[j+2], xs[j+1]], [ys[j+2], ys[j+1]], [zs[j+2], zs[j+1]], color='b')
                        ax.plot([xs[j+1], xs[j]],     [ys[j+1], ys[j]],     [zs[j+1], zs[j]],     color='g')

    # --- vertical smoothness within segment (requires dip component present) ---
    if len(fault_type) >= 2 and int(fault_type[1]) != 0 and fdip != 0:
        # iterate through segments
        for seg_j in range(len(num_grid)):
            # mode 2 (three-layer Laplace)
            if mode == 2:
                # k goes from layer down to 3 (MATLAB inclusive)
                for k in range(layer, 2, -1):  # k = layer, layer-1, ..., 3
                    i1 = np.where((id_arr == k) & (ip == seg_j))[0]
                    i2 = np.where((id_arr == k-1) & (ip == seg_j))[0]
                    i3 = np.where((id_arr == k-2) & (ip == seg_j))[0]
                    if i1.size == 0 or i2.size == 0 or i3.size == 0:
                        continue
                    # use the first element's len as representative (as in MATLAB)
                    len1 = len_arr[i1[0]]
                    len2 = len_arr[i2[0]]
                    len3 = len_arr[i3[0]]
                    for li, p1 in enumerate(i1):
                        x11 = len1 * (li)
                        x12 = len1 * (li+1)
                        for mi, p2 in enumerate(i2):
                            x21 = len2 * (mi)
                            x22 = len2 * (mi+1)
                            for ni, p3 in enumerate(i3):
                                x31 = len3 * (ni)
                                x32 = len3 * (ni+1)
                                if (x21 < x12 and x22 > x11) and (x31 < x22 and x32 > x21):
                                    # build two rows: strike and dip
                                    if ncomp >= 1:
                                        col = np.zeros(total_vars, dtype=float)
                                        col[p3*ncomp + 0] = -2.0 / (wid[p2] + wid[p3])
                                        col[p2*ncomp + 0] =  8.0 / (2*wid[p2] + wid[p1] + wid[p3])
                                        col[p1*ncomp + 0] = -2.0 / (wid[p2] + wid[p1])
                                        smooth_rows.append(col)
                                    if ncomp >= 2:
                                        col = np.zeros(total_vars, dtype=float)
                                        col[p3*ncomp + 1] = -2.0 / (wid[p2] + wid[p3])
                                        col[p2*ncomp + 1] =  8.0 / (2*wid[p2] + wid[p1] + wid[p3])
                                        col[p1*ncomp + 1] = -2.0 / (wid[p2] + wid[p1])
                                        smooth_rows.append(col)
                                    if ax is not None:
                                        ax.plot([xs[p3], xs[p2]], [ys[p3], ys[p2]], [zs[p3], zs[p2]], color='c')
                                        ax.plot([xs[p2], xs[p1]], [ys[p2], ys[p1]], [zs[p2], zs[p1]], color='y')
            # mode 1 (two-layer)
            if mode == 1:
                for k in range(layer, 1, -1):  # k = layer ... 2
                    i1 = np.where((id_arr == k) & (ip == seg_j))[0]
                    i2 = np.where((id_arr == k-1) & (ip == seg_j))[0]
                    if i1.size == 0 or i2.size == 0:
                        continue
                    len1 = len_arr[i1[0]]
                    len2 = len_arr[i2[0]]
                    for li, p1 in enumerate(i1):
                        x11 = len1 * (li)
                        x12 = len1 * (li+1)
                        for mi, p2 in enumerate(i2):
                            x21 = len2 * (mi)
                            x22 = len2 * (mi+1)
                            if x21 < x12 and x22 > x11:
                                if ncomp >= 1:
                                    col = np.zeros(total_vars, dtype=float)
                                    col[p2*ncomp + 0] = -2.0 / (wid[p2] + wid[p1])
                                    col[p1*ncomp + 0] =  2.0 / (wid[p2] + wid[p1])
                                    smooth_rows.append(col)
                                if ncomp >= 2:
                                    col = np.zeros(total_vars, dtype=float)
                                    col[p2*ncomp + 1] = -2.0 * fdip / (wid[p2] + wid[p1])
                                    col[p1*ncomp + 1] =  2.0 * fdip / (wid[p2] + wid[p1])
                                    smooth_rows.append(col)
                                if ax is not None:
                                    ax.plot([xs[p2], xs[p1]], [ys[p2], ys[p1]], [zs[p2], zs[p1]], color='b')

    # --- horizontal smoothness between segments (sid) ---
    if sseg != 0 and sid is not None and len(sid) > 0:
        sid_arr = np.asarray(sid)
        # convert 1-based sid to 0-based segment indices
        sid0 = sid_arr.astype(int) - 1
        for row in sid0:
            seg_a, seg_b = int(row[0]), int(row[1])
            for j in range(1, layer+1):
                # find last patch in seg_a with depth j and first patch in seg_b with depth j
                i1_candidates = np.where((ip == seg_a) & (id_arr == j))[0]
                i2_candidates = np.where((ip == seg_b) & (id_arr == j))[0]
                if i1_candidates.size == 0 or i2_candidates.size == 0:
                    continue
                i1 = i1_candidates[-1]
                i2 = i2_candidates[0]
                d = np.hypot(xs[i1] - xs[i2], ys[i1] - ys[i2])
                if d == 0:
                    continue
                if ncomp >= 1:
                    col = np.zeros(total_vars, dtype=float)
                    col[i1*ncomp + 0] = -2.0 / d
                    col[i2*ncomp + 0] =  2.0 / d
                    smooth_rows.append(col)
                if ncomp >= 2:
                    col = np.zeros(total_vars, dtype=float)
                    col[i1*ncomp + 1] = -2.0 * fdip / d
                    col[i2*ncomp + 1] =  2.0 * fdip / d
                    smooth_rows.append(col)
                if ax is not None:
                    ax.plot([xs[i2], xs[i1]], [ys[i2], ys[i1]], [zs[i2], zs[i1]], color='r')

        # ssid handling (intersection)
        if ssid is not None and len(ssid) > 0:
            ssid0 = np.asarray(ssid).astype(int) - 1  # convert indexing; third col (flag) will be 0 or 1 in MATLAB? careful
            # Because MATLAB ssid third column was 1 or 2, subtracting 1 makes it 0 or 1. We must compare original values:
            # So better to load raw and check original 1/2:
            ssid_raw = np.asarray(ssid).astype(int)
            for k_i, row in enumerate(ssid_raw):
                seg1 = int(row[0]) - 1
                seg2 = int(row[1]) - 1
                flag = int(row[2])  # keep original 1 or 2
                for j in range(1, layer+1):
                    if flag == 2:
                        cand = np.where((ip == seg1) & (id_arr == j))[0]
                        if cand.size == 0:
                            continue
                        i1 = cand[-1]
                    elif flag == 1:
                        cand = np.where((ip == seg1) & (id_arr == j))[0]
                        if cand.size == 0:
                            continue
                        i1 = cand[0]
                    else:
                        continue
                    i_candidates = np.where((ip == seg2) & (id_arr == j))[0]
                    if i_candidates.size == 0:
                        continue
                    # choose nearest based on two different distance metrics
                    if flag == 2:
                        # projection distance criteria used in MATLAB
                        # compute squared distances and pick minimum
                        xs_i = xs[i_candidates] - xs[i1] - len_arr[i1]*np.sin(strike[i1])
                        ys_i = ys[i_candidates] - ys[i1] - len_arr[i1]*np.cos(strike[i1])
                        d2 = xs_i**2 + ys_i**2
                        kmin = np.argmin(d2)
                        i2 = i_candidates[kmin]
                    else:
                        # flag ==1: Euclidean distance to i1
                        d2 = (xs[i_candidates] - xs[i1])**2 + (ys[i_candidates] - ys[i1])**2
                        kmin = np.argmin(d2)
                        i2 = i_candidates[kmin]
                    d = np.hypot(xs[i1] - xs[i2], ys[i1] - ys[i2])
                    if d == 0:
                        continue
                    if ncomp >= 1:
                        col = np.zeros(total_vars, dtype=float)
                        col[i1*ncomp + 0] = -2.0 / d
                        col[i2*ncomp + 0] =  2.0 / d
                        smooth_rows.append(col)
                    if ax is not None:
                        ax.plot([xs[i2], xs[i1]], [ys[i2], ys[i1]], [zs[i2], zs[i1]], color='g')

    # --- edge constraints top / bottom ---
    if top != 0:
        ii = np.where(id_arr == 1)[0]
        for p in ii:
            if ncomp >= 1:
                col = np.zeros(total_vars, dtype=float)
                col[p*ncomp + 0] = 2.0 / wid[p] * ed_weigh
                smooth_rows.append(col)
            if ncomp >= 2:
                col = np.zeros(total_vars, dtype=float)
                col[p*ncomp + 1] = 2.0 / wid[p] * ed_weigh
                smooth_rows.append(col)
            if ax is not None:
                ax.scatter(xs[p], ys[p], zs[p], marker='*', color='m')

    if bot != 0:
        ii = np.where(id_arr == layer)[0]
        for p in ii:
            if ncomp >= 1:
                col = np.zeros(total_vars, dtype=float)
                col[p*ncomp + 0] = 2.0 / wid[p] * ed_weigh
                smooth_rows.append(col)
            if ncomp >= 2:
                col = np.zeros(total_vars, dtype=float)
                col[p*ncomp + 1] = 2.0 / wid[p] * ed_weigh
                smooth_rows.append(col)
            if ax is not None:
                ax.scatter(xs[p], ys[p], zs[p], marker='*', color='m')

    # --- side constraints ---
    if side != 0 and sideid is not None and len(sideid) > 0:
        sarr = np.asarray(sideid).astype(int)
        for j in range(1, layer+1):
            for row in sarr:
                seg = int(row[0]) - 1
                entries = []
                if int(row[1]) != 0:
                    cand = np.where((ip == seg) & (id_arr == j))[0]
                    if cand.size > 0:
                        entries.append(cand[0])
                if int(row[2]) != 0:
                    cand = np.where((ip == seg) & (id_arr == j))[0]
                    if cand.size > 0:
                        entries.append(cand[-1])
                if len(entries) == 0:
                    continue
                for p in entries:
                    if ncomp >= 1:
                        col = np.zeros(total_vars, dtype=float)
                        col[p*ncomp + 0] = 2.0 / len_arr[p] * ed_weigh
                        smooth_rows.append(col)
                    if ncomp >= 2:
                        col = np.zeros(total_vars, dtype=float)
                        col[p*ncomp + 1] = 2.0 / len_arr[p] * ed_weigh
                        smooth_rows.append(col)
                if ax is not None:
                    xs_e = xs[entries]; ys_e = ys[entries]; zs_e = zs[entries]
                    ax.scatter(xs_e, ys_e, zs_e, marker='*', color='m')

    # finalize
    if len(smooth_rows) == 0:
        smooth = np.empty((0, total_vars), dtype=float)
    else:
        smooth = np.vstack(smooth_rows)

    if ax is not None:
        plt.show()

    return smooth

def generate_green_ramp(xP, yP, xA, yA, xgrd, ygrd,
                        dat_ph, dat_ph_grd, dat_az,
                        dat_gps, dat_gov, ratio):
    """
    Generate ramp matrix (ax + by + c) for InSAR/GPS data.
    Fully MATLAB-compatible indexing.
    
    Parameters
    ----------
    xP, yP : 1D array, phase points
    xA, yA : 1D array, azimuth points
    xgrd, ygrd : 1D array, gridded phase points
    dat_ph, dat_ph_grd, dat_az : list of ints
        Number of observations for each track
    dat_gps : list of ints
        Number of GPS stations
    dat_gov : list of ints
        Geological observations
    ratio : float
        Scaling factor for ramps
    
    Returns
    -------
    rmp : ndarray, shape (total_obs, 3 * n_rmp)
    """
    # ensure 1D numpy arrays
    xP = np.atleast_1d(np.asarray(xP, dtype=float))
    yP = np.atleast_1d(np.asarray(yP, dtype=float))
    xgrd = np.atleast_1d(np.asarray(xgrd, dtype=float))
    ygrd = np.atleast_1d(np.asarray(ygrd, dtype=float))
    xA = np.atleast_1d(np.asarray(xA, dtype=float))
    yA = np.atleast_1d(np.asarray(yA, dtype=float))

    # dat arrays should be 1D integer vectors (counts per track)
    dat_ph = np.atleast_1d(np.asarray(dat_ph, dtype=int))
    dat_ph_grd = np.atleast_1d(np.asarray(dat_ph_grd, dtype=int))
    dat_az = np.atleast_1d(np.asarray(dat_az, dtype=int))
    dat_gps = np.atleast_1d(np.asarray(dat_gps, dtype=int))
    dat_gov = np.atleast_1d(np.asarray(dat_gov, dtype=int))

    # Concatenate coordinates in MATLAB order: x = [xP; xgrd; xA]
    x_all = np.concatenate([xP, xgrd, xA])
    y_all = np.concatenate([yP, ygrd, yA])

    # dat_coords must be same order as x_all: dat = [dat_ph dat_ph_grd dat_az]
    dat_coords = np.concatenate([dat_ph, dat_ph_grd, dat_az])
    # n_rmp is number of "tracks" for which we create ramps: length(dat)
    n_rmp = dat_coords.size

    # total length (lnth) as in MATLAB
    lnth = int(np.sum(dat_ph) + np.sum(dat_ph_grd) + np.sum(dat_az) + np.sum(dat_gps)*3 + np.sum(dat_gov))

    # sanity checks: total coordinates available should equal sum(dat_coords)
    if x_all.size != np.sum(dat_coords):
        raise ValueError(f"Coordinate vector length mismatch: len(x_all)={x_all.size} vs sum(dat_coords)={np.sum(dat_coords)}")

    # preallocate final rmp matrix: (lnth, 3*n_rmp)
    if n_rmp == 0:
        return np.empty((lnth, 0), dtype=float)

    rmp = np.zeros((lnth, 3 * n_rmp), dtype=float)

    # cumulative indices (python-friendly): cum_dat[0]=0, cum_dat[1]=dat_coords[0], ...
    cum_dat = np.concatenate([[0], np.cumsum(dat_coords, dtype=int)])

    for j in range(n_rmp):
        start = int(cum_dat[j])          # inclusive, 0-based
        end = int(cum_dat[j+1])          # exclusive in python slice
        length = end - start             # should equal dat_coords[j]

        if length < 0:
            raise ValueError("Negative segment length computed - check dat_coords values.")

        # fill the corresponding rows in rmp, and the 3 columns for this ramp
        col_base = 3 * j
        # x_all[start:end] corresponds to those obs coordinates
        # But they must be placed starting at the same row indices start:end in the lnth vector
        rmp[start:end, col_base + 0] = x_all[start:end] / ratio
        rmp[start:end, col_base + 1] = y_all[start:end] / ratio
        rmp[start:end, col_base + 2] = 1.0

    return rmp

def plot_res(
    Greens, rmp, U, D, Urmp0, rmp0,
    dat_ph, dat_ph_grd, dat_az, dat_gps, dat_gov,
    xP, yP, xgrd, ygrd, xA, yA, xG, yG, xO, yO,
    output, zone, xo, yo,patch
):
    """
    Python version of MATLAB plot_res.
    Non-GPS data: residuals via plot_data per segment.
    GPS data: horizontal & vertical quivers, original + predicted.
    """

    s = 5500  # scale factor for GPS arrows

    # flatten inputs
    D_vec = np.asarray(D).ravel()
    G_all = np.hstack((Greens, rmp)) if rmp.size > 0 else Greens
    if G_all.shape[1] != U.size:
        # try to ravel / reshape U - but if mismatch, print and attempt best-effort
        print(f"Warning: mismatch between G_all cols ({G_all.shape[1]}) and U length ({U.size}). Attempting ravel.")
        U = U.ravel()
    forwards = G_all.dot(U)

    # add rmp0*Urmp0 if available and shapes permit
    try:
        if rmp0 is not None and Urmp0 is not None:
            rmp0_arr = np.asarray(rmp0)
            Urmp0_arr = np.asarray(Urmp0).ravel()
            if rmp0_arr.size > 0 and Urmp0_arr.size > 0:
                # try dot product; if shape mismatch, will raise
                forwards = forwards + rmp0_arr.dot(Urmp0_arr)
    except Exception as e:
        print("Warning: failed to add rmp0*Urmp0:", e)

    # residuals (predicted - observed)
    if forwards.shape[0] != D_vec.shape[0]:
        print(f"Warning: forwards rows ({forwards.shape[0]}) != D length ({D_vec.shape[0]}).")
        # try to trim/pad to smallest length to avoid exceptions
        L = min(forwards.shape[0], D_vec.shape[0])
        forwards = forwards[:L]
        D_vec = D_vec[:L]

    r = forwards - D_vec

    # concatenate coordinates
    x = np.concatenate((xP, xgrd, xA, xG, xO))
    y = np.concatenate((yP, ygrd, yA, yG, yO))

    # construct num_dat: 0 + each segment length
    num_dat = [0] + list(dat_ph) + list(dat_ph_grd) + list(dat_az) + [d*3 for d in dat_gps] + list(dat_gov)
    num_cum = np.cumsum(num_dat)
    # iterate each segment
    print("=== Data count summary ===")
    print(f"dat_ph={dat_ph}, dat_ph_grd={dat_ph_grd}, dat_az={dat_az}, dat_gps={dat_gps}, dat_gov={dat_gov}")

    num_dat = [0] + dat_ph + dat_ph_grd + dat_az + [d * 3 for d in dat_gps] + dat_gov
    num_dat = np.array(num_dat, dtype=int)
    num_cum = np.cumsum(num_dat)

    print("num_dat:", num_dat)
    print("num_cum:", num_cum)
    print("===========================")
    # iterate each segment
    for j in range(1, len(num_dat)):
        start = num_cum[j-1]
        end = num_cum[j]
        if end <= start:
            continue

        xx = x[start:end]
        yy = y[start:end]
        rr = r[start:end]
        ff = forwards[start:end]

        # --- non-GPS segments ---
        idx_ph_end = len(dat_ph)
        idx_grd_end = idx_ph_end + len(dat_ph_grd)
        idx_az_end  = idx_grd_end + len(dat_az)
        idx_gps_end = idx_az_end + len(dat_gps)

        if j <= idx_az_end:
            if len(xx) > 0:
                fig, ax = plot_data(xx, yy, rr)
                plot_trace(patch, ax)

                # --- 残差统计分析 ---
                ax_inset = inset_axes(ax, width="60%", height="60%", 
                      bbox_to_anchor=(0.63, 0.4, 0.6, 0.55),
                      bbox_transform=ax.transAxes, loc='upper left')

                # 直方图，归一化为密度
                n, bins, _ = ax_inset.hist(rr, bins=30, density=True, alpha=0.6, color='skyblue')
                ax_inset.set_xlabel("Residual", fontsize=9)
                ax_inset.set_ylabel("Density", fontsize=9)
                ax_inset.set_ylim(0, np.max(n)*1.1)  # 缩放纵轴

                # 高斯拟合
                mu, sigma = norm.fit(rr)
                x_fit = np.linspace(np.min(rr), np.max(rr), 100)
                ax_inset.plot(x_fit, norm.pdf(x_fit, mu, sigma), 'r-', lw=2)
                ax_inset.set_title(f"Residual dist.\nμ={mu:.3g}, σ={sigma:.3g}", fontsize=8)
                ax_inset.tick_params(axis='both', which='major', labelsize=8)
                ax_inset.grid(True, linestyle='--', alpha=0.5)

                if output:
                    np.savetxt(f"dataset{j}.residuals", np.column_stack([xx, yy, rr]), fmt="%.6f", delimiter="\t")
                    np.savetxt(f"dataset{j}.forwards", np.column_stack([xx, yy, ff]), fmt="%.6f", delimiter="\t")

        # --- GPS segments ---
        elif j <= idx_gps_end:
            npts = len(xx) // 3
            if npts == 0:
                print(f"[Warning] GPS segment {j} has no data (len(xx)={len(xx)}). Skipping...")
                continue

            gps_e = D[start:start+npts]
            gps_n = D[start+npts:start+2*npts]
            gps_u = D[start+2*npts:start+3*npts]

            gps_ef = forwards[start:start+npts]
            gps_nf = forwards[start+npts:start+2*npts]
            gps_uf = forwards[start+2*npts:start+3*npts]

            # 检查是否为空
            if len(gps_e) == 0 or len(gps_n) == 0 or len(gps_u) == 0:
                print(f"[Warning] Empty GPS arrays in segment {j}, skip plotting.")
                continue

            Xg = xG[:npts]
            Yg = yG[:npts]
            # 计算水平和垂直分量模长
            U_cm = gps_e 
            V_cm = gps_n 
            W_cm = gps_u 

            mag_h = np.sqrt(U_cm**2 + V_cm**2)
            max_h = np.nanmax(mag_h) if mag_h.size else 1.0
            max_u = np.nanmax(np.abs(W_cm)) if W_cm.size else 1.0

            # 图幅范围 (km)
            x_min, x_max = np.nanmin(Xg), np.nanmax(Xg)
            y_min, y_max = np.nanmin(Yg), np.nanmax(Yg)
            axis_extent = max(x_max - x_min, y_max - y_min, 1.0)  # m
            axis_extent_km = axis_extent / 1000.0  # km

            # 缩放因子: 最大箭头 ~ 图长度的 0.3
            scale_h = max_h / (0.5 * axis_extent_km)
            scale_u = max_u / (0.3 * axis_extent_km)

            # --- 水平 + 垂直分量 (带参考比例尺) ---
            plt.figure(figsize=(8,6))

            # 水平箭头：观测值 (蓝色)
            q_obs_h = plt.quiver(Xg/1000, Yg/1000, gps_e, gps_n,
                                angles='xy', scale_units='xy', scale=scale_h,
                                color='tab:blue', label='Obs (E/N)')

            # 水平箭头：预测值 (红色)
            q_pred_h = plt.quiver(Xg/1000, Yg/1000, gps_ef, gps_nf,
                                angles='xy', scale_units='xy', scale=scale_h,
                                color='tab:red', alpha=0.8, label='Pred (E/N)')
            # 添加参考比例尺箭头
            plt.quiverkey(q_obs_h, X=0.8, Y=0.1, U=10, label="10 cm horizontal",
                        labelpos="E", coordinates="axes", color="tab:blue")
            plt.xlabel("UTM W - E (km)")
            plt.ylabel("UTM S - N (km)")
            plt.title(f"GPS prediction vs observation, segment {j}")
            plt.axis('equal')
            plt.grid(True)
            plt.legend()
            plt.show()
            plt.figure(figsize=(8,6))
            # 垂直箭头：观测值 (紫色)
            q_obs_v = plt.quiver(Xg/1000, Yg/1000, np.zeros_like(Xg), gps_u,
                                angles='xy', scale_units='xy', scale=scale_u,
                                color='purple', label='Obs Up')

            # 垂直箭头：预测值 (橙色)
            q_pred_v = plt.quiver(Xg/1000, Yg/1000, np.zeros_like(Xg), gps_uf,
                                angles='xy', scale_units='xy', scale=scale_u,
                                color='orange', alpha=0.8, label='Pred Up')
            plt.quiverkey(q_obs_v, X=0.8, Y=0.05, U=2, label="2 cm vertical",
                        labelpos="E", coordinates="axes", color="purple")

            plt.xlabel("UTM W - E (km)")
            plt.ylabel("UTM S - N (km)")
            plt.title(f"GPS prediction vs observation, segment {j}")
            plt.axis('equal')
            plt.grid(True)
            plt.legend()
            plt.show()

            # --- 散点图: 预测 vs 观测 ---
            plt.figure(figsize=(6,5))
            plt.plot(gps_e, gps_ef, 'bo', label='E')
            plt.plot(gps_n, gps_nf, 'g^', label='N')
            plt.plot(gps_u, gps_uf, 'rs', label='U')

            # y=x 参考线
            a = min(np.min(gps_e), np.min(gps_n), np.min(gps_u))
            b = max(np.max(gps_e), np.max(gps_n), np.max(gps_u))
            plt.plot([a,b],[a,b],'k--', linewidth=0.8)

            plt.xlabel('Observed (cm)')
            plt.ylabel('Predicted (cm)')
            plt.title(f"GPS scatter: predicted vs observed, segment {j}")
            plt.grid(True)
            plt.legend()
            plt.show()
        # --- GOV (geological / optical offset) segment plotting (replace the old else branch) ---
        else:
            # start, end 已经在循环外计算好：start = num_cum[j-1], end = num_cum[j]
            seg_len = end - start
            if seg_len <= 0:
                print(f"[Skip] GOV segment {j} empty (start={start}, end={end}).")
                continue

            # 计算 xO 在拼接数组 x 中的起始偏移量（拼接顺序： xP, xgrd, xA, xG, xO）
            offset_xO = len(xP) + len(xgrd) + len(xA) + len(xG)

            # 在 xO, yO, dO, azO 中对应的区间
            s0 = start - offset_xO
            e0 = end - offset_xO

            # 检查索引有效性
            if s0 < 0 or e0 > len(xO) or s0 >= e0:
                print(f"[Warning] GOV segment {j} index mapping invalid: s0={s0}, e0={e0}, len(xO)={len(xO)}. Skipping.")
                continue

            # 原始（观测）与预测的数据
            obs_x = xO[s0:e0]
            obs_y = yO[s0:e0]

            # 观测量（通常 D 存储的是观测的位移量 dO）
            obs_d = np.ravel(D[start:end])        # 1D
            pred_d = np.ravel(forwards[start:end])  # 1D (预测的同量纲值，和 obs_d 对齐)

            # 若 azO 可用则取对应段的方位，否则默认 0（不会发生错误）
            if len(azO) >= (e0 - s0):
                obs_az = np.asarray(azO[s0:e0], dtype=float)
            else:
                obs_az = np.zeros(e0 - s0, dtype=float)
                print(f"[Info] azO length ({len(azO)}) < required ({e0-s0}), using zeros for azimuths.")

            # 保证所有数组长度一致
            L = min(len(obs_x), len(obs_y), len(obs_d), len(pred_d), len(obs_az))
            if L == 0:
                print(f"[Warning] GOV segment {j} has no valid points after trimming. Skipping.")
                continue

            obs_x = obs_x[:L]; obs_y = obs_y[:L]
            obs_d = obs_d[:L]; pred_d = pred_d[:L]; obs_az = obs_az[:L]

            # 计算分量（与你的 plot_geological_obs 中一致：将 az 转换为 90-az）
            ang = np.deg2rad(90.0 - obs_az)
            Ux_obs = obs_d * np.cos(ang)
            Uy_obs = obs_d * np.sin(ang)

            Ux_pred = pred_d * np.cos(ang)
            Uy_pred = pred_d * np.sin(ang)

            # 残差向量 = 观测向量 - 预测向量（矢量差）
            Ux_res = Ux_obs - Ux_pred
            Uy_res = Uy_obs - Uy_pred

            # 绘图：调用你的 plot_geological_obs 画观测（它会创建 fig/ax 若未传入）
            scale_ref = 100.0
            fig, ax = plt.subplots(figsize=(7, 6))
            # 使用 plot_geological_obs（传入 obs arrays），它会返回 ax
            ax = plot_geological_obs(obs_x, obs_y, obs_d, obs_az, scale_ref=scale_ref, ax=ax)

            # 叠加预测（红色）与残差（黑色）箭头 —— 使用与 plot_geological_obs 相似的 scale_units/参数
            # 注意：plot_geological_obs 在内部使用 xO/1000 (以 km 为坐标轴单位)，因此也使用相同的 x/1000
            ax.quiver(
                obs_x / 1000.0, obs_y / 1000.0, Ux_pred, Uy_pred,
                color="tab:red", scale=0.5, scale_units="xy", angles="xy",
                width=0.004, headwidth=3.5, headlength=5, alpha=0.85
            )

            ax.quiver(
                obs_x / 1000.0, obs_y / 1000.0, Ux_res, Uy_res,
                color="k", scale=0.5, scale_units="xy", angles="xy",
                width=0.006, headwidth=3, headlength=4, alpha=1.0
            )

            # 自定义图例（quiver对象不直接入 legend，使用 Line2D 代替图例项）
            from matplotlib.lines import Line2D
            handles = [
                Line2D([0], [0], color="tab:green", lw=2, label="Observed (proj.)"),
                Line2D([0], [0], color="tab:red", lw=2, label="Predicted"),
                Line2D([0], [0], color="k", lw=2, label="Residual (Obs-Pred)"),
            ]
            ax.legend(handles=handles, loc="upper right", fontsize=10)

            ax.set_title("Geological / optical offsets: observed (green), predicted (red), residual (black)")
            ax.set_xlabel("UTM W - E (km)")
            ax.set_ylabel("UTM S - N (km)")
            ax.set_aspect("equal", adjustable="box")
            ax.grid(True)
            plt.show()

            # 可选：将残差与预测保存到文件（保持与你原来 MATLAB 行为相似）
            if output:
                try:
                    lon, lat = utm2ll(obs_x + xo, obs_y + yo, zone, 2)
                except Exception:
                    lon, lat = obs_x, obs_y
                # 保存： lon lat obs_d pred_d (cm) residuals (cm)
                out_arr = np.column_stack((lon, lat, obs_d, pred_d, obs_d - pred_d))
                np.savetxt(f"gov_segment_{j}_obs_pred_res.txt", out_arr, fmt="%.6f", delimiter="\t",
                        header="lon\tlat\tobs_cm\tpred_cm\tresid_cm")

    print("plot_res completed.")



def check_bounds_and_columns(Greens, rmp, lb, ub, dat_sizes, dataset_names=None):
    """
    检查每个数据集对应 Greens/rmp 列的 lb/ub 情况。

    Parameters
    ----------
    Greens : ndarray, shape (ndata, ncols1)
        主反演矩阵
    rmp : ndarray, shape (ndata, ncols2)
        ramp 矩阵
    lb : ndarray, shape (ncols,)
        下界
    ub : ndarray, shape (ncols,)
        上界
    dat_sizes : list of int
        每个数据集的行数/数据量，例如 [sum(dat_ph), sum(dat_ph_grd), ...]
    dataset_names : list of str, optional
        每个数据集的名字，用于打印
    """
    total_cols = Greens.shape[1] + rmp.shape[1]
    col_mask = np.ones(total_cols, dtype=bool)  # 所有列有效

    # 数据集名称
    if dataset_names is None:
        dataset_names = [f"dataset_{i+1}" for i in range(len(dat_sizes))]

    print("=== Check lb/ub and Greens/rmp columns ===")
    for i, nrows in enumerate(dat_sizes):
        # 对应的数据集列可能没有明确分段，这里我们检查整列是否被 lb==ub==0
        # 对每列检查是否所有值都为 0
        lb_zero = np.all(lb == 0)
        ub_zero = np.all(ub == 0)
        # 如果 lb==ub==0，说明该列被固定为 0
        if lb_zero and ub_zero:
            status = "FIXED TO ZERO!"
        else:
            status = "OK"

        print(f"{dataset_names[i]}: nrows={nrows}, lb_all_zero={lb_zero}, ub_all_zero={ub_zero}, status={status}")

    print("Total Greens+rmp shape:", Greens.shape[0], "+", rmp.shape[1], "->", total_cols)
    print("=======================================")


def check_D_Green_segments(D_all, Greens, rmp, dat_ph, dat_ph_grd, dat_az, dat_gps, dat_gov):
    """
    检查 D_all 和 Greens/rmp 对应的数据段是否合理
    """
    # 合并 Greens 和 rmp
    Green_all = np.hstack([Greens, rmp])
    n_rows, n_cols = Green_all.shape
    print(f"Total Greens+rmp shape: {n_rows} x {n_cols}")

    # 数据段累积索引
    num_dat = [0, np.sum(dat_ph), np.sum(dat_ph_grd), np.sum(dat_az), np.sum(dat_gps), np.sum(dat_gov)]
    labels = ["phase", "phase_grad", "azimuth", "gps", "gov"]

    for j in range(len(num_dat)-1):
        start = int(num_dat[j])
        end = int(num_dat[j+1])
        if end > start:
            D_segment = D_all[start:end]
            Green_segment = Green_all[start:end, :]
            zero_D = np.all(D_segment == 0)
            zero_G = np.all(Green_segment == 0, axis=0)
            print(f"{labels[j]}: rows={end-start}, D_all all zero={zero_D}, "
                  f"any Greens/rmp col all zero={np.any(zero_G)}")
            print(f"   D_all min={np.min(D_segment):.3e}, max={np.max(D_segment):.3e}, mean={np.mean(D_segment):.3e}")
        else:
            print(f"{labels[j]}: rows=0, skipped")

def check_Greens_rmp_segments(Greens, rmp, D_all, dat_ph, dat_ph_grd, dat_az, dat_gps, dat_gov):
    """
    检查 Greens + rmp 每段列对应 D_all 的情况，标出哪些列全为 0。
    
    Parameters
    ----------
    Greens : np.ndarray
        基础 Greens 矩阵, shape = (n_rows, n_cols_Greens)
    rmp : np.ndarray
        ramp 矩阵, shape = (n_rows, n_cols_rmp)
    D_all : np.ndarray
        数据向量, shape = (n_rows,)
    dat_ph, dat_ph_grd, dat_az, dat_gps, dat_gov : int
        各段数据行数
    
    Returns
    -------
    None
    """
    # 总矩阵
    G_total = np.hstack([Greens, rmp])
    n_rows, n_cols = G_total.shape
    print(f"Total Greens+rmp shape: {n_rows} x {n_cols}")
    
    # 每段行数及名称
    segments = [
        ("phase", int(np.sum(dat_ph))),
        ("phase_grad", int(np.sum(dat_ph_grd))),
        ("azimuth", int(np.sum(dat_az))),
        ("gps", int(np.sum(dat_gps))),
        ("gov", int(np.sum(dat_gov)))
    ]
    
    start_row = 0
    for name, n_seg_rows in segments:
        if n_seg_rows == 0:
            print(f"{name}: rows={n_seg_rows}, skipped")
            continue
        
        end_row = start_row + n_seg_rows
        G_seg = G_total[start_row:end_row, :]
        D_seg = D_all[start_row:end_row].flatten()
        
        # 检查该段列是否全为零
        cols_all_zero = np.where(np.all(G_seg == 0, axis=0))[0]
        lb_all_zero = False  # 这里可扩展，如果有 lb 也检查
        ub_all_zero = False
        
        print(f"{name}: rows={n_seg_rows}, D_all all zero={np.all(D_seg==0)}, any Greens/rmp col all zero={len(cols_all_zero)>0}")
        if len(cols_all_zero) > 0:
            print(f"   Columns all zero: {cols_all_zero}")
        
        start_row = end_row


def check_and_plot_Greens_segments(Greens, rmp, segments, segment_names):
    """
    检查 Greens+rmp 每个 segment 的列是否全为 0，并绘制可视化
    
    Parameters
    ----------
    Greens : np.ndarray, shape (nrows, ncols_greens)
        Green's functions matrix
    rmp : np.ndarray, shape (nrows, ncols_rmp)
        Ramp matrix
    segments : list of tuples
        每个 segment 在 Greens+rmp 中的列索引范围，例如 [(0,832),(832,858),...]
    segment_names : list of str
        每个 segment 的名字，例如 ['phase','phase_grad','azimuth','gps','gov']
    """
    Gtotal = np.hstack([Greens, rmp])
    nrows, ncols = Gtotal.shape
    
    print(f"Total Greens+rmp shape: {nrows} x {ncols}")
    
    plt.figure(figsize=(12,6))
    plt.imshow(Gtotal != 0, aspect='auto', cmap='Greys', origin='lower')
    plt.xlabel('Columns (Greens+rmp)')
    plt.ylabel('Rows (data points)')
    plt.title('Non-zero structure of Greens+rmp')
    
    for seg_idx, (start, end) in enumerate(segments):
        seg_name = segment_names[seg_idx]
        cols = np.arange(start, end)
        seg_data = Gtotal[:, start:end]
        
        # 找出全零列
        all_zero_cols = cols[np.all(seg_data == 0, axis=0)]
        if len(all_zero_cols) > 0:
            print(f"{seg_name}: rows={nrows}, any Greens+rmp col all zero=True")
            print(f"   Columns all zero: {all_zero_cols.tolist()}")
        else:
            print(f"{seg_name}: rows={nrows}, any Greens+rmp col all zero=False, status=OK")
        
        # 在图上标出 segment 范围
        plt.axvline(x=start-0.5, color='red', linestyle='--', alpha=0.6)
        plt.text((start+end)/2, nrows*1.02, seg_name, ha='center', color='blue', fontsize=10)
    
    plt.colorbar(label='Non-zero (1) / zero (0)')
    plt.tight_layout()
    plt.show()