import numpy as np
import xarray as xr
import matplotlib as mpl
import matplotlib.pyplot as plt

def generate_green_p(XS, YS, ZS, LL, WW, DIP, STRIKE, xP, yP, tpP, look, nu, fault_type):
    """
    Generate Green's function matrix for InSAR LOS data.

    Parameters
    ----------
    XS, YS, ZS : array-like (n_patches,)
        patch origins (UTM or model coordinates)
    LL, WW : array-like (n_patches,)
        patch length and width (meters)
    DIP, STRIKE : array-like (n_patches,) in degrees
    xP, yP : array-like (n_obs,)
        observation coordinates (same units as XS/YS)
    tpP : scalar or array_like
        passed to calc_okada (keeps same as your MATLAB pipeline)
    look : array-like (n_obs, 3)
        look vector components (columns for x,y,z)
    nu : float
        Poisson ratio for calc_okada
    fault_type : array-like (len>=3)
        e.g. [1,1,0] to enable strike and dip components only

    Returns
    -------
    GreenP : ndarray, shape (n_obs, n_columns)
        Each column corresponds to a patch-slip-component projected to LOS.
    """
    # convert inputs to numpy arrays
    XS = np.asarray(XS, dtype=float)
    YS = np.asarray(YS, dtype=float)
    ZS = np.asarray(ZS, dtype=float)
    XC = np.zeros_like(XS, dtype=float)
    YC = np.zeros_like(YS, dtype=float)
    ZC = np.zeros_like(ZS, dtype=float)
    LL = np.asarray(LL, dtype=float)
    WW = np.asarray(WW, dtype=float)
    DIP = np.asarray(DIP, dtype=float)
    STRIKE = np.asarray(STRIKE, dtype=float)
    xP = np.asarray(xP, dtype=float)
    yP = np.asarray(yP, dtype=float)
    look = np.asarray(look, dtype=float)
    fault_type = np.asarray(fault_type, dtype=int)

    # handle empty observation list
    if xP.size == 0:
        return np.empty((0, 0))

    # ensure look shape is (n_obs, 3)
    if look.ndim == 1 and look.size == 3:
        look = look.reshape(1, 3)
    if look.ndim == 2 and look.shape[1] != 3 and look.shape[0] == 3:
        look = look.T
    if look.ndim != 2 or look.shape[1] != 3:
        raise ValueError("`look` must be shape (n_obs, 3) or (3, n_obs)")

    n_obs = xP.size
    if look.shape[0] != n_obs:
        raise ValueError(f"Number of look vectors ({look.shape[0]}) must match number of observations ({n_obs})")

    # convert angles to radians (MATLAB did STRIKE = STRIKE/180*pi)
    STRIKE_rad = np.deg2rad(STRIKE)
    DIP_rad = np.deg2rad(DIP)

    # compute patch centers (same formula as MATLAB)
    XC = XS + 0.5 * LL * np.sin(STRIKE_rad)
    YC = YS + 0.5 * LL * np.cos(STRIKE_rad)
    ZC = ZS  # depth

    # prepare list to collect columns
    cols = []

    # iterate over patches
    n_patches = XC.size
    for j in range(n_patches):
        # relative coordinates from patch center to observation points
        x_rel = xP - XC[j]
        y_rel = yP - YC[j]

        # loop through slip components 1=strike,2=dip,3=normal (MATLAB style)
        for k in (1, 2, 3):
            if k - 1 < fault_type.size:
                enabled = fault_type[k - 1] != 0
            else:
                enabled = False

            if not enabled:
                continue

            # call calc_okada
            # MATLAB call: calc_okada(1,1,x,y,nu,DIP(j),-ZC(j),LL(j),WW(j),k,STRIKE(j),tpP)
            # We assume Python calc_okada has compatible argument ordering.
            ux, uy, uz = calc_okada(1, 1, x_rel, y_rel, nu,
                                     DIP_rad[j], -ZC[j], LL[j], WW[j],
                                     k, STRIKE_rad[j], tpP)

            # ensure returned ux,uy,uz are 1D arrays of length n_obs
            ux = np.atleast_1d(np.asarray(ux, dtype=float)).ravel()
            uy = np.atleast_1d(np.asarray(uy, dtype=float)).ravel()
            uz = np.atleast_1d(np.asarray(uz, dtype=float)).ravel()
            if ux.size != n_obs or uy.size != n_obs or uz.size != n_obs:
                raise ValueError("calc_okada must return arrays of length equal to number of observations")

            # project onto look vector: ux*look[:,0] + uy*look[:,1] + uz*look[:,2]
            proj = ux * look[:, 0] + uy * look[:, 1] + uz * look[:, 2]

            cols.append(proj)

    # if no columns were produced, return empty array with shape (n_obs,0)
    if len(cols) == 0:
        return np.empty((n_obs, 0))

    # stack columns horizontally -> shape (n_obs, n_cols)
    GreenP = np.column_stack(cols)
    return GreenP

def generate_greenp_grd(Xc, Yc, Zc, ll, ww, DIP, STRIKE,
                        xP, yP, tpP, looks, nu, fault_type, dat_ph_grd, dx, dy):
    """
    Compute LOS Green's function gradients (∂LOS/∂x and ∂LOS/∂y)
    using finite differences, equivalent to MATLAB generate_greenp_grd.
    """
    # ensure high-precision, Fortran layout to mimic MATLAB
    Xc = np.asarray(Xc, dtype=np.float64, order='F')
    Yc = np.asarray(Yc, dtype=np.float64, order='F')
    Zc = np.asarray(Zc, dtype=np.float64, order='F')
    ll  = np.asarray(ll,  dtype=np.float64, order='F')
    ww  = np.asarray(ww,  dtype=np.float64, order='F')
    DIP = np.asarray(DIP, dtype=np.float64, order='F')
    STRIKE = np.asarray(STRIKE, dtype=np.float64, order='F')

    xP = np.asarray(xP, dtype=np.float64, order='F')
    yP = np.asarray(yP, dtype=np.float64, order='F')
    tpP = np.asarray(tpP, dtype=np.float64, order='F')
    looks = np.asarray(looks, dtype=np.float64, order='F')
    # dat_ph_grd can stay int
    dat_ph_grd = np.asarray(dat_ph_grd, dtype=np.int64)

    n = len(dat_ph_grd)
    n_half = n // 2
    n_x = np.sum(dat_ph_grd[:n_half])
    n_y = np.sum(dat_ph_grd[n_half:])

    # --- X-direction finite difference ---
    idx_x = np.arange(int(n_x))
    Gp_xp = generate_green_p(Xc, Yc, Zc, ll, ww, DIP, STRIKE,
                             xP[idx_x] + dx, yP[idx_x], tpP[idx_x],
                             looks[idx_x, :], nu, fault_type)
    Gp_xm = generate_green_p(Xc, Yc, Zc, ll, ww, DIP, STRIKE,
                             xP[idx_x] - dx, yP[idx_x], tpP[idx_x],
                             looks[idx_x, :], nu, fault_type)
    Gx = (Gp_xp - Gp_xm) / (2 * dx)

    # --- Y-direction finite difference ---
    idx_y = np.arange(int(n_x), len(xP))
    Gp_yp = generate_green_p(Xc, Yc, Zc, ll, ww, DIP, STRIKE,
                             xP[idx_y], yP[idx_y] + dy, tpP[idx_y],
                             looks[idx_y, :], nu, fault_type)
    Gp_ym = generate_green_p(Xc, Yc, Zc, ll, ww, DIP, STRIKE,
                             xP[idx_y], yP[idx_y] - dy, tpP[idx_y],
                             looks[idx_y, :], nu, fault_type)
    Gy = (Gp_yp - Gp_ym) / (2 * dy)

    # --- Combine ---
    Greenp_grd = np.vstack((Gx, Gy))
    return Greenp_grd


def generate_green_a(XS, YS, ZS, LL, WW, DIP, STRIKE, xA, yA, tpA, phi, nu, fault_type, dat_az):
    """
    Generate Green's function matrix for azimuth offset data, MATLAB-consistent.

    Parameters
    ----------
    XS, YS, ZS, LL, WW, DIP, STRIKE : 1D arrays, length = n_patches
        Patch parameters.
    xA, yA, tpA : 1D arrays, length = sum(dat_az)
        Observation coordinates and topography.
    phi : 1D array, length = n_groups
        Azimuth angles in degrees.
    nu : float
        Poisson ratio.
    fault_type : iterable of length 3
        Slip components enabled, e.g., [1,1,0] for strike+dip.
    dat_az : 1D array, length = n_groups
        Number of observations per azimuth group.
        
    Returns
    -------
    GreenA : ndarray, shape (sum(dat_az), n_columns)
        Each column corresponds to a patch-slip-component projected to azimuth.
    """
    # flatten inputs
    XS = np.asarray(XS).ravel().astype(float)
    YS = np.asarray(YS).ravel().astype(float)
    ZS = np.asarray(ZS).ravel().astype(float)
    LL = np.asarray(LL).ravel().astype(float)
    WW = np.asarray(WW).ravel().astype(float)
    DIP = np.asarray(DIP).ravel().astype(float)
    STRIKE = np.asarray(STRIKE).ravel().astype(float)

    xA = np.asarray(xA).ravel().astype(float)
    yA = np.asarray(yA).ravel().astype(float)
    tpA = np.asarray(tpA).ravel().astype(float)

    phi = np.asarray(phi).ravel().astype(float)
    dat_az = np.asarray(dat_az).ravel().astype(int)
    fault_type = np.asarray(fault_type).ravel().astype(int)

    if xA is None or len(xA) == 0:
        return np.zeros((0, 0))

    # build PHI
    phi = np.deg2rad(phi)
    PHI = np.concatenate([np.full((dat_az[j],), phi[j]) for j in range(len(phi))])

    STRIKE = np.deg2rad(STRIKE)
    DIP = np.deg2rad(DIP)

    XC = XS + 0.5 * LL * np.sin(STRIKE)
    YC = YS + 0.5 * LL * np.cos(STRIKE)
    ZC = ZS

    GreenA = np.empty((len(xA), 0))

    for j in range(len(XC)):
        x = xA - XC[j]
        y = yA - YC[j]

        for k in range(3):
            if fault_type[k] != 0:
                ux, uy, uz = calc_okada(1, 1, x, y, nu,
                                        DIP[j], -ZC[j], LL[j], WW[j],
                                        k+1, STRIKE[j], tpA)
                ux = np.ravel(ux)
                uy = np.ravel(uy)

                # match MATLAB elementwise multiplication
                tmp = (ux * np.sin(PHI) + uy * np.cos(PHI)).reshape(-1, 1)
                GreenA = np.hstack([GreenA, tmp])

    return GreenA

def generate_green_g(XS, YS, ZS, LL, WW, DIP, STRIKE, xG, yG, tpG,
                     nu, fault_type, gps_type, dat_gps):
    """
    Generate Green's function matrix for GPS observations (Python port of MATLAB generate_green_g).

    Inputs (array-likes):
      XS,YS,ZS,LL,WW,DIP,STRIKE : 1D arrays of same length = number of patches
          DIP, STRIKE are in degrees (will be converted to radians inside).
      xG, yG, tpG : 1D arrays containing concatenated station coordinates/topo,
          grouped according to dat_gps.
      nu : Poisson ratio (scalar)
      fault_type : iterable length 3, values 0/1/2/3 etc. (1->strike,2->dip,3->tensile). 
                   MATLAB used m idx 1..3; here we expect fault_type[0] corresponds to m=1, etc.
      gps_type : 2 x n_groups (or n_groups x 2) array-like. First row -> horizontal flag,
                 second row -> vertical flag. Non-zero means that component exists for that group.
      dat_gps : 1D array of length n_groups giving number of stations in each group

    Output:
      GreenG : numpy.ndarray shape (n_observations, n_columns)
              n_observations = sum_k ((gps_type[0,k]!=0)*2 + (gps_type[1,k]!=0)) * dat_gps[k]
              Columns correspond to each patch j and each active slip component m in order.
    """
    # convert to numpy arrays
    XS = np.asarray(XS, dtype=float)
    YS = np.asarray(YS, dtype=float)
    ZS = np.asarray(ZS, dtype=float)
    LL = np.asarray(LL, dtype=float)
    WW = np.asarray(WW, dtype=float)
    DIP = np.asarray(DIP, dtype=float)
    STRIKE = np.asarray(STRIKE, dtype=float)

    xG = np.asarray(xG, dtype=float).ravel()
    yG = np.asarray(yG, dtype=float).ravel()
    tpG = np.asarray(tpG, dtype=float).ravel()

    dat_gps = np.asarray(dat_gps, dtype=int).ravel()
    # print(XS.shape,YS.shape,ZS.shape,LL.shape,WW.shape,DIP.shape,xG.shape,yG.shape,tpG.shape,dat_gps)
    n_groups = dat_gps.size
    gps_type_arr = np.asarray(gps_type)
    # normalize gps_type to 2 x n_groups
    if gps_type_arr.ndim == 1:
        # if shape (2,) or (n_groups,), try to reshape sensibly
        if gps_type_arr.size == 2 and n_groups == 1:
            gps_type_arr = gps_type_arr.reshape(2, 1)
        else:
            raise ValueError("gps_type 1D ambiguous; expected 2 x n_groups or n_groups x 2.")
    if gps_type_arr.shape[0] == 2 and gps_type_arr.shape[1] == n_groups:
        gps_type_mat = gps_type_arr
    elif gps_type_arr.shape[1] == 2 and gps_type_arr.shape[0] == n_groups:
        gps_type_mat = gps_type_arr.T
    else:
        raise ValueError("gps_type must be 2 x n_groups or n_groups x 2.")

    # quick empty check (MATLAB returns empty if xG empty)
    if xG.size == 0:
        return np.empty((0, 0))

    # convert strike/dip degrees -> radians as in MATLAB
    STRIKE_rad = np.deg2rad(STRIKE)
    DIP_rad = np.deg2rad(DIP)

    # compute patch centers - same formula as MATLAB
    XC = XS + 0.5 * LL * np.sin(STRIKE_rad)
    YC = YS + 0.5 * LL * np.cos(STRIKE_rad)
    ZC = ZS

    # compute total number of observation rows
    n_obs = 0
    for k in range(n_groups):
        ncomp = (int(gps_type_mat[0, k]) != 0) * 2 + (int(gps_type_mat[1, k]) != 0)
        n_obs += ncomp * int(dat_gps[k])

    # pre-check that xG/yG/tpG length matches sum(dat_gps)

    # if xG.size != dat_gps.sum() or yG.size != dat_gps.sum() or tpG.size != dat_gps.sum():
    #     raise ValueError("Length of xG/yG/tpG must equal sum(dat_gps).")

    # container for columns
    columns = []

    # iterate patches and slip components (m = 1..3)
    n_patches = XC.size
    for j in range(n_patches):
        for m in (1, 2, 3):
            if int(fault_type[m - 1]) == 0:
                # MATLAB: if fault_type(m) == 0 => skip (no column appended)
                continue

            # accumulate pieces (groups) for this column
            G_group_pieces = []
            idx_start = 0
            for k in range(n_groups):
                n_stations = int(dat_gps[k])
                idx_end = idx_start + n_stations

                # make x,y relative to patch center like MATLAB
                x_slice = xG[idx_start:idx_end] - XC[j]
                y_slice = yG[idx_start:idx_end] - YC[j]
                tp_slice = tpG[idx_start:idx_end]

                # call calc_okada: HF=1, U=1 as in MATLAB call
                ux, uy, uz = calc_okada(
                    1, 1, x_slice, y_slice, nu,
                    DIP_rad[j], -ZC[j], LL[j], WW[j],
                    m, STRIKE_rad[j], tp_slice
                )

                # for this group, stack components in the order MATLAB used:
                # first (if present) ux (all stations), then uy (all stations),
                # then (if present) uz (all stations)
                group_components = []
                if gps_type_mat[0, k] != 0:
                    group_components.append(np.asarray(ux).ravel())
                    group_components.append(np.asarray(uy).ravel())
                if gps_type_mat[1, k] != 0:
                    group_components.append(np.asarray(uz).ravel())

                if group_components:
                    # concatenating horizontally the arrays for this group
                    # results in vector length = n_stations * n_components_in_group
                    G_group_pieces.append(np.concatenate(group_components, axis=0))

                idx_start = idx_end

            # after collecting all groups, concatenate vertically to full column
            if G_group_pieces:
                G_col = np.concatenate(G_group_pieces, axis=0)
            else:
                # No gps components at all -> would correspond to empty G in MATLAB, skip
                continue

            # validate length
            if G_col.size != n_obs:
                raise RuntimeError(
                    f"Generated column length {G_col.size} != expected n_obs {n_obs} "
                    f"(patch {j}, m={m}). Check gps_type/dat_gps ordering."
                )

            columns.append(G_col)

    # stack columns to form GreenG matrix
    if len(columns) == 0:
        return np.empty((n_obs, 0))
    GreenG = np.column_stack(columns)
    return GreenG

def generate_green_o(XS, YS, ZS, LL, WW, DIP, STRIKE,
                     xO, yO, azO, tpO, nu, fault_type):
    """
    Generate Green's function for geological offset data.

    Parameters
    ----------
    XS, YS, ZS : array-like (n_patches,)
        Patch origins
    LL, WW : array-like (n_patches,)
        Patch length and width
    DIP, STRIKE : array-like (n_patches,)
        Dip and strike (degrees)
    xO, yO : array-like (n_obs,)
        Observation coordinates
    azO : float or array-like
        Azimuth of geological offset (degrees, azimuth direction of slip measurement)
    tpO : array-like
        Additional topo parameter (like Okada input)
    nu : float
        Poisson's ratio
    fault_type : list or array (len>=3)
        Slip component selector [strike, dip, tensile]

    Returns
    -------
    GreenO : ndarray (n_obs, n_cols)
        Green's function matrix for geological data.
    """

    # 转 numpy 数组
    XS, YS, ZS = map(lambda a: np.atleast_1d(np.asarray(a, float)), (XS, YS, ZS))
    LL, WW = map(lambda a: np.atleast_1d(np.asarray(a, float)), (LL, WW))
    DIP, STRIKE = map(lambda a: np.atleast_1d(np.asarray(a, float)), (DIP, STRIKE))
    xO, yO, tpO = map(lambda a: np.atleast_1d(np.asarray(a, float)), (xO, yO, tpO))
    azO = np.atleast_1d(np.asarray(azO, float))
    fault_type = np.atleast_1d(np.asarray(fault_type, int))

    # 如果没有观测点，返回空
    if xO.size == 0:
        return np.empty((0, 0))

    # 转为弧度
    STRIKE_rad = np.deg2rad(STRIKE)
    DIP_rad = np.deg2rad(DIP)

    # patch 中心
    XC = XS + 0.5 * LL * np.sin(STRIKE_rad)
    YC = YS + 0.5 * LL * np.cos(STRIKE_rad)
    ZC = ZS

    cols = []
    for j in range(XC.size):
        x = xO - XC[j]
        y = yO - YC[j]

        # dx, dy 按 azimuth 偏移
        dx = np.cos(np.deg2rad(180.0 - azO)) * 10.0
        dy = np.sin(np.deg2rad(180.0 - azO)) * 10.0

        for k in (1, 2, 3):
            if k - 1 < fault_type.size and fault_type[k - 1] != 0:
                # Okada 位移差分
                ux1, uy1, uz1 = calc_okada(
                    1, 1, x + dx, y + dy, nu,
                    DIP_rad[j], -ZC[j], LL[j], WW[j], k, STRIKE_rad[j], tpO
                )
                ux2, uy2, uz2 = calc_okada(
                    1, 1, x - dx, y - dy, nu,
                    DIP_rad[j], -ZC[j], LL[j], WW[j], k, STRIKE_rad[j], tpO
                )

                ux1, uy1 = np.asarray(ux1, float).ravel(), np.asarray(uy1, float).ravel()
                ux2, uy2 = np.asarray(ux2, float).ravel(), np.asarray(uy2, float).ravel()

                # 投影到走向 azimuth
                proj = (ux1 - ux2) * np.cos(np.deg2rad(90.0 - azO)) \
                     + (uy1 - uy2) * np.sin(np.deg2rad(90.0 - azO))

                cols.append(proj)

    if len(cols) == 0:
        return np.empty((xO.size, 0))

    GreenO = np.column_stack(cols)
    return GreenO

def fBi(sig, eta, parvec, p, q):
    """
    Python translation of MATLAB fBi(sig, eta, parvec, p, q)
    parvec = [a, delta, fault_type]
      - delta is expected in radians (same as MATLAB)
      - fault_type: 1 (strike-slip), 2 (dip-slip), 3 (tensile)

    Returns: f1, f2, f3 (numpy arrays, shape broadcasted to input)
    """
    # unpack parameters
    a = parvec[0]
    delta = parvec[1]
    fault_type = int(parvec[2])

    epsn = 1.0e-15

    # trigonometric helpers
    cosd = np.cos(delta)
    sind = np.sin(delta)
    tand = np.tan(delta)
    cosd2 = cosd**2
    sind2 = sind**2
    cssnd = cosd * sind

    # ensure arrays and broadcast
    sig = np.atleast_1d(np.asarray(sig))
    eta = np.atleast_1d(np.asarray(eta))
    q   = np.atleast_1d(np.asarray(q))
    sig, eta, q = np.broadcast_arrays(sig, eta, q)

    # core geometric quantities
    R = np.sqrt(sig**2 + eta**2 + q**2)
    X = np.sqrt(sig**2 + q**2)
    ytil = eta * cosd + q * sind
    dtil = eta * sind - q * cosd

    Rdtil = R + dtil
    Rsig  = R + sig
    Reta  = R + eta
    RX    = R + X

    # logs and recipricals
    # compute safely (will adjust problematic entries below)
    with np.errstate(divide='ignore', invalid='ignore'):
        lnRdtil = np.log(Rdtil)
        lnReta  = np.log(Reta)
        lnReta0 = -np.log(R - eta)   # used when Reta ~ 0

        ORRsig = 1.0 / (R * Rsig)
        ORsig  = 1.0 / Rsig
        OReta  = 1.0 / Reta
        ORReta = 1.0 / (R * Reta)

    # fix near-singular entries (match MATLAB logic)
    mask_Reta_small = np.abs(Reta) < epsn
    if np.any(mask_Reta_small):
        lnReta[mask_Reta_small] = lnReta0[mask_Reta_small]
        OReta[mask_Reta_small]  = 0.0
        ORReta[mask_Reta_small] = 0.0

    mask_Rsig_small = np.abs(Rsig) < epsn
    if np.any(mask_Rsig_small):
        ORsig[mask_Rsig_small]  = 0.0
        ORRsig[mask_Rsig_small] = 0.0

    # theta handling: avoid dividing by q=0
    theta = np.zeros_like(R)
    mask_q_zero = np.abs(q) <= epsn
    if np.any(mask_q_zero):
        # for q==0 set theta = 0
        theta[mask_q_zero] = 0.0
        mask_q_nonzero = ~mask_q_zero
        if np.any(mask_q_nonzero):
            theta[mask_q_nonzero] = np.arctan(
                (sig[mask_q_nonzero] * eta[mask_q_nonzero]) /
                (q[mask_q_nonzero] * R[mask_q_nonzero])
            )
    else:
        theta = np.arctan((sig * eta) / (q * R))

    # I1..I5 computation
    if np.abs(cosd) < epsn:
        # cosd == 0 branch (delta ~ pi/2)
        I5 = -a * sig * sind / Rdtil
        I4 = -a * q / Rdtil
        I3 = (a / 2.0) * (eta / Rdtil + (ytil * q) / (Rdtil**2) - lnReta)
        I2 = -a * lnReta - I3
        I1 = - (a / 2.0) * (sig * q) / (Rdtil**2)
    else:
        # default branch
        # compute argument for atan safely; follow MATLAB using atan (not atan2)
        numer = (eta * (X + q * cosd) + X * RX * sind)
        denom = (sig * RX * cosd)
        # avoid invalid division: denom may be zero for some entries
        # compute fraction with errstate
        with np.errstate(divide='ignore', invalid='ignore'):
            frac = numer / denom
            I5 = a * 2.0 / cosd * np.arctan(frac)

        # set I5 to zero where sig == 0 (MATLAB did)
        mask_sig_zero = np.abs(sig) < epsn
        if np.any(mask_sig_zero):
            I5[mask_sig_zero] = 0.0

        I4 = a / cosd * (lnRdtil - sind * lnReta)
        I3 = a * ( (1.0 / cosd) * (ytil / Rdtil) - lnReta ) + tand * I4
        I2 = -a * lnReta - I3
        I1 = -a / cosd * (sig / Rdtil) - tand * I5

    # fault-specific f1,f2,f3
    if fault_type == 1:
        # Strike-slip (eqn.25)
        f1 = (sig * q) * ORReta + theta + I1 * sind
        f2 = (ytil * q) * ORReta + (q * cosd) * OReta + I2 * sind
        f3 = (dtil * q) * ORReta + (q * sind) * OReta + I4 * sind
    elif fault_type == 2:
        # Dip-slip (eqn.26)
        f1 = q / R - I3 * cssnd
        f2 = (ytil * q) * ORRsig + cosd * theta - I1 * cssnd
        f3 = (dtil * q) * ORRsig + sind * theta - I5 * cssnd
    elif fault_type == 3:
        # Tensile (eqn.27)
        f1 = q**2 * ORReta - I3 * sind2
        f2 = (-dtil * q) * ORRsig - sind * ((sig * q) * ORReta - theta) - I1 * sind2
        f3 = (ytil * q) * ORRsig + cosd * ((sig * q) * ORReta - theta) - I5 * sind2
    else:
        raise ValueError("fault_type must be 1, 2, or 3")

    return f1, f2, f3

def calc_okada(HF, U, x, y, nu, delta, d, length, W, fault_type, strike, tp=None):
    """
    Python translation of MATLAB calc_okada.
    Inputs:
      HF, U : scalars (HF is horizontal scaling, U is slip amplitude)
      x, y, d, tp : scalars or arrays (observation coords and topography)
      nu, delta, strike, length, W : scalars (nu Poisson's ratio; delta, strike in radians)
      fault_type : integer 1,2,3 (strike, dip, opening)
      tp : optional topography array (same shape as x,y) or scalar; if None -> zeros
    Returns:
      ux, uy, uz : arrays (same shape as broadcast(x,y,d,tp))
    Note: expects delta and strike in radians (same as MATLAB).
    """

    # convert to numpy arrays and broadcast shapes
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    d = np.asarray(d, dtype=float)
    if tp is None:
        tp = np.zeros_like(x)
    else:
        tp = np.asarray(tp, dtype=float)

    # Broadcast all to the same shape (works for scalars too)
    x, y, d, tp = np.broadcast_arrays(x, y, d, tp)

    # local trig
    cosd = np.cos(delta)
    sind = np.sin(delta)

    # define parameters with respect to the BOTTOM of the fault (MATLAB logic)
    # (note we operate on local arrays, not overwriting user inputs)
    d_bot = d + W * sind
    x_adj = x - W * cosd * np.cos(strike)
    y_adj = y + W * cosd * np.sin(strike)

    # add topography
    d_bot = d_bot + tp

    # rotate coordinates by (strike -> -strike + pi/2) as in MATLAB
    strike_rot = -strike + np.pi / 2.0
    coss = np.cos(strike_rot)
    sins = np.sin(strike_rot)

    # rotated coordinates
    rotx = x_adj * coss + y_adj * sins
    roty = -x_adj * sins + y_adj * coss

    # half-length and constant factor
    L = length / 2.0
    Const = -U / (2.0 * np.pi)

    # p and q as in MATLAB eqn (30)
    p_arr = roty * cosd + d_bot * sind
    q_arr = roty * sind - d_bot * cosd

    a = 1.0 - 2.0 * nu
    parvec = np.array([a, delta, fault_type])

    # call fBi for four corners (note fBi expects arrays/scalars and broadcasts)
    f1a, f2a, f3a = fBi(rotx + L, p_arr, parvec, rotx + L, q_arr)
    f1b, f2b, f3b = fBi(rotx + L, p_arr - W, parvec, rotx + L, q_arr)
    f1c, f2c, f3c = fBi(rotx - L, p_arr, parvec, rotx - L, q_arr)
    f1d, f2d, f3d = fBi(rotx - L, p_arr - W, parvec, rotx - L, q_arr)

    # displacement combinations (Eqns. 25-27 in MATLAB code)
    uxj = Const * (f1a - f1b - f1c + f1d)
    uyj = Const * (f2a - f2b - f2c + f2d)
    uz  = Const * (f3a - f3b - f3c + f3d)

    # rotate back horizontals to original coordinate system, apply HF
    ux = HF * (-uyj * sins + uxj * coss)
    uy = HF * ( uxj * sins + uyj * coss)

    return ux, uy, uz

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

    # plot / save as in your original forward_3D_dis
    if show_plot == 1:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)
        # your original plot_data accepted (axes[i], UTM_X, UTM_Y, arr,...)
        # build UTM_X and UTM_Y arrays in meters as centers
        UTM_X = np.arange(xmin, xmax + dx, dx)
        UTM_Y = np.arange(ymin, ymax + dy, dy)
        plot_data(axes[0], UTM_X, UTM_Y, ue, "West - East Component, cm", clim=(-50, 50))
        plot_data(axes[1], UTM_X, UTM_Y, un, "South - North Component, cm", clim=(-50, 50))
        plot_data(axes[2], UTM_X, UTM_Y, uz, "Vertical Component, cm", clim=(-50, 50))
        fig.suptitle("Forward 3-D Displacement Field", fontsize=16, weight="bold")
        plt.show()

    # project to LOS (same as original)
    track = np.array([190, -10]); look = np.array([30, 30])
    proj_vec1 = np.array([np.sin(np.deg2rad(look[0])) * np.cos(np.deg2rad(track[0])),
                          np.sin(np.deg2rad(look[0])) * np.sin(np.deg2rad(track[0])),
                          np.cos(np.deg2rad(look[0]))])
    proj_vec2 = np.array([np.sin(np.deg2rad(look[1])) * np.cos(np.deg2rad(track[1])),
                          np.sin(np.deg2rad(look[1])) * np.sin(np.deg2rad(track[1])),
                          np.cos(np.deg2rad(look[1]))])

    data_matrix = np.column_stack((De, Dn, Dz))
    data_dec = (data_matrix @ proj_vec1).reshape(ue.shape)
    data_asc = (data_matrix @ proj_vec2).reshape(ue.shape)

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