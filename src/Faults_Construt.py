import numpy as np
import os
from Read_Config import  parse_config
 # 之前帮你写的配置文件读取函数
from src import generate_grid, write_inv, plot_patches # 之前写的构建断层网格函数

def build_fault_model(config_path, output_inv="SSD_model.inv"):
    # 1. 读取配置文件
    cfg = parse_config(config_path)

    # 断层网格参数
    dw = float(cfg["forward"]["fault_params"]["top_patch_width"])
    dl = float(cfg["forward"]["fault_params"]["top_patch_length"])
    inc = float(cfg["forward"]["fault_params"]["patch_increment_factor"])
    amp = float(cfg["forward"]["fault_params"]["amplitude"])

    # 模型参数
    num_of_patches = int(cfg["model"]["model_params"]["num_of_patches"])
    # fault_type = int(cfg["model"]["model_params"]["fault_type"])
    slip_modes = np.atleast_1d(np.array(cfg["model"]["slip_patch"]["slip_modes"], dtype=int))
    seis_fault = np.atleast_1d(np.array(cfg["model"]["model_params"]["seis_fault"], dtype=int))
    num_of_faults = np.atleast_1d(np.array(cfg["model"]["model_params"]["num_of_faults"], dtype=int))
    num_of_seis_faults = int(cfg["model"]["model_params"]["num_of_seis_faults"])
    x_decay_facs = np.atleast_1d(np.array(cfg["model"]["slip_patch"]["x_decay_facs"], dtype=float))
    y_decay_facs = np.atleast_1d(np.array(cfg["model"]["slip_patch"]["y_decay_facs"], dtype=float))
    z_decay_facs = np.atleast_1d(np.array(cfg["model"]["slip_patch"]["z_decay_facs"], dtype=float))
    threshold1s = np.atleast_1d(np.array(cfg["model"]["slip_patch"]["threshod1s"], dtype=float))
    threshold2s = np.atleast_1d(np.array(cfg["model"]["slip_patch"]["threshod2s"], dtype=float))

    # 2. 读取每个断层的几何参数
    patches = []
    for i in range(1, num_of_seis_faults+1):
        for j in range(1, num_of_faults[i-1]+1):
            sec = f"seis_fault{i}"
            trace = f"trace{j}"
            patch = {
                "x": float(cfg[sec][trace]["x"]),
                "y": float(cfg[sec][trace]["y"]),
                "z": float(cfg[sec][trace]["z"]),
                "len": float(cfg[sec][trace]["len"]),
                "wid": float(cfg[sec][trace]["wid"]),
                "dip": float(cfg[sec][trace]["dip"]),
                "strike": float(cfg[sec][trace]["strike"]),
            }
            if patch["strike"] == 90:
                patch["strike"] = 89.9
            patches.append(patch)

    print(f"Your seismogenic fault is trace{{{seis_fault}}}")

    # 3. 构建每个地震断层的几何中心
    Xo, Yo, Zo = [], [], []
    for j in range(num_of_seis_faults):
        xs, ys, zs, xb, yb, zb, ll, ww, nw, nl, nL, dL, dW = generate_grid(1, patches[seis_fault[j]-1], dw, dl, inc, 1)
        Xo.append(np.mean(xs))
        Yo.append(np.mean(ys))
        Zo.append(np.mean(zs))

    # 4. 初始化全局数组
    XS, YS, ZS = [], [], []
    XB, YB, ZB = [], [], []
    LL, WW = [], []
    DIP, STRIKE = [], []
    num_grid = []
    SS, SD = [], []

    # 5. 遍历所有断层
    for k in range(num_of_seis_faults):
        mode = slip_modes[k]
        x_decay_fac = x_decay_facs[k]
        y_decay_fac = y_decay_facs[k]
        z_decay_fac = z_decay_facs[k]
        threshold1 = threshold1s[k]
        threshold2 = threshold2s[k]
        xo, yo, zo = Xo[k], Yo[k], Zo[k]

        for j in range(num_of_faults[k]):
            idx = sum(num_of_faults[:k]) + j
            patch = patches[idx]

            # 特殊情况：fault_type=2
            # if fault_type == 2 and j == 0:
            #     dw, dl = 3.0e4, 4.444444e4
            # elif fault_type == 2 and j == 1:
            #     dw, dl = 2.366890e4, 4.0e4
            # else:
            dw = float(cfg["forward"]["fault_params"]["top_patch_width"])
            dl = float(cfg["forward"]["fault_params"]["top_patch_length"])

            xs, ys, zs, xb, yb, zb, ll, ww, nw, nl, nL, dL, dW = generate_grid(
                1, patch, dw, dl, inc, 1
            )

            # 累积保存几何
            XS.extend(xs); YS.extend(ys); ZS.extend(zs)
            XB.extend(xb); YB.extend(yb); ZB.extend(zb)
            LL.extend(ll); WW.extend(ww)
            DIP.extend([patch["dip"]] * len(xs))
            STRIKE.extend([patch["strike"]] * len(xs))
            num_grid.append(len(xs))

            # 初始化 slip
            ss = np.zeros(len(xs))
            sd = np.zeros(len(xs))

            # 遍历层
            index = np.concatenate(([0], np.cumsum(nL)))
            for iLayer in range(nw):
                idx_range = range(index[iLayer], index[iLayer+1])
                xs_layer, ys_layer, zs_layer = xs[idx_range], ys[idx_range], zs[idx_range]

                # 层的几何中心
                if nL[iLayer] % 2 == 1:
                    center_x = xs[index[iLayer] + nL[iLayer]//2]
                    center_y = ys[index[iLayer] + nL[iLayer]//2]
                else:
                    center_x = 0.5 * (xs[index[iLayer] + nL[iLayer]//2 - 1] + xs[index[iLayer] + nL[iLayer]//2])
                    center_y = 0.5 * (ys[index[iLayer] + nL[iLayer]//2 - 1] + ys[index[iLayer] + nL[iLayer]//2])

                # 三种 slip 模式
                if mode == 1:  # Gaussian (single)
                    dx = (xs_layer - xo) / x_decay_fac
                    dy = (ys_layer - yo) / y_decay_fac
                    dz = (zs_layer - zo + 5e3) / z_decay_fac
                    c_vec = np.sqrt(dx**2 + dy**2 + dz**2)
                    sd[idx_range] = amp * np.exp(-(c_vec**2))

                elif mode == 2:  # Layered Gaussian
                    top_z = zs_layer[0]
                    if top_z >= threshold1:
                        dx = (xs_layer - center_x) / x_decay_fac
                        dy = (ys_layer - center_y) / y_decay_fac
                        c_vec = np.sqrt(dx**2 + dy**2)
                        sd[idx_range] = amp * np.exp(-(c_vec**2))
                    elif top_z <= threshold2:
                        sd[idx_range] = 0
                    else:
                        a = amp * (top_z - threshold2) / (threshold1 - threshold2)
                        dx = (xs_layer - xo) / x_decay_fac
                        dy = (ys_layer - yo) / y_decay_fac
                        dz = (zs_layer - threshold1) / z_decay_fac
                        c_vec = np.sqrt(dx**2 + dy**2 + dz**2)
                        sd[idx_range] = a * np.exp(-(c_vec**2))

                elif mode == 3:  # Layered Gaussian + SSD
                    top_z = zs_layer[0]
                    if top_z >= threshold1:
                        a = amp * ((top_z - threshold1)/threshold2 + 1)
                        dx = (xs_layer - xo) / x_decay_fac
                        dy = (ys_layer - yo) / y_decay_fac
                        horiz = np.sqrt(dx**2 + dy**2)
                        zterm = (zs_layer - threshold1)**2 / ((z_decay_fac/2)**2)
                        c_vec = horiz + zterm
                        sd[idx_range] = a * np.exp(-(c_vec**2))
                    elif top_z <= threshold2:
                        sd[idx_range] = 0
                    else:
                        a = amp * (top_z - threshold2) / (threshold1 - threshold2)
                        dx = (xs_layer - xo) / x_decay_fac
                        dy = (ys_layer - yo) / y_decay_fac
                        horiz = np.sqrt(dx**2 + dy**2)
                        zterm = (zs_layer - threshold1)**2 / (z_decay_fac**2)
                        c_vec = horiz + zterm
                        sd[idx_range] = a * np.exp(-(c_vec**2))

            SS.extend(ss)
            SD.extend(sd)

    # 6. 保存结果
    # np.savez(output_model,
    #          patch=patches, XS=XS, YS=YS, ZS=ZS,
    #          XB=XB, YB=YB, ZB=ZB,
    #          LL=LL, WW=WW,
    #          DIP=DIP, STRIKE=STRIKE,
    #          num_grid=num_grid, SS=SS, SD=SD)

    # TODO: write_inv() -> 你如果已有 Python 版写 inv 文件的函数，可以在这里调用
    write_inv(output_inv, XS, YS, ZS, LL, WW, DIP, STRIKE, SS, SD, 0, num_grid)

    print(f"Model saved to {output_inv}")

    # ===== 使用示例 =====
if __name__ == "__main__":
    config_file = "config.inv";
    build_fault_model(config_file, output_inv="test_model.inv");
    plot_patches("test_model.inv",23,5e-3);