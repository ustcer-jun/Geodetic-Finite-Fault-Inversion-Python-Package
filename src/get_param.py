import sys

def parse_value(value):
    """将字符串转成数字或数字列表，如果已经是数字则直接返回"""
    if isinstance(value, (int, float)):
        return value
    if isinstance(value, str):
        parts = value.split()
        if len(parts) == 1:
            try:
                return int(parts[0])
            except ValueError:
                try:
                    return float(parts[0])
                except ValueError:
                    return parts[0]
        else:
            nums = []
            for p in parts:
                try:
                    nums.append(int(p))
                except ValueError:
                    nums.append(float(p))
            return nums
    return value  # 兜底
def get_param(cfg, keys):
    """从嵌套字典中获取参数值"""
    node = cfg
    for k in keys:
        node = node[k]
    return parse_value(node)

if __name__ == "__main__":
    from Read_Config import parse_config  # 直接复用你已有的函数

    # 支持命令行传入文件名
    filename = sys.argv[1] if len(sys.argv) > 1 else "config.inv"
    cfg = parse_config(filename)

    # 示例：读取 fault_params 的参数
    dw = get_param(cfg, ["forward","fault_params","top_patch_width"])
    dl = get_param(cfg, ["forward","fault_params","top_patch_length"])
    inc = get_param(cfg, ["forward","fault_params","patch_increment_factor"])
    amp = get_param(cfg, ["forward","fault_params","amplitude"])

    print("dw =", dw)
    print("dl =", dl)
    print("inc =", inc)
    print("amp =", amp)

    # 示例：读取 model_params
    num_of_patches = get_param(cfg, ["model","model_params","num_of_patches"])
    seis_fault = get_param(cfg, ["model","model_params","seis_fault"])
    num_of_faults = get_param(cfg, ["model","model_params","num_of_faults"])

    print("num_of_patches =", num_of_patches)
    print("seis_fault =", seis_fault)
    print("num_of_faults =", num_of_faults)

    # 示例：读取某个断层的 trace 参数
    trace1_x = get_param(cfg, ["seis_fault1","trace1","x"])
    trace1_y = get_param(cfg, ["seis_fault1","trace1","y"])
    trace1_z = get_param(cfg, ["seis_fault1","trace1","z"])

    print("seis_fault1 trace1 x =", trace1_x)
    print("seis_fault1 trace1 y =", trace1_y)
    print("seis_fault1 trace1 z =", trace1_z)