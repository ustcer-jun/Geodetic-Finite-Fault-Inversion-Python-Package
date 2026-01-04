import re
import json

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False


def parse_config(filename):
    """
    解析自定义格式的配置文件
    返回: dict 格式
    """
    config = {}
    current_section = None
    current_subsection = None

    def convert_value(val):
        """尝试把字符串转成 int 或 float"""
        try:
            if "." in val or "e" in val or "E" in val:
                return float(val)
            else:
                return int(val)
        except ValueError:
            return val  # 保持原字符串

    with open(filename, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):  # 跳过空行和注释
                continue

            # 区块 [forward], [model], [seis_fault1] ...
            if line.startswith("[") and line.endswith("]"):
                current_section = line[1:-1].strip()
                config[current_section] = {}
                current_subsection = None
                continue

            # 子区块 {fault_params}, {smooth}, {trace1} ...
            if line.startswith("{") and line.endswith("}"):
                current_subsection = line[1:-1].strip()
                config[current_section][current_subsection] = {}
                continue

            # 参数行: key = value(s)
            if "=" in line:
                key, value = line.split("=", 1)
                key = key.strip()
                values = value.split()

                # 单个值 → 转数字；多个值 → 列表
                if len(values) == 1:
                    val = convert_value(values[0])
                else:
                    val = [convert_value(v) for v in values]

                # 存储到对应位置
                if current_subsection:
                    config[current_section][current_subsection][key] = val
                else:
                    config[current_section][key] = val

    return config

def save_config(config, filename, fmt="json"):
    """
    保存 config 到文件
    fmt = "json" 或 "yaml"
    """
    if fmt == "json":
        with open(filename, "w") as f:
            json.dump(config, f, indent=4)
    elif fmt == "yaml":
        if not HAS_YAML:
            raise ImportError("未安装 pyyaml，请先 `pip install pyyaml`")
        with open(filename, "w") as f:
            yaml.safe_dump(config, f, sort_keys=False, allow_unicode=True)
    else:
        raise ValueError("不支持的格式: " + fmt)

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

# ===== 使用示例 =====
if __name__ == "__main__":
    cfg = parse_config("config.inv")

    # 示例: 获取参数
    print("top_patch_width =", cfg["forward"]["fault_params"]["top_patch_width"])
    print("threshod1s =", cfg["model"]["slip_patch"]["threshod1s"])
    # 保存为 JSON
    save_config(cfg, "config.json", fmt="json")

    # 保存为 YAML（需要 pip install pyyaml）
    if HAS_YAML:
        save_config(cfg, "config.yaml", fmt="yaml")
