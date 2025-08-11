# 文件: forwards/dnf_forward.py (Jittor版本)

from utils.registry import FORWARD_REGISTRY
# Jittor改动: 导入jittor用于性能分析
import jittor as jt
from jittor import nn


@FORWARD_REGISTRY.register(suffix='DNF')
def train_forward(config, model, data):
    # Jittor改动: 无需手动.cuda()，Jittor会自动处理设备
    raw = data['noisy_raw']
    raw_gt = data['clean_raw']
    rgb_gt = data['clean_rgb']
    
    rgb_out, raw_out = model(raw)
    
    ###### | output                         | label
    return {'rgb': rgb_out, 'raw': raw_out}, {'rgb': rgb_gt, 'raw': raw_gt}


@FORWARD_REGISTRY.register(suffix='DNF')
def test_forward(config, model, data):
    # Jittor改动: 无需手动.cuda()
    raw = data['noisy_raw']
    raw_gt = data['clean_raw']
    rgb_gt = data['clean_rgb']
    img_files = data['img_file']
    lbl_files = data['lbl_file']

    rgb_out, raw_out = model(raw)

    ###### | output                         | label                              | img and label names
    return {'rgb': rgb_out, 'raw': raw_out}, {'rgb': rgb_gt, 'raw': raw_gt}, img_files, lbl_files


@FORWARD_REGISTRY.register(suffix='DNF')  # without label, for inference only
def inference(config, model, data):
    # Jittor改动: 无需手动.cuda()
    raw = data['noisy_raw']
    img_files = data['img_file']

    rgb_out, raw_out = model(raw)

    ###### | output                         | img names
    return {'rgb': rgb_out, 'raw': raw_out}, img_files





from collections import defaultdict


# 口径：1 次乘加 = 1 FLOP（更贴近 fvcore 输出；如需乘加记 2，把它改为 2）
FLOPS_PER_MAC = 1

def _numel(x):
    if isinstance(x, jt.Var):
        return int(x.numel())
    if isinstance(x, (tuple, list)) and x and isinstance(x[0], jt.Var):
        return int(sum(int(v.numel()) for v in x))
    return 0

def _shape(x):
    if isinstance(x, jt.Var):
        return list(x.shape)
    if isinstance(x, (tuple, list)) and x and isinstance(x[0], jt.Var):
        return [list(v.shape) for v in x]
    return None

def _fmt_units(n, unit_for_params=True):
    # params: K/M; flops: K/M/G。保留 3 位小数，去掉尾随 0
    n = float(n)
    units = ["", "K", "M", "G"] if not unit_for_params else ["", "K", "M"]
    base = 1000.0
    i = 0
    while i+1 < len(units) and n >= base:
        n /= base
        i += 1
    s = f"{n:.3f}".rstrip("0").rstrip(".")
    return f"{s}{units[i]}"

def _fmt_table(rows):
    """
    rows: list of [module_name, param_str, flop_str]
    返回对齐的 markdown 表格字符串
    """
    # 计算每列最大宽度
    col_widths = [0, 0, 0]
    for r in rows:
        for i, cell in enumerate(r):
            col_widths[i] = max(col_widths[i], len(cell))

    # 表头
    header = ["module", "#parameters or shape", "#flops"]
    col_widths = [max(col_widths[i], len(header[i])) for i in range(3)]

    head_line = "| " + " | ".join(f"{header[i]:<{col_widths[i]}}" for i in range(3)) + " |"
    align_line = "|:" + ":|:".join("-" * (col_widths[i]-1) for i in range(3)) + ":|"

    # 内容
    body_lines = []
    for r in rows:
        body_lines.append("| " + " | ".join(f"{r[i]:<{col_widths[i]}}" for i in range(3)) + " |")

    return "\n".join([head_line, align_line] + body_lines)

def _build_name_maps(model):
    """返回：
       - name_map: id(module) -> module_path_name（'' 为根）
       - children: parent_name -> [child_name,...]
    """
    name_map = {}
    children = defaultdict(list)
    for name, m in model.named_modules():  
        name_map[id(m)] = name  # 根模块 name == ''
    # build children map
    all_names = [n for n in name_map.values()]
    for n in all_names:
        if n == "":
            continue
        parent = n.rsplit(".", 1)[0] if "." in n else ""
        children[parent].append(n)
    # 排序保证稳定输出
    for k in children:
        children[k].sort()
    return name_map, children

def _gather_param_infos(model):
    """返回：
       - param_total: module_name -> (sum of params under subtree)
       - param_direct_shapes: module_name -> list of (param_basename, shape_str)
    """
    # 收集所有参数（全路径名）
    named_params = list(model.named_parameters())
    # 统计总参数：对每个模块名，累计所有以该模块名为前缀的参数
    param_total = defaultdict(int)
    # 记录直接参数：x.y.weight -> 只归入模块 x.y，且记形状
    param_direct_shapes = defaultdict(list)

    for full_name, p in named_params:
        # full_name 可能是 "layer.weight" 或 "block.0.conv.weight"
        if "." in full_name:
            mod_name, base = full_name.rsplit(".", 1)
        else:
            mod_name, base = "", full_name  # 根模块的直接参数
        # 直接参数形状
        try:
            shp = tuple(int(s) for s in p.shape)
        except:
            shp = tuple(p.shape)
        param_direct_shapes[mod_name].append((base, f"({', '.join(map(str, shp))})"))

        # 累计到所有祖先（含根）
        parts = mod_name.split(".") if mod_name != "" else []
        for k in range(len(parts)+1):
            anc = ".".join(parts[:k])
            param_total[anc] += int(p.numel())

    return param_total, param_direct_shapes

def _leaf_modules(model):
    leaves = set()
    for m in model.modules():
        if len(list(m.children())) == 0:
            leaves.add(m)
    return leaves

def _make_flops_hook(name, flops_map):
    def hook(mod, args, out):
        flops = 0
        try:
            # Conv
            if isinstance(mod, nn.Conv):
                oshp = _shape(out)
                if isinstance(oshp, list) and len(oshp) == 4:
                    _, C_out, H_out, W_out = oshp
                else:
                    raise RuntimeError("Unsupported Conv output shape")
                Cin_per_group = mod.in_channels // mod.groups
                ksz = mod.kernel_size
                if isinstance(ksz, tuple):
                    kH, kW = ksz
                else:
                    kH = kW = int(ksz)
                macs = Cin_per_group * kH * kW * H_out * W_out * C_out
                flops = macs * FLOPS_PER_MAC

            # Linear
            elif isinstance(mod, nn.Linear):
                oshp = _shape(out)
                Cin = mod.in_features
                Cout = mod.out_features
                N = oshp[0] if isinstance(oshp, list) and len(oshp) >= 1 else 1
                macs = int(N) * Cin * Cout
                flops = macs * FLOPS_PER_MAC

            # Pool（粗略）
            elif isinstance(mod, (nn.Pool, nn.AdaptiveAvgPool2d, nn.AdaptiveMaxPool2d)):
                elems = _numel(out)
                if hasattr(mod, "kernel_size"):
                    if isinstance(mod.kernel_size, tuple):
                        kH, kW = mod.kernel_size
                    else:
                        kH = kW = int(mod.kernel_size)
                    flops = elems * (kH * kW)
                else:
                    flops = elems

            # Upsample（近似）
            elif isinstance(mod, nn.Upsample):
                elems = _numel(out)
                flops = elems * 4

            # BN / 激活 / Pad / 形状操作：不给 FLOPs 或按元素近似也可以，但为了贴近 fvcore，这里统一记 0
            else:
                flops = 0
        except:
            flops = 0
        flops_map[name] += int(flops)
    return hook

@FORWARD_REGISTRY.register()
def DNF_profile(config, model, data, logger):
    # 1) 准备输入
    jt.flags.use_cuda = 1
    x = data['noisy_raw']
    if not isinstance(x, jt.Var):
        x = jt.array(x)

    # 2) 名称与树结构
    name_map, children = _build_name_maps(model)
    # 3) 参数统计
    param_total, param_direct_shapes = _gather_param_infos(model)

    # 4) 只给叶子模块挂 hook 收集 FLOPs
    flops_map_leaf = defaultdict(int)
    handles = []
    for name, m in model.named_modules():
        if name == "":
            continue
        if len(list(m.children())) == 0:
            h = m.register_forward_hook(_make_flops_hook(name, flops_map_leaf))
            handles.append(h)

    # 5) 前向一次
    with jt.no_grad():
        _ = model(x)

    # 6) 去钩子
    for h in handles:
        try:
            h.remove_forward_hook()
        except:
            pass

    # 7) 聚合 FLOPs 到各级模块（含根 'model'）
    flops_total_map = defaultdict(int)
    # 叶子名集合
    leaf_names = set(flops_map_leaf.keys())
    # 对每个模块名，累计所有以其为前缀的叶子 FLOPs
    all_module_names = [n for n, _ in model.named_modules()]  # 包含 '' 根
    for mod_name in all_module_names:
        prefix = (mod_name + ".") if mod_name != "" else ""
        s = 0
        for ln in leaf_names:
            if ln == mod_name or ln.startswith(prefix):
                s += flops_map_leaf[ln]
        flops_total_map[mod_name] = s

    # 8) 生成表格行
    rows = []

    def emit_module_row(name, depth):
        # name == "" 表示根
        title = "model" if name == "" else name
        pcount = param_total.get(name, 0)
        fcount = flops_total_map.get(name, 0)

        # ① 模块总览行：三列
        rows.append([
            " " * (2*depth) + title,                       # 模块名（带缩进）
            _fmt_units(pcount, unit_for_params=True),      # 参数量（K/M）
            _fmt_units(fcount, unit_for_params=False),     # FLOPs（K/M/G）
        ])

        # ② 直接参数形状行（第三列留空）
        for pname, pshape in param_direct_shapes.get(name, []):
            pname_full = (f"{title}.{pname}" if name != "" else f"model.{pname}")
            rows.append([
            " " * (2*(depth+1)) + pname_full,              # 参数名（带更深缩进）
            pshape,                                        # "(..., ...)" 形状
            ""                                             # FLOPs 空
        ])

    # 深度优先遍历
    def dfs(name, depth):
        emit_module_row(name, depth)
        for ch in children.get(name, []):
            dfs(ch, depth+1)

    dfs("", 0)  # 从根开始

    # 9) 输出
    logger.info("Detaild FLOPs:\n" + _fmt_table(rows))
    # 总 FLOPs：根的累计
    total_flops = flops_total_map.get("", 0)
    logger.info(f"Total FLOPs: {int(total_flops):,}")