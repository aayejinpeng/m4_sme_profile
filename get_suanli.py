import json
import pandas as pd

# profile 文件路径
profile_file = "resnet50_profile_2025-09-17_17-33-13.json"

# 读取 JSON
with open(profile_file, "r") as f:
    prof_data = json.load(f)

records = []

for entry in prof_data:
    # 只处理 complete events
    if entry.get("ph") != "X":
        continue

    name = entry.get("name", "")
    args = entry.get("args", {})
    op_type = args.get("op_name", "")
    dur_ms = entry.get("dur", 0) / 1000.0  # 微秒 -> 毫秒

    # 默认 FLOPs 为 None
    flops = None

    # 如果是卷积
    if op_type.lower() in ["conv", "fusedconv"]:
        try:
            inp_shapes = args.get("input_type_shape", [])
            out_shapes = args.get("output_type_shape", [])
            if len(inp_shapes) >= 2 and len(out_shapes) >= 1:
                # 提取尺寸
                batch = inp_shapes[0]["float"][0]
                C_in = inp_shapes[1]["float"][1] if len(inp_shapes[1]["float"]) >= 2 else inp_shapes[1]["float"][0]
                K_h = inp_shapes[1]["float"][-2]
                K_w = inp_shapes[1]["float"][-1]
                C_out = out_shapes[0]["float"][1] if len(out_shapes[0]["float"]) >= 2 else out_shapes[0]["float"][0]
                H_out = out_shapes[0]["float"][2]
                W_out = out_shapes[0]["float"][3]
                # 计算 FLOPs
                flops = 2 * batch * C_in * K_h * K_w * H_out * W_out * C_out
        except:
            flops = None

    # 如果是 MatMul
    if op_type.lower() == "matmul":
        try:
            inp_shapes = args.get("input_type_shape", [])
            if len(inp_shapes) >= 2:
                M = inp_shapes[0]["float"][0]
                K = inp_shapes[0]["float"][1]
                N = inp_shapes[1]["float"][1]
                flops = 2 * M * N * K
        except:
            flops = None

    # 实际 GFLOPS
    gflops = flops / (dur_ms / 1000) / 1e9 if flops is not None else None

    records.append({
        "name": name,
        "op_type": op_type,
        "duration_ms": dur_ms,
        "FLOPs": flops,
        "GFLOPS": gflops
    })

# 转 DataFrame
df = pd.DataFrame(records)
df_sorted = df.sort_values(by="duration_ms", ascending=False)

# 打印前 20 层耗时算子
print(df_sorted.head(20))

# 保存 CSV
df_sorted.to_csv("resnet50_layer_gflops.csv", index=False)
print("Saved profile table to resnet50_layer_gflops.csv")
