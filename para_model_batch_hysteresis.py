# -*- coding: utf-8 -*-
"""
批量纤维模型滞回分析脚本
基于 para_model.py 的建模方式改写：
1. 按参数表逐行批量建模；
2. 使用统一的位移角加载制度，不再依赖实验 dispData / forceData；
3. 输出每个试件的滞回曲线数据、骨架图、截面图（可选）与汇总表。

单位体系：
- 长度: mm
- 力: N
- 强度: MPa = N/mm^2
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import openseespy.opensees as ops

try:
    import opsvis as opsv
except Exception:
    opsv = None

plt.rcParams["font.family"] = ["Times New Roman", "SimSun"]
plt.rcParams["mathtext.fontset"] = "cm"

DRIFT_LEVELS = [
    0.0005,
    0.0010,
    0.0015,
    0.0020,
    0.0025,
    0.0050,
    0.0075,
    0.0100,
    0.0150,
    0.0200,
    0.0250,
    0.0350,
]


def safe_float(row: Dict[str, str], key: str, default: float = 0.0) -> float:
    try:
        val = row.get(key, default)
        if val is None or str(val).strip() == "":
            return float(default)
        return float(val)
    except Exception:
        return float(default)


def safe_int(row: Dict[str, str], key: str, default: int = 0) -> int:
    return int(round(safe_float(row, key, default)))


def read_rows(csv_path: str) -> List[Dict[str, str]]:
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def build_displacement_history(height_for_drift: float, cycles_per_level: int = 1) -> np.ndarray:
    history = [0.0]
    for drift in DRIFT_LEVELS:
        u = drift * height_for_drift
        for _ in range(cycles_per_level):
            history.extend([u, 0.0, -u, 0.0])
    return np.array(history, dtype=float)


def compute_mander_params(row: Dict[str, str]) -> Dict[str, float]:
    lend = safe_float(row, "lend")
    tw = safe_float(row, "tw")
    lw = safe_float(row, "lw")
    hw = safe_float(row, "hw")
    c = safe_float(row, "c")
    nlc = safe_float(row, "nlc")
    dlc = safe_float(row, "dlc")
    slc = safe_float(row, "slc")
    contype = safe_int(row, "contype")
    sconf = safe_float(row, "sconf")
    dconf = safe_float(row, "dconf")
    fc = safe_float(row, "f'c")
    fyconf = safe_float(row, "fyconf")
    Ec = safe_float(row, "Ec")
    hl = safe_float(row, "hl", 0.0)

    cd = c + dlc / 2.0 + dconf

    bc = max(lend - 2 * c, 1.0)
    dc = max(tw - 2 * c - dconf, 1.0)
    s = max(sconf, 1.0)
    sp = max(sconf - dconf, 1.0)
    ec0 = 0.002

    Acc = bc * dc - nlc * math.pi * dlc**2 / 4.0
    Acc = max(Acc, 1.0)
    Ai = (nlc / 2 - 1) * max(slc - dlc, 0.0) ** 2 / 3.0 + max(tw - 2 * (c + dlc + dconf), 0.0) ** 2 / 3.0
    Ae = (bc * dc - Ai) * (1 - sp / (2 * bc)) * (1 - sp / (2 * dc))
    Ae = max(Ae, 0.0)
    ke = max(min(Ae / Acc, 1.0), 0.0)

    Asx = 2 * math.pi * dconf**2 / 4.0
    Asy = max(contype, 1) * math.pi * dconf**2 / 4.0
    rx = Asx / (s * dc)
    ry = Asy / (bc * s)
    flx = rx * fyconf
    fly = ry * fyconf
    flxp = ke * flx
    flyp = ke * fly

    fco = fc
    if max(flxp, flyp) == 0:
        fcc = fco
        ecc = ec0
    else:
        r1 = min(flxp, flyp) / max(flxp, flyp)
        xba = (flxp + flyp) / 2.0 / max(fco, 1e-6)
        A = 6.8886 - (0.6069 + 17.275 * r1) * math.exp(-4.989 * r1)
        B = 4.5 / (5 / A * (0.9849 - 0.6306 * math.exp(-3.8939 * r1) - 0.1)) - 5
        fccfco = 1 + A * xba * (0.1 + 0.9 / (1 + B * xba))
        fcc = fccfco * fc
        ecc = ec0 * (1 + 5 * (fcc / max(fc, 1e-6) - 1))

    return {
        "lend": lend,
        "tw": tw,
        "lw": lw,
        "hw": hw,
        "hl": hl,
        "c": c,
        "cd": cd,
        "nlc": nlc,
        "dlc": dlc,
        "fc": fc,
        "fcc": fcc,
        "ecc": ecc,
        "Ec": Ec,
    }


def setup_model(row: Dict[str, str], out_dir: Path, plot_section: bool = False) -> Dict[str, float]:
    ops.wipe()
    ops.model("basic", "-ndm", 2, "-ndf", 3)

    name = str(row.get("name", "specimen"))
    pconf = compute_mander_params(row)

    lend = pconf["lend"]
    tw = pconf["tw"]
    lw = pconf["lw"]
    hw = pconf["hw"]
    hl = pconf["hl"]
    cd = pconf["cd"]
    fc = pconf["fc"]
    fcc = pconf["fcc"]
    ecc = pconf["ecc"]
    Ec = pconf["Ec"]

    nlc = safe_float(row, "nlc")
    dlc = safe_float(row, "dlc")
    nlw = safe_float(row, "nlw")
    dweb = safe_float(row, "dweb")
    dconf = safe_float(row, "dconf")
    fylc = safe_float(row, "fylc")
    fylw = safe_float(row, "fylw")
    N = safe_float(row, "N")

    height = hw
    cover = cd
    bw = tw

    As_end = math.pi * dlc * dlc / 4.0
    As_web = math.pi * dweb * dweb / 4.0
    n_end = nlc / 2.0
    n_web = nlw / 2.0

    ops.node(1, 0.0, 0.0)
    ops.node(2, 0.0, height)
    ops.fix(1, 1, 1, 1)

    fc_confined = round(fcc, 3)
    fc_unconfined = fc
    ec0_confined = max(round(ecc, 6), 0.002)
    ec0_unconfined = 0.002

    ops.uniaxialMaterial("Concrete01", 1, -fc_confined, -ec0_confined, -5.0, -0.05)
    ops.uniaxialMaterial("Concrete01", 2, -fc_unconfined, -ec0_unconfined, -3.0, -0.05)

    E = 200600.0
    ops.uniaxialMaterial("Steel02", 3, fylc, E, 0.015, 18.5, 0.925, 0.15)
    ops.uniaxialMaterial("Steel02", 4, fylw, E, 0.015, 18.5, 0.925, 0.15)

    V = 0.2
    G = Ec / (2.0 * (1.0 + V)) * lw * bw
    ops.uniaxialMaterial("Elastic", 256, G)

    y1 = lw / 2.0
    z1 = tw / 2.0
    nf_core_x = max(1, round((lend - 2 * cover) / 100))
    nf_cover_x = max(1, round(2 * y1 / 100))
    nf_cover_z = max(1, round(2 * (z1 - cover) / 100))
    nf_mid_x = max(1, round(2 * (y1 - lend + cover) / 100))

    fib_sec_1 = [
        ["section", "Fiber", 1, "-GJ", Ec * tw * lw**3 / 12.0],
        ["patch", "rect", 1, nf_core_x, 2, cover - y1, cover - z1, -y1 + lend - cover, z1 - cover],
        ["patch", "rect", 1, nf_core_x, 2, y1 - lend + cover, cover - z1, y1 - cover, z1 - cover],
        ["patch", "rect", 2, nf_cover_x, 1, -y1, z1 - cover, y1, z1],
        ["patch", "rect", 2, nf_cover_x, 1, -y1, -z1, y1, cover - z1],
        ["patch", "rect", 2, 1, nf_cover_z, -y1, cover - z1, cover - y1, z1 - cover],
        ["patch", "rect", 2, 1, nf_cover_z, y1 - cover, cover - z1, y1, z1 - cover],
        ["patch", "rect", 2, nf_mid_x, 2, -y1 + lend - cover, cover - z1, y1 - lend + cover, z1 - cover],
        ["layer", "straight", 3, round(n_end), As_end, -(y1 - cover), z1 - cover, -(y1 + cover - lend), z1 - cover],
        ["layer", "straight", 3, round(n_end), As_end, -(y1 - cover), -(z1 - cover), -(y1 + cover - lend), -(z1 - cover)],
        ["layer", "straight", 3, round(n_end), As_end, y1 + cover - lend, z1 - cover, y1 - cover, z1 - cover],
        ["layer", "straight", 3, round(n_end), As_end, y1 + cover - lend, -(z1 - cover), y1 - cover, -(z1 - cover)],
        ["layer", "straight", 4, round(n_web), As_web, -y1 - cover + lend, z1 - cover, y1 + cover - lend, z1 - cover],
        ["layer", "straight", 4, round(n_web), As_web, -y1 - cover + lend, cover - z1, y1 + cover - lend, cover - z1],
    ]

    if opsv is not None:
        opsv.fib_sec_list_to_cmds(fib_sec_1)
        if plot_section:
            try:
                matcolor = ["gold", "lightgrey", "gold", "gold", "w", "w", "w", "w"]
                plt.figure(figsize=(5, 3))
                opsv.plot_fiber_section(fib_sec_1, matcolor=matcolor)
                plt.axis("equal")
                plt.tight_layout()
                plt.savefig(out_dir / f"{name}_fiber_section.png", dpi=200, bbox_inches="tight")
                plt.close()
            except Exception:
                plt.close("all")
    else:
        raise RuntimeError("未安装 opsvis，无法由 fiber list 创建截面命令。请先安装 opsvis。")

    ops.section("Aggregator", 12, 256, "Vy", "-section", 1)
    ops.geomTransf("Linear", 1)
    ops.beamIntegration("Lobatto", 1, 12, 6)
    ops.element("forceBeamColumn", 1000, 1, 2, 1, 1)

    ops.timeSeries("Linear", 1)
    ops.pattern("Plain", 1, 1)
    ops.load(2, 0.0, -N, 0.0)

    return {
        "name": name,
        "hw": hw,
        "hl": hl,
        "N": N,
    }


def run_gravity() -> bool:
    ops.system("BandGeneral")
    ops.constraints("Transformation")
    ops.numberer("RCM")
    ops.test("NormDispIncr", 1.0e-10, 50, 0)
    ops.algorithm("Newton")
    ops.integrator("LoadControl", 0.1)
    ops.analysis("Static")
    ok = ops.analyze(10)
    if ok == 0:
        ops.loadConst("-time", 0.0)
    return ok == 0


def setup_cyclic_recorders(case_dir: Path) -> None:
    ops.recorder("Node", "-file", str(case_dir / "top_disp.txt"), "-node", 2, "-dof", 1, "disp")
    ops.recorder("Node", "-file", str(case_dir / "base_force.txt"), "-node", 1, "-dof", 1, "reaction")
    ops.recorder("Element", "-file", str(case_dir / "element_forces.out"), "-ele", 1000, "force")
    ops.recorder("Element", "-file", str(case_dir / "fiber_data.out"), "-ele", 1000, "section", 1, "fiberStressStrain")


def run_cyclic_analysis(disp_history: np.ndarray, step_divisor: int = 10, max_subdivide: int = 8) -> Tuple[bool, int, float]:
    ops.timeSeries("Linear", 2)
    ops.pattern("Plain", 2, 2)
    ops.load(2, 1.0, 0.0, 0.0)

    ops.system("BandGeneral")
    ops.numberer("RCM")
    ops.constraints("Transformation")
    ops.test("NormDispIncr", 1e-5, 50)
    ops.algorithm("KrylovNewton")
    ops.integrator("DisplacementControl", 2, 1, 0.0)
    ops.analysis("Static")

    current_disp = 0.0
    finished_steps = 0

    for idx, target_disp in enumerate(disp_history[1:], start=1):
        delta_disp = float(target_disp - current_disp)
        ref = max(abs(target_disp) / step_divisor, 0.02)
        n_incr = max(1, int(math.ceil(abs(delta_disp) / ref)))
        ok = -1

        for subdiv in range(max_subdivide):
            trial_n = n_incr * (2 ** subdiv)
            d_step = delta_disp / trial_n
            success = True
            for _ in range(trial_n):
                ops.integrator("DisplacementControl", 2, 1, d_step)
                ok = ops.analyze(1)
                if ok != 0:
                    success = False
                    break
            if success:
                current_disp = target_disp
                finished_steps = idx
                ok = 0
                break
            ops.test("NormDispIncr", 1e-5, 100)
            ops.algorithm("NewtonLineSearch", 0.8)

        if ok != 0:
            return False, finished_steps, current_disp

        ops.test("NormDispIncr", 1e-5, 50)
        ops.algorithm("KrylovNewton")

    return True, finished_steps, current_disp


def postprocess_case(case_dir: Path, name: str) -> Dict[str, float]:
    out = {"max_abs_disp_mm": np.nan, "max_abs_force_N": np.nan, "energy_Nmm": np.nan}
    disp_path = case_dir / "top_disp.txt"
    force_path = case_dir / "base_force.txt"
    if not (disp_path.exists() and force_path.exists()):
        return out

    sim_disp = np.loadtxt(disp_path)
    sim_force = np.loadtxt(force_path)
    sim_disp = np.atleast_1d(sim_disp)
    sim_force = np.atleast_1d(sim_force)
    if sim_disp.ndim > 1:
        sim_disp = sim_disp[:, -1]
    if sim_force.ndim > 1:
        sim_force = sim_force[:, -1]

    n = min(len(sim_disp), len(sim_force))
    sim_disp = sim_disp[:n]
    sim_force = -sim_force[:n]

    if n:
        out["max_abs_disp_mm"] = float(np.max(np.abs(sim_disp)))
        out["max_abs_force_N"] = float(np.max(np.abs(sim_force)))
    if n > 1:
        out["energy_Nmm"] = float(np.sum(np.abs(0.5 * (sim_force[1:] + sim_force[:-1]) * np.diff(sim_disp))))

    plt.figure(figsize=(5, 4))
    plt.plot(sim_disp, sim_force, "r-", lw=1.0)
    plt.xlabel("位移 (mm)")
    plt.ylabel("底部反力 (N)")
    plt.title(f"{name} 滞回曲线")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(case_dir / f"{name}_hysteresis.png", dpi=200, bbox_inches="tight")
    plt.close()
    return out


def analyze_one_row(row: Dict[str, str], case_dir: Path, cycles_per_level: int, plot_section: bool) -> Dict[str, float]:
    case_dir.mkdir(parents=True, exist_ok=True)
    name = str(row.get("name", "specimen"))
    result = {
        "name": name,
        "status": "INIT",
        "message": "",
        "steps_finished": 0,
        "max_abs_disp_mm": np.nan,
        "max_abs_force_N": np.nan,
        "energy_Nmm": np.nan,
        "hw_mm": safe_float(row, "hw"),
        "hl_mm": safe_float(row, "hl", 0.0),
        "lw_mm": safe_float(row, "lw"),
        "tw_mm": safe_float(row, "tw"),
        "fc_MPa": safe_float(row, "f'c"),
        "N_N": safe_float(row, "N"),
    }

    try:
        setup_model(row, case_dir, plot_section=plot_section)
        if not run_gravity():
            result["status"] = "GRAVITY_FAIL"
            result["message"] = "重力分析失败"
            return result

        setup_cyclic_recorders(case_dir)
        height_for_drift = safe_float(row, "hw") + safe_float(row, "hl", 0.0)
        if height_for_drift <= 0:
            height_for_drift = safe_float(row, "hw")
        disp_history = build_displacement_history(height_for_drift, cycles_per_level=cycles_per_level)
        pd.DataFrame({"target_disp_mm": disp_history}).to_csv(case_dir / "target_history.csv", index=False, encoding="utf-8-sig")

        ok, finished_steps, last_disp = run_cyclic_analysis(disp_history)
        result["steps_finished"] = finished_steps
        result.update(postprocess_case(case_dir, name))

        if ok:
            result["status"] = "OK"
            result["message"] = "分析完成"
        else:
            result["status"] = "CYCLIC_FAIL"
            result["message"] = f"滞回分析中止，最后达到位移 {last_disp:.3f} mm"

    except Exception as e:
        result["status"] = "ERROR"
        result["message"] = str(e)

    finally:
        try:
            ops.wipe()
        except Exception:
            pass
        plt.close("all")

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="批量墙纤维模型滞回分析")
    parser.add_argument("--csv", type=str, default="wall_generated_for_modeling_1200.csv", help="参数表 CSV 路径")
    parser.add_argument("--outdir", type=str, default="batch_hysteresis_results", help="输出目录")
    parser.add_argument("--start", type=int, default=1, help="起始行号（1-based，含）")
    parser.add_argument("--end", type=int, default=0, help="结束行号（1-based，含）；0 表示到末尾")
    parser.add_argument("--cycles", type=int, default=1, help="每个位移角级别的循环次数，默认 1")
    parser.add_argument("--plot-section", action="store_true", help="是否保存每个试件纤维截面图")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    rows = read_rows(args.csv)
    start_idx = max(args.start - 1, 0)
    end_idx = len(rows) if args.end in [0, None] else min(args.end, len(rows))
    selected = rows[start_idx:end_idx]

    summary = []
    for k, row in enumerate(selected, start=start_idx + 1):
        name = str(row.get("name", f"row_{k:04d}"))
        case_dir = outdir / f"{k:04d}_{name}"
        print(f"[{k}/{end_idx}] Running {name} ...")
        res = analyze_one_row(row, case_dir, cycles_per_level=args.cycles, plot_section=args.plot_section)
        res["row_id_1based"] = k
        summary.append(res)
        pd.DataFrame(summary).to_csv(outdir / "summary_running.csv", index=False, encoding="utf-8-sig")

    df_sum = pd.DataFrame(summary)
    df_sum.to_csv(outdir / "summary.csv", index=False, encoding="utf-8-sig")

    ok_count = int((df_sum["status"] == "OK").sum()) if len(df_sum) else 0
    fail_count = len(df_sum) - ok_count
    with open(outdir / "README.txt", "w", encoding="utf-8") as f:
        f.write("批量墙纤维模型滞回分析结果\n")
        f.write(f"参数表: {args.csv}\n")
        f.write(f"行范围: {args.start} ~ {args.end if args.end else len(rows)}\n")
        f.write(f"每级位移角循环数: {args.cycles}\n")
        f.write(f"试件总数: {len(df_sum)}\n")
        f.write(f"成功: {ok_count}\n")
        f.write(f"失败: {fail_count}\n")
        f.write("\n位移角级别:\n")
        for d in DRIFT_LEVELS:
            f.write(f"{d:.4f}\n")

    print("Done.")
    print(f"Summary saved to: {outdir / 'summary.csv'}")


if __name__ == "__main__":
    main()
