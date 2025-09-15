import argparse
import math
import struct
import subprocess
import shutil
from pathlib import Path

import numpy as np
import yaml


def load_prior_tcw(prior_yaml_path: Path) -> dict[str, np.ndarray]:
    with open(prior_yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    poses = data.get("camera_poses", {})
    tcw = {}
    for k, v in poses.items():
        T = np.array(v["transform_matrix"], dtype=float)
        # key 到图像名的映射：images/0 -> 0.png
        if k.startswith("images/"):
            stem = k.split("/")[-1]
            name = f"{stem}.png"
        else:
            name = k
        tcw[name] = T
    return tcw


# 默认路径（可直接改这里免命令行）
DEFAULT_PRIOR = Path("/root/autodl-tmp/mpsfm/local/example/camera_poses.yaml")
DEFAULT_REC = Path("/root/autodl-tmp/mpsfm/local/example/sfm_outputs/rec")


def _q_to_R_wxyz(qw, qx, qy, qz) -> np.ndarray:
    n = math.sqrt(qw*qw + qx*qx + qy*qy + qz*qz)
    qw, qx, qy, qz = qw/n, qx/n, qy/n, qz/n
    # 单位四元数转R
    return np.array([
        [1-2*qy*qy-2*qz*qz, 2*qx*qy-2*qz*qw, 2*qx*qz+2*qy*qw],
        [2*qx*qy+2*qz*qw, 1-2*qx*qx-2*qz*qz, 2*qy*qz-2*qx*qw],
        [2*qx*qz-2*qy*qw, 2*qy*qz+2*qx*qw, 1-2*qx*qx-2*qy*qy]
    ])


def _read_images_bin(images_bin: Path) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    images = {}
    with open(images_bin, 'rb') as f:
        num_images = struct.unpack('<Q', f.read(8))[0]
        for _ in range(num_images):
            _img_id = struct.unpack('<Q', f.read(8))[0]
            qw, qx, qy, qz = struct.unpack('<dddd', f.read(32))
            tx, ty, tz = struct.unpack('<ddd', f.read(24))
            _cam_id = struct.unpack('<Q', f.read(8))[0]
            # read null-terminated name
            name_bytes = []
            while True:
                c = f.read(1)
                if c == b'\x00':
                    break
                name_bytes.append(c)
            name = b''.join(name_bytes).decode('utf-8', errors='ignore')
            num_points2D = struct.unpack('<Q', f.read(8))[0]
            f.read(num_points2D * (8+8+8))  # skip x(double), y(double), point3D_id(uint64)
            R = _q_to_R_wxyz(qw, qx, qy, qz)
            t = np.array([tx, ty, tz], dtype=float)
            images[name] = (R, t)
    return images


def _have_colmap() -> bool:
    return shutil.which("colmap") is not None


def _convert_bin_to_txt(rec_dir: Path, out_dir: Path | None = None) -> Path | None:
    rec_dir = Path(rec_dir)
    if out_dir is None:
        out_dir = rec_dir.parent / (rec_dir.name + "_txt")
    out_dir.mkdir(parents=True, exist_ok=True)
    if not _have_colmap():
        return None
    try:
        subprocess.run([
            "colmap", "model_converter",
            "--input_path", str(rec_dir),
            "--output_path", str(out_dir),
            "--output_type", "TXT",
        ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        return None
    images_txt = out_dir / "images.txt"
    return out_dir if images_txt.exists() else None


def load_opt_tcw(reconstruction_dir: Path) -> dict[str, np.ndarray]:
    reconstruction_dir = Path(reconstruction_dir)
    images_txt = reconstruction_dir / 'images.txt'
    if images_txt.exists():
        tcw = {}
        with open(images_txt, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                parts = line.split()
                if len(parts) < 10:
                    continue
                # id qw qx qy qz tx ty tz cam_id name
                qw, qx, qy, qz = map(float, parts[1:5])
                tx, ty, tz = map(float, parts[5:8])
                name = parts[9]
                R = _q_to_R_wxyz(qw, qx, qy, qz)
                T = np.eye(4)
                T[:3, :3] = R
                T[:3, 3] = [tx, ty, tz]
                tcw[name] = T
        return tcw
    # fallback to BIN
    images_bin = reconstruction_dir / 'images.bin'
    tcw = {}
    for name, (R, t) in _read_images_bin(images_bin).items():
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = t
        tcw[name] = T
    return tcw


def rotation_angle_deg(R: np.ndarray) -> float:
    cos = max(-1.0, min(1.0, (np.trace(R) - 1.0) * 0.5))
    return math.degrees(math.acos(cos))


def cam_center(Tcw: np.ndarray) -> np.ndarray:
    R = Tcw[:3, :3]
    t = Tcw[:3, 3]
    return -R.T @ t


def _umeyama(A: np.ndarray, B: np.ndarray, with_scale: bool) -> tuple[float, np.ndarray, np.ndarray]:
    mu_A = A.mean(0)
    mu_B = B.mean(0)
    X = A - mu_A
    Y = B - mu_B
    C = (Y.T @ X) / A.shape[0]
    U, S, Vt = np.linalg.svd(C)
    d = np.ones(3)
    if np.linalg.det(U @ Vt) < 0:
        d[-1] = -1
    D = np.diag(d)
    R = U @ D @ Vt
    if with_scale:
        var = np.mean(np.sum(X * X, axis=1))
        if var <= 1e-12:
            s = 1.0
        else:
            s = float(np.sum(S * d) / var)
    else:
        s = 1.0
    t = mu_B - s * (R @ mu_A)
    return float(s), R, t


def _fmt_mat4(T: np.ndarray) -> str:
    return "\n".join(" ".join(f"{v:.8f}" for v in row) for row in T)


def compare(prior_tcw: dict, opt_tcw: dict, limit: int | None = None, align: str = "sim3", out_path: Path | None = None):
    rows = []
    names = []
    T_inis = []
    T_opts = []
    for name_opt, T_opt in opt_tcw.items():
        name = name_opt
        # 允许 0 -> 0.png 对齐
        if name not in prior_tcw and name.endswith(".png"):
            alt = name.rsplit(".", 1)[0]
            if alt in prior_tcw:
                name = alt
        if name not in prior_tcw:
            continue
        names.append(name_opt)
        T_inis.append(prior_tcw[name])
        T_opts.append(T_opt)

    if not names:
        print("No overlapping images between prior and optimized results.")
        return

    C_ini = np.vstack([cam_center(T) for T in T_inis])
    C_opt = np.vstack([cam_center(T) for T in T_opts])

    s, Rw, tw = 1.0, np.eye(3), np.zeros(3)
    if align in {"se3", "sim3"}:
        s, Rw, tw = _umeyama(C_opt, C_ini, with_scale=(align == "sim3"))
    # 对齐优化结果的中心与旋转
    C_opt_aligned = (s * (Rw @ C_opt.T)).T + tw
    R_opt_aligned = [T[:3, :3] @ Rw.T for T in T_opts]

    details = []
    for name_opt, T_ini, T_opt_raw, R_opt_a, c_opt_a in zip(names, T_inis, T_opts, R_opt_aligned, C_opt_aligned):
        R_ini = T_ini[:3, :3]
        dR = R_opt_a @ R_ini.T
        rot_err = rotation_angle_deg(dR)
        c_ini = cam_center(T_ini)
        cen_err = float(np.linalg.norm(c_opt_a - c_ini))
        rows.append((name_opt, rot_err, cen_err))
        # 计算对齐后的 t
        t_opt_a = - R_opt_a @ c_opt_a
        T_opt_aligned = np.eye(4)
        T_opt_aligned[:3, :3] = R_opt_a
        T_opt_aligned[:3, 3] = t_opt_a
        details.append({
            "name": name_opt,
            "rot_err_deg": rot_err,
            "center_err": cen_err,
            "prior_Tcw": T_ini,
            "opt_Tcw_raw": T_opt_raw,
            "opt_Tcw_aligned": T_opt_aligned,
        })

    rows.sort(key=lambda x: int(Path(x[0]).stem))
    if limit is not None:
        rows = rows[:limit]

    rot_list = [r for _, r, _ in rows]
    cen_list = [c for _, _, c in rows]
    print("name, rot_err_deg, center_err")
    for name, r, c in rows:
        print(f"{name}, {r:.3f}, {c:.4f}")
    print("-- stats --")
    print(f"rot_err_deg: mean={np.mean(rot_list):.3f}, median={np.median(rot_list):.3f}, max={np.max(rot_list):.3f}")
    print(f"center_err:  mean={np.mean(cen_list):.4f}, median={np.median(cen_list):.4f}, max={np.max(cen_list):.4f}")

    # 写出详细对比
    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(f"align={align}  s={s:.9f}\n")
            f.write("Rw=\n" + _fmt_mat4(np.vstack([np.hstack([Rw, np.zeros((3,1))]), [0,0,0,1]])) + "\n")
            f.write(f"tw= {tw.tolist()}\n\n")
            f.write("name, rot_err_deg, center_err\n")
            for name, r, c in rows:
                f.write(f"{name}, {r:.6f}, {c:.9f}\n")
            f.write("\n-- per-pose matrices --\n")
            for d in details:
                f.write(f"\n[{d['name']}]\n")
                f.write("prior_Tcw:\n" + _fmt_mat4(d["prior_Tcw"]) + "\n")
                f.write("opt_Tcw_raw:\n" + _fmt_mat4(d["opt_Tcw_raw"]) + "\n")
                f.write("opt_Tcw_aligned:\n" + _fmt_mat4(d["opt_Tcw_aligned"]) + "\n")
        print(f"saved report to: {out_path}")


def main():
    ap = argparse.ArgumentParser(description="Compare optimized poses vs prior (both as Tcw)")
    ap.add_argument("--prior", type=Path, default=None, help="camera_poses.yaml (prior, Tcw); default uses code constant")
    ap.add_argument("--rec", type=Path, default=None, help="COLMAP model dir (bin/txt); default uses code constant")
    ap.add_argument("--limit", type=int, default=None, help="Only print first N images")
    ap.add_argument("--align", type=str, default="sim3", choices=["none", "se3", "sim3"], help="Alignment before comparison")
    ap.add_argument("--out", type=Path, default=None, help="Path to save detailed comparison txt")
    args = ap.parse_args()

    prior_path = args.prior or DEFAULT_PRIOR
    rec_path = args.rec or DEFAULT_REC

    # 若是bin目录，尝试转换成txt
    txt_dir = rec_path if (rec_path / "images.txt").exists() else _convert_bin_to_txt(rec_path)
    rec_for_load = txt_dir if txt_dir is not None else rec_path

    prior = load_prior_tcw(prior_path)
    opt = load_opt_tcw(rec_for_load)
    default_out = (rec_for_load / "pose_compare.txt") if isinstance(rec_for_load, Path) else None
    out = args.out or default_out
    compare(prior, opt, args.limit, args.align, out)


if __name__ == "__main__":
    main()


