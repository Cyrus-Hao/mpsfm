import yaml
import os
import sys
import csv
import numpy as np
from pathlib import Path

# 可配置的CSV文件路径
CAMERA_MATRIX_CSV_PATH = "/root/autodl-tmp/mpsfm/local/example/36ba581007/camera_matrix.csv"
ODOMETRY_CSV_PATH = "/root/autodl-tmp/mpsfm/local/example/36ba581007/odometry.csv"

def rename_images_sequentially(images_dir: str) -> int:
    """将目录内图片重命名为 0..N-1，保留扩展名；若无图则返回 0。"""
    images_path = Path(images_dir)
    if not images_path.exists():
        return 0

    valid_exts = {".png", ".jpg", ".jpeg", ".bmp"}
    image_files = [p for p in images_path.iterdir() if p.is_file() and p.suffix.lower() in valid_exts]

    if not image_files:
        return 0

    image_files.sort(key=lambda p: p.name)

    # 第一步：重命名为临时文件，避免命名冲突
    temp_files = []
    for idx, p in enumerate(image_files):
        tmp_path = p.with_name(f"__renametmp_{idx}{p.suffix.lower()}")
        p.rename(tmp_path)
        temp_files.append(tmp_path)

    # 第二步：重命名为目标序号
    for idx, tmp in enumerate(temp_files):
        final_path = tmp.with_name(f"{idx}{tmp.suffix.lower()}")
        tmp.rename(final_path)

    print(f"已重命名 {len(temp_files)} 张图片为 0..{len(temp_files)-1}")
    return len(temp_files)

def read_camera_matrix(csv_path):
    """读取 3x3 相机内参矩阵"""
    try:
        with open(csv_path, 'r') as f:
            reader = csv.reader(f)
            matrix = []
            for row in reader:
                matrix.append([float(x.strip()) for x in row])
        return np.array(matrix)
    except FileNotFoundError:
        print(f"警告: 找不到相机内参文件 {csv_path}")
        return None
    except Exception as e:
        print(f"读取相机内参文件时出错: {e}")
        return None

def read_odometry_data(csv_path):
    """读取里程计 CSV（x,y,z,qx,qy,qz,qw）。"""
    try:
        odometry_data = []
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                odometry_data.append({
                    'timestamp': float(row['timestamp']),
                    'frame': row[' frame'].strip(),  # 处理列名中的空格
                    'x': float(row[' x']),  # 处理列名中的空格
                    'y': float(row[' y']),  # 处理列名中的空格
                    'z': float(row[' z']),  # 处理列名中的空格
                    'qx': float(row[' qx']),  # 处理列名中的空格
                    'qy': float(row[' qy']),  # 处理列名中的空格
                    'qz': float(row[' qz']),  # 处理列名中的空格
                    'qw': float(row[' qw'])  # 处理列名中的空格
                })
        return odometry_data
    except FileNotFoundError:
        print(f"警告: 找不到里程计文件 {csv_path}")
        return []
    except Exception as e:
        print(f"读取里程计文件时出错: {e}")
        return []

def quaternion_to_rotation_matrix(qx, qy, qz, qw):
    """四元数(qx,qy,qz,qw)→R(3x3)。"""
    n = np.sqrt(qx*qx + qy*qy + qz*qz + qw*qw)
    qx, qy, qz, qw = qx/n, qy/n, qz/n, qw/n
    return np.array([
        [1-2*qy*qy-2*qz*qz, 2*qx*qy-2*qz*qw, 2*qx*qz+2*qy*qw],
        [2*qx*qy+2*qz*qw, 1-2*qx*qx-2*qz*qz, 2*qy*qz-2*qx*qw],
        [2*qx*qz-2*qy*qw, 2*qy*qz+2*qx*qw, 1-2*qx*qx-2*qy*qy]
    ])

def create_transform_matrix(x, y, z, qx, qy, qz, qw):
    """创建 4x4 变换矩阵。"""
    R = quaternion_to_rotation_matrix(qx, qy, qz, qw)
    t = np.array([x, y, z])
    
    # 创建4x4变换矩阵
    transform = np.eye(4)
    transform[:3, :3] = R
    transform[:3, 3] = t
    
    return transform.tolist()

def process_csv_data(sample_interval=60):
    """处理 CSV（固定 Twc 输入），输出 camera_poses 与 intrinsics。"""
    camera_matrix = read_camera_matrix(CAMERA_MATRIX_CSV_PATH)
    
    odometry_data = read_odometry_data(ODOMETRY_CSV_PATH)
    
    if camera_matrix is None or not odometry_data:
        print("错误: 无法读取必要的CSV文件")
        return None, None
    
    # 从相机矩阵提取内参（转为原生float）
    fx = float(camera_matrix[0, 0])  # 焦距x
    fy = float(camera_matrix[1, 1])  # 焦距y
    cx = float(camera_matrix[0, 2])  # 主点x
    cy = float(camera_matrix[1, 2])  # 主点y
    
    # 初始化输出数据
    camera_poses = {"camera_poses": {}}
    intrinsics = {}
    
    # 按间隔抽取里程计数据
    sampled_indices = list(range(0, len(odometry_data), sample_interval))
    
    for frame_idx, data_idx in enumerate(sampled_indices):
        if data_idx >= len(odometry_data):
            break
            
        odom = odometry_data[data_idx]
        
        # 文件路径 - 使用简单的数字格式，从0开始
        file_path = f"images/{frame_idx}"  # 从0开始：0, 1, 2, ...
        
        # Twc -> 取逆得到 Tcw
        R_wc = quaternion_to_rotation_matrix(odom['qx'], odom['qy'], odom['qz'], odom['qw'])
        t_wc = np.array([odom['x'], odom['y'], odom['z']], dtype=float)
        R_cw = R_wc.T
        t_cw = -R_cw @ t_wc
        T = np.eye(4)
        T[:3, :3] = R_cw
        T[:3, 3] = t_cw
        transform_matrix = T.tolist()
        
        camera_poses["camera_poses"][file_path] = {
            "transform_matrix": transform_matrix
        }
        
        # 相机内参（确保为原生float）
        intrinsics[frame_idx + 1] = {
            "params": [float(fx), float(fy), float(cx), float(cy)],
            "images": [f"{frame_idx}.png"]  # 从0开始：0.png, 1.png, ...
        }
    
    return camera_poses, intrinsics

def save_files(camera_poses, intrinsics):
    """保存 YAML。"""
    output_dir = "/root/autodl-tmp/mpsfm/local/example"
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存camera_poses.yaml
    with open(f"{output_dir}/camera_poses.yaml", 'w', encoding='utf-8') as f:
        yaml.safe_dump(camera_poses, f, default_flow_style=False, allow_unicode=True)
    
    # 保存intrinsics.yaml
    with open(f"{output_dir}/intrinsics.yaml", 'w', encoding='utf-8') as f:
        yaml.safe_dump(intrinsics, f, default_flow_style=False, allow_unicode=True)
    
    print(f"已保存到: {output_dir}，相机数: {len(intrinsics)}")

if __name__ == "__main__":
    sample_interval = 60
    if len(sys.argv) > 1:
        try:
            sample_interval = int(sys.argv[1])
            print(f"使用自定义抽取间隔: {sample_interval}帧")
        except ValueError:
            print(f"警告: 无效的抽取间隔 '{sys.argv[1]}'，使用默认值60")
    else:
        print("使用默认抽取间隔: 60帧")

    try:
        rename_images_sequentially("/root/autodl-tmp/mpsfm/local/example/images")
        print("开始处理CSV数据...")
        camera_poses, intrinsics = process_csv_data(sample_interval)
        if camera_poses and intrinsics:
            save_files(camera_poses, intrinsics)
            print("处理完成！")
        else:
            print("错误: 无法处理数据")
    except Exception as e:
        print(f"错误: {e}")
