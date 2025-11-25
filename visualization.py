import argparse
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm

def load_xyz(path: str):
    """
    .xyz 파일 로드
    - 최소 3컬럼: x, y, z
    - 4번째 컬럼이 있으면 scalar로 취급 (경로 정보 등)
    """
    data = np.loadtxt(path)
    if data.ndim == 1:
        data = data[None, :]  # 한 줄짜리도 대응

    if data.shape[1] < 3:
        raise ValueError(f"{path} : 최소 3개의 컬럼(x,y,z)이 필요합니다. shape={data.shape}")

    points = data[:, :3]
    scalar = data[:, 3] if data.shape[1] >= 4 else None
    return points, scalar


def scalar_to_color(scalar: np.ndarray, cmap_name: str = "viridis"):
    """
    1D scalar array -> RGB color (0~1)로 매핑
    """
    scalar = scalar.astype(np.float64)
    s_min, s_max = scalar.min(), scalar.max()
    if s_max > s_min:
        s_norm = (scalar - s_min) / (s_max - s_min)
    else:
        # 전부 같은 값이면 전부 0으로
        s_norm = np.zeros_like(scalar, dtype=np.float64)

    cmap = cm.get_cmap(cmap_name)
    colors = cmap(s_norm)[:, :3]  # RGBA 중 RGB만 사용
    return colors


def visualize_single(path: str, cmap_name: str = "viridis"):
    """
    단일 .xyz 파일 시각화
    - 4번째 컬럼이 있으면 scalar 색상
    - 없으면 단색
    """
    print(f"[INFO] Loading {path}")
    points, scalar = load_xyz(path)

    # Matplotlib을 이용한 3D 시각화
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    if scalar is not None:
        colors = scalar_to_color(scalar, cmap_name)
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=colors, s=1)
        print(f"[INFO] Scalar field detected (4th column). "
              f"Min={scalar.min():.3f}, Max={scalar.max():.3f}")
    else:
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], c='b', s=1)
        print("[INFO] No scalar field detected. Using uniform color.")

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()


def visualize_pair(path_a: str, path_b: str, offset: float = 0.0):
    """
    두 개의 .xyz를 한 화면에서 비교
    - A: 파란색
    - B: 빨간색 (원하는 경우 x축 방향으로 offset 만큼 평행 이동해서 겹치지 않게 볼 수 있음)
    """
    print(f"[INFO] Loading A: {path_a}")
    pts_a, _ = load_xyz(path_a)

    print(f"[INFO] Loading B: {path_b}")
    pts_b, _ = load_xyz(path_b)

    if offset != 0.0:
        pts_b = pts_b.copy()
        pts_b[:, 0] += offset  # x축으로 평행 이동

    # Matplotlib을 이용한 3D 시각화
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(pts_a[:, 0], pts_a[:, 1], pts_a[:, 2], c='b', s=1)  # A: 파란색
    ax.scatter(pts_b[:, 0], pts_b[:, 1], pts_b[:, 2], c='r', s=1)  # B: 빨간색

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--file", "-f",
        type=str,
        required=True,
        help="시각화할 .xyz 파일 경로 (예: data/test_data/results/.../xxx_output_end.xyz)",
    )
    parser.add_argument(
        "--file2", "-f2",
        type=str,
        default=None,
        help="두 번째 .xyz 파일 경로 (예: noisy input). 주어지면 두 파일 비교 모드로 실행",
    )
    parser.add_argument(
        "--offset",
        type=float,
        default=0.0,
        help="두 파일 비교 시, 두 번째 포인트 클라우드를 x축으로 평행 이동할 거리 (기본 0.0)",
    )
    parser.add_argument(
        "--cmap",
        type=str,
        default="viridis",
        help="단일 파일 시각화 시 사용할 matplotlib 컬러맵 이름 (기본: viridis)",
    )
    args = parser.parse_args()

    if args.file2 is None:
        visualize_single(args.file, cmap_name=args.cmap)
    else:
        visualize_pair(args.file, args.file2, offset=args.offset)


if __name__ == "__main__":
    main()
