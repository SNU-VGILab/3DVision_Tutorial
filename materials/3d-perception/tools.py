from pathlib import Path
import os

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import torch


SCANNET_SEMANTIC_PALETTE = np.array(
    [
        [174, 199, 232],
        [152, 223, 138],
        [31, 119, 180],
        [255, 187, 120],
        [188, 189, 34],
        [140, 86, 75],
        [255, 152, 150],
        [214, 39, 40],
        [197, 176, 213],
        [148, 103, 189],
        [196, 156, 148],
        [23, 190, 207],
        [247, 182, 210],
        [219, 219, 141],
        [255, 127, 14],
        [158, 218, 229],
        [44, 160, 44],
        [112, 128, 144],
        [227, 119, 194],
        [82, 84, 163],
    ],
    dtype=np.uint8,
)

SCANNET_SEMANTIC_CLASS_NAMES = [
    "wall",
    "floor",
    "cabinet",
    "bed",
    "chair",
    "sofa",
    "table",
    "door",
    "window",
    "bookshelf",
    "picture",
    "counter",
    "desk",
    "curtain",
    "refrigerator",
    "shower curtain",
    "toilet",
    "sink",
    "bathtub",
    "other furniture",
]


def visualize_point_cloud(
    points,
    colors=None,
    title="Point Cloud",
    point_size=2,
    colorscale="Viridis",
    show_axis=True,
):
    if isinstance(points, torch.Tensor):
        points = points.detach().cpu().numpy()

    if colors is None:
        colors = points[:, 2]
    elif isinstance(colors, torch.Tensor):
        colors = colors.detach().cpu().numpy()
    else:
        colors = np.asarray(colors)

    marker_color = colors
    marker = dict(size=point_size, opacity=0.9)

    if isinstance(colors, np.ndarray) and colors.ndim == 2 and colors.shape[1] >= 3:
        rgb = colors[:, :3]
        if rgb.max() <= 1.0:
            rgb = rgb * 255.0
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)
        marker_color = [f"rgb({r},{g},{b})" for r, g, b in rgb]
        marker.update(color=marker_color, showscale=False)
    else:
        marker.update(
            color=marker_color,
            colorscale=colorscale,
            showscale=True,
            colorbar=dict(title="Value", thickness=20, len=0.7),
        )

    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=points[:, 0],
                y=points[:, 1],
                z=points[:, 2],
                mode="markers",
                marker=marker,
                text=[
                    f"Point {i}<br>x: {x:.3f}<br>y: {y:.3f}<br>z: {z:.3f}"
                    for i, (x, y, z) in enumerate(points[: min(1000, len(points))])
                ],
                hovertemplate="%{text}<extra></extra>",
            )
        ]
    )

    fig.update_layout(
        title=dict(text=title, x=0.5, xanchor="center"),
        scene=dict(
            xaxis=dict(title="X" if show_axis else "", showgrid=True, gridwidth=1),
            yaxis=dict(title="Y" if show_axis else "", showgrid=True, gridwidth=1),
            zaxis=dict(title="Z" if show_axis else "", showgrid=True, gridwidth=1),
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.5)),
            aspectmode="auto",
        ),
        width=900,
        height=700,
        margin=dict(r=20, b=10, l=10, t=40),
        showlegend=False,
    )

    fig.show()
    return fig


def visualize_multiple_point_clouds(point_clouds_dict, title="Multiple Point Clouds"):
    fig = go.Figure()
    colors = px.colors.qualitative.Set1

    for idx, (name, points) in enumerate(point_clouds_dict.items()):
        if isinstance(points, torch.Tensor):
            points = points.detach().cpu().numpy()

        color = colors[idx % len(colors)]
        fig.add_trace(
            go.Scatter3d(
                x=points[:, 0],
                y=points[:, 1],
                z=points[:, 2],
                mode="markers",
                name=name,
                marker=dict(size=3, color=color, opacity=0.8),
                text=[f"{name} - Point {i}" for i in range(min(100, len(points)))],
                hovertemplate="%{text}<br>x: %{x:.3f}<br>y: %{y:.3f}<br>z: %{z:.3f}<extra></extra>",
            )
        )

    fig.update_layout(
        title=dict(text=title, x=0.5, xanchor="center"),
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z",
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.5)),
        ),
        width=1000,
        height=700,
        showlegend=True,
        legend=dict(x=0.02, y=0.98, bgcolor="rgba(255,255,255,0.8)"),
    )

    fig.show()
    return fig


def visualize_voxels(voxel_coords, voxel_size=0.1, colors=None, title="Sparse Voxels"):
    if isinstance(voxel_coords, torch.Tensor):
        voxel_coords = voxel_coords.detach().cpu().numpy()

    if colors is None:
        colors = voxel_coords[:, 2]
    elif isinstance(colors, torch.Tensor):
        colors = colors.detach().cpu().numpy()

    if len(colors.shape) > 1:
        colors = colors[:, 0]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter3d(
            x=voxel_coords[:, 0],
            y=voxel_coords[:, 1],
            z=voxel_coords[:, 2],
            mode="markers",
            marker=dict(
                size=5,
                color=colors,
                colorscale="Viridis",
                showscale=True,
                symbol="square",
                colorbar=dict(title="Value", thickness=20, len=0.7),
            ),
            text=[
                f"Voxel {i}<br>x: {x:.2f}<br>y: {y:.2f}<br>z: {z:.2f}"
                for i, (x, y, z) in enumerate(voxel_coords[: min(500, len(voxel_coords))])
            ],
            hovertemplate="%{text}<extra></extra>",
            name="Voxels",
        )
    )

    fig.update_layout(
        title=dict(text=title, x=0.5, xanchor="center"),
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z",
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.5)),
            aspectmode="data",
        ),
        width=900,
        height=700,
        showlegend=False,
    )

    fig.show()
    return fig


def generate_sample_point_cloud(n_points=1000, shape="sphere", noise_level=0.0):
    if shape == "sphere":
        theta = np.random.uniform(0, 2 * np.pi, n_points)
        phi = np.random.uniform(0, np.pi, n_points)
        radius = np.random.uniform(0.8, 1.2, n_points)
        x = radius * np.sin(phi) * np.cos(theta)
        y = radius * np.sin(phi) * np.sin(theta)
        z = radius * np.cos(phi)
    elif shape == "cube":
        x = np.random.uniform(-1, 1, n_points)
        y = np.random.uniform(-1, 1, n_points)
        z = np.random.uniform(-1, 1, n_points)
    elif shape == "torus":
        theta = np.random.uniform(0, 2 * np.pi, n_points)
        phi = np.random.uniform(0, 2 * np.pi, n_points)
        major_radius, minor_radius = 1.0, 0.3
        x = (major_radius + minor_radius * np.cos(phi)) * np.cos(theta)
        y = (major_radius + minor_radius * np.cos(phi)) * np.sin(theta)
        z = minor_radius * np.sin(phi)
    elif shape == "cylinder":
        theta = np.random.uniform(0, 2 * np.pi, n_points)
        z = np.random.uniform(-1, 1, n_points)
        radius = np.random.uniform(0.8, 1.0, n_points)
        x = radius * np.cos(theta)
        y = radius * np.sin(theta)
    else:
        raise ValueError(f"Unknown shape: {shape}")

    points = np.stack([x, y, z], axis=1).astype(np.float32)
    if noise_level > 0:
        points += (np.random.randn(*points.shape) * noise_level).astype(np.float32)
    features = np.random.rand(n_points, 3).astype(np.float32)
    return points, features


def get_scannet_root(root=None):
    if root is not None:
        return Path(root)

    candidates = []
    if os.environ.get("SCANNET_DATA_DIR"):
        candidates.append(Path(os.environ["SCANNET_DATA_DIR"]))
    candidates.extend([Path("/data/scannet_3d"), Path("../../data/scannet_3d"), Path("data/scannet_3d")])

    for candidate in candidates:
        if (candidate / "scannetv2_train.txt").exists():
            return candidate
    raise FileNotFoundError("ScanNet data root not found. Run the ScanNet loading cell first or set SCANNET_DATA_DIR.")


def find_scannet_scene(root, scene_name, split=None):
    candidate_splits = [split] if split is not None else ["train", "val", "test"]
    for candidate_split in candidate_splits:
        scene_path = root / candidate_split / f"{scene_name}_vh_clean_2.pth"
        if scene_path.exists():
            return candidate_split, scene_path
    raise FileNotFoundError(
        f"Could not find {scene_name}_vh_clean_2.pth under {root}. Check the scene name and split."
    )


def load_scannet_scene_arrays(scene_name, split=None, root=None):
    scannet_root = get_scannet_root(root)
    split, scene_path = find_scannet_scene(scannet_root, scene_name, split)
    coords, colors, labels = torch.load(scene_path, weights_only=False)
    return split, np.asarray(coords, dtype=np.float32), np.asarray(colors, dtype=np.float32), np.asarray(labels, dtype=np.int32)


def scannet_colors_to_uint8(colors):
    colors = np.asarray(colors, dtype=np.float32)
    if colors.size == 0:
        return colors.astype(np.uint8)
    if colors.min() < 0.0:
        colors = (colors + 1.0) * 127.5
    elif colors.max() <= 1.0:
        colors = colors * 255.0
    return np.clip(np.rint(colors), 0, 255).astype(np.uint8)


def semantic_labels_to_uint8_colors(labels):
    labels = np.asarray(labels, dtype=np.int32)
    colors = SCANNET_SEMANTIC_PALETTE[np.mod(np.maximum(labels, 0), len(SCANNET_SEMANTIC_PALETTE))]
    colors = colors.astype(np.uint8, copy=True)
    colors[labels == 255] = np.array([80, 80, 80], dtype=np.uint8)
    return colors


def sample_scene_arrays(coords, colors, labels, max_points=None):
    if max_points is None or coords.shape[0] <= max_points:
        return coords, colors, labels
    sample_idx = np.linspace(0, coords.shape[0] - 1, max_points).astype(np.int64)
    return coords[sample_idx], colors[sample_idx], labels[sample_idx]


def voxel_size_to_tag(voxel_size):
    return str(voxel_size).replace(".", "p")


def voxelize_scene_arrays(coords, colors, labels, voxel_size):
    if voxel_size is None:
        return coords, colors, labels

    discrete_coords = np.floor(coords / voxel_size).astype(np.int64)
    unique_coords, inverse, counts = np.unique(
        discrete_coords,
        axis=0,
        return_inverse=True,
        return_counts=True,
    )

    voxel_centers = (unique_coords.astype(np.float32) + 0.5) * float(voxel_size)

    color_sums = np.zeros((unique_coords.shape[0], colors.shape[1]), dtype=np.float64)
    np.add.at(color_sums, inverse, colors.astype(np.float64))
    voxel_colors = (color_sums / counts[:, None]).astype(np.float32)

    order = np.argsort(inverse)
    sorted_inverse = inverse[order]
    sorted_labels = labels[order]
    boundaries = np.flatnonzero(np.diff(sorted_inverse)) + 1
    boundaries = np.concatenate(([0], boundaries, [sorted_inverse.shape[0]]))

    voxel_labels = np.empty(unique_coords.shape[0], dtype=np.int32)
    for voxel_index, (start, end) in enumerate(zip(boundaries[:-1], boundaries[1:])):
        values, value_counts = np.unique(sorted_labels[start:end], return_counts=True)
        voxel_labels[voxel_index] = values[np.argmax(value_counts)]

    return voxel_centers, voxel_colors, voxel_labels


def write_point_cloud_ply(output_path, coords, colors, labels):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {coords.shape[0]}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("property int label\n")
        f.write("end_header\n")
        for xyz, rgb, label in zip(coords, colors, labels):
            f.write(
                f"{xyz[0]:.6f} {xyz[1]:.6f} {xyz[2]:.6f} "
                f"{int(rgb[0])} {int(rgb[1])} {int(rgb[2])} {int(label)}\n"
            )
    return output_path


def save_scannet_scene_as_ply(
    scene_name,
    split=None,
    max_points=None,
    output_dir=Path("ply_exports"),
    color_mode="rgb",
    suffix=None,
    voxel_size=None,
    root=None,
):
    split, coords, colors, labels = load_scannet_scene_arrays(scene_name, split=split, root=root)
    coords, colors, labels = sample_scene_arrays(coords, colors, labels, max_points=max_points)
    coords, colors, labels = voxelize_scene_arrays(coords, colors, labels, voxel_size=voxel_size)

    voxel_suffix = "" if voxel_size is None else f"_voxel_{voxel_size_to_tag(voxel_size)}"
    if color_mode == "rgb":
        ply_colors = scannet_colors_to_uint8(colors)
        suffix = voxel_suffix if suffix is None else suffix
    elif color_mode == "semantic":
        ply_colors = semantic_labels_to_uint8_colors(labels)
        suffix = f"{voxel_suffix}_semantic" if suffix is None else suffix
    else:
        raise ValueError('color_mode must be "rgb" or "semantic".')

    output_path = Path(output_dir) / f"{scene_name}{suffix}.ply"
    write_point_cloud_ply(output_path, coords, ply_colors, labels)
    voxel_text = "raw" if voxel_size is None else f"voxel_size={voxel_size}"
    print(f"Saved {coords.shape[0]:,} vertices from {split}/{scene_name} to {output_path} ({color_mode}, {voxel_text})")
    return output_path


def display_scannet_semantic_legend():
    from IPython.display import HTML, display

    legend_entries = []
    for label_id, (class_name, rgb) in enumerate(zip(SCANNET_SEMANTIC_CLASS_NAMES, SCANNET_SEMANTIC_PALETTE)):
        r, g, b = [int(value) for value in rgb]
        legend_entries.append((label_id, class_name, f"rgb({r}, {g}, {b})", f"#{r:02x}{g:02x}{b:02x}"))
    legend_entries.append((255, "ignore / unlabeled", "rgb(80, 80, 80)", "#505050"))

    rows = "".join(
        f"""
        <tr>
            <td style='padding:4px 10px; text-align:right; font-family:monospace;'>{label_id}</td>
            <td style='padding:4px 10px;'>{class_name}</td>
            <td style='padding:4px 10px; font-family:monospace;'>{rgb_text}</td>
            <td style='padding:4px 10px;'><span style='display:inline-block; width:44px; height:16px; background:{hex_color}; border:1px solid #555; vertical-align:middle;'></span></td>
        </tr>
        """
        for label_id, class_name, rgb_text, hex_color in legend_entries
    )

    display(
        HTML(
            f"""
            <table style='border-collapse:collapse; font-size:13px;'>
                <thead>
                    <tr>
                        <th style='padding:4px 10px; text-align:right;'>label</th>
                        <th style='padding:4px 10px; text-align:left;'>class</th>
                        <th style='padding:4px 10px; text-align:left;'>color</th>
                        <th style='padding:4px 10px; text-align:left;'>swatch</th>
                    </tr>
                </thead>
                <tbody>{rows}</tbody>
            </table>
            """
        )
    )


def semantic_labels_to_colors_for_visuals(labels):
    return semantic_labels_to_uint8_colors(labels)


def write_semantic_colored_ply(output_path, coords, labels):
    coords = np.asarray(coords, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.int32)
    colors = semantic_labels_to_colors_for_visuals(labels)
    return write_point_cloud_ply(output_path, coords, colors, labels)


def save_visual_predictions_as_ply(visual_data, output_dir, max_items=4):
    ply_dir = Path(output_dir) / "semantic_visuals"
    saved_paths = []

    for item_idx, item in enumerate(visual_data[:max_items]):
        coords = torch.as_tensor(item["coords"]).cpu().numpy().astype(np.float32)
        preds = torch.as_tensor(item["preds"]).cpu().numpy().astype(np.int32)
        labels = torch.as_tensor(item["labels"]).cpu().numpy().astype(np.int32)

        num_points = min(coords.shape[0], preds.shape[0], labels.shape[0])
        coords = coords[:num_points]
        preds = preds[:num_points]
        labels = labels[:num_points]

        pred_path = ply_dir / f"sample_{item_idx:03d}_pred.ply"
        gt_path = ply_dir / f"sample_{item_idx:03d}_gt.ply"
        saved_paths.append(write_semantic_colored_ply(pred_path, coords, preds))
        saved_paths.append(write_semantic_colored_ply(gt_path, coords, labels))

    if saved_paths:
        print(f"Saved semantic-colored prediction/GT PLY files to {ply_dir}")
    return saved_paths


def save_latest_visual_data_as_plys(visual_data, output_dir, max_items=4):
    if not visual_data:
        raise ValueError("No visual data available. Run a checkpoint inference cell first.")
    return save_visual_predictions_as_ply(visual_data, str(output_dir), max_items=max_items)
