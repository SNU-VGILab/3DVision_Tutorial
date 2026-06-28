# 3DVision Tutorial

> **3D Vision 강의용 실습 자료**

GPU 서버 한 대에서 Docker 이미지를 빌드한 뒤, 수강생별로 GPU와 포트를 분리한 컨테이너
(`user0`~`user7`)를 띄우고 각자 JupyterLab으로 접속해 실습 진행. 모든 무거운 의존성
(COLMAP, GLOMAP, WarpConvNet, nerfstudio, gsplat 등)은 이미지에 미리 포함되어 있음.

### 강의 기록

- **2024** — 삼성전자 AI Expert 강의 (조교: 도승욱)
- **2025** — 삼성전자 AI Expert 강의 (조교: 도승욱, 이다은)
- **2026** — 삼성전자 AI Expert 강의 (조교: 서민균, 이다은)

---

## Prerequisites

- **NVIDIA GPU** — one GPU per student is recommended (the default setup assumes 8 = `user0`–`user7`).
- **NVIDIA Driver + [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)** — required for `--gpus` inside the container.
- **Docker** — `make` wraps the build/run commands.
- **Disk / memory headroom** — the image build compiles COLMAP, GLOMAP, tiny-cuda-nn, etc. from
  source, so **the build takes a long time** and uses a lot of disk space.

> Base image: `pytorch/pytorch:2.5.1-cuda12.1-cudnn9-devel` (CUDA 12.1).
> Make sure the host driver supports the CUDA 12.1 runtime.

---

## Quick Start

For a full first-time setup, use `make all`. It builds the Docker image, downloads the shared
data, distributes all lecture materials to `user0`-`user7`, and launches all 8 containers.

```bash
make all
```

Then open the assigned Jupyter port in a browser. For `user0`:

```text
http://<server-ip>:9000
```

If you want to run each step manually, follow the sections below.

---

## Repository Structure

```
3DVision_Tutorial/
├── Dockerfile          # Tutorial environment image (CUDA/PyTorch + all 3D libraries)
├── Makefile            # Build / run / data-prep / material-distribution commands
├── data/               # Shared data dir (mounted as /data in the container, git-ignored)
│   └── .placeholder
├── materials/          # Lecture materials (per-topic folders, see "Lecture Materials" below)
│   ├── 3d-perception/
│   ├── siren-and-nerf/
│   └── nerf-and-3dgs/
└── user0/ ... user7/   # Per-student workspaces (auto-created on run, mounted as /app, git-ignored)
```

`data/` and `user*/` are listed in `.gitignore` and are not version-controlled.

---

## 0. Full Setup With `make all`

If you want the standard 8-user classroom setup, run only:

```bash
make all
```

This runs the full setup sequence:

1. `make build`
2. `make download-data`
3. `make copy-materials`
4. `make run-users`

After this, you can skip the manual build/data-download/material-copy/container-launch steps below
unless you need to customize or rerun a specific part.

---

## 1. Build the Image

```bash
make build
```

- Resulting image: `snuvgilab/aiexpert:latest`
- The build **injects the host UID/GID** (`--build-arg`) so the in-container user matches the host
  user. This avoids permission conflicts when reading/writing files in the mounted `./userN` dirs.
- Key components: COLMAP 3.9.1, GLOMAP 1.0.0, WarpConvNet, hloc v1.4, tiny-cuda-nn,
  gsplat v1.4.0, nerfstudio v1.1.5, JupyterLab.

---

## 2. Run Containers / Per-Student Deployment (core workflow)

The `run-userN` target pins student N to **GPU N**, **Jupyter port `900N`**, and
**nerfstudio port `1000N`**.

| Target        | DIR     | GPU | Jupyter port | nerfstudio port |
|---------------|---------|-----|--------------|-----------------|
| `run-user0`   | `user0` | 0   | 9000         | 10000           |
| `run-user1`   | `user1` | 1   | 9001         | 10001           |
| `run-user2`   | `user2` | 2   | 9002         | 10002           |
| `run-user3`   | `user3` | 3   | 9003         | 10003           |
| `run-user4`   | `user4` | 4   | 9004         | 10004           |
| `run-user5`   | `user5` | 5   | 9005         | 10005           |
| `run-user6`   | `user6` | 6   | 9006         | 10006           |
| `run-user7`   | `user7` | 7   | 9007         | 10007           |

```bash
make run-users    # start user0-user7 without rebuilding
make run-user3    # start a single student
```

**Mounts**

- `./userN` → container `/app` (per-student workspace, auto-created on launch if missing)
- `./data` → container `/data` (shared across all students, read-only data)

**Access**

- The container serves JupyterLab on `0.0.0.0:8888`, mapped to host port `900N`.
- Open `http://<server-ip>:900N` in a browser.
- The nerfstudio `viewer` is exposed via port `1000N` (used in the NeRF/3DGS sessions).

**Customize** — set the variables directly to launch any combination.

```bash
make run DIR=ta GPU_ID=0 PORT=9100 NERFSTUDIO_PORT=10100
```

> Containers run with `--rm`, so stopping one removes the container itself, but your work remains
> in the mounted `./userN` directory.

---

## 3. Prepare Data

```bash
make download-data
```

Downloads ScanNet's preprocessed 3D data (hosted by ETH, OpenScene) and extracts it into `./data`.
**Required for the semantic segmentation exercise in the 3D Perception session.** The other
sessions (SIREN / NeRF / 3DGS) need no separate data download.

---

## 4. Distribute Materials

Copy the notebooks and helper code for all sessions into the student workspaces (`./user0`-`./user7`):

```bash
make copy-materials
```

This creates one folder per session inside each student workspace:

```text
user0/
  3d-perception/
  siren-and-nerf/
  nerf-and-3dgs/
```

To copy only one session to one user:

```bash
make copy_materials_single USER_ID=0 SESSION=3d-perception
```

Answer notebooks (`*_answer.ipynb`) are not distributed to student folders. If old answer
notebooks were already copied into `userN`, `make copy-materials` removes them.

---

## Lecture Materials

Each notebook ships as an exercise (`*.ipynb`) / solution (`*_answer.ipynb`) pair.

### `materials/3d-perception` — 3D Perception
3D object classification and semantic segmentation of indoor scenes.

- `3DPerception.ipynb` / `3DPerception_answer.ipynb` — point cloud / voxel representations,
  voxel-size visualization, SparseConv implementation, SparseConvNet semantic segmentation,
  and PointTransformerV3 semantic segmentation.
- `mink_unet.py`, `point_transformer_v3.py` — model definitions imported by the notebook.
- **Required data:** ScanNet (`make download-data`).

### `materials/siren-and-nerf` — SIREN & NeRF
- `siren.ipynb` / `siren_answer.ipynb` — SIREN: image fitting, Poisson's equation, and SDF fitting
  from point clouds (vs. a ReLU baseline).
- `nerf.ipynb` / `nerf_answer.ipynb` — NeRF basics: TinyNeRF implementation and NeRFStudio usage.

### `materials/nerf-and-3dgs` — NeRF & 3D Gaussian Splatting
- `nerf_and_3dgs.ipynb` — train and customize Nerfacto, Instant-NGP, and Splatfacto (3DGS) with
  NeRFStudio across various scenes.
- `wandb-tutorial.ipynb` — experiment tracking/tuning with Weights & Biases (W&B) (in Korean).
  Before starting, get an API key at [https://wandb.ai/authorize](https://wandb.ai/authorize).

---

## 참고

- 본 자료는 SNU VGI-Lab 내부 자료로, 랩에서 진행하는 3D Vision 강의용으로 제작되었습니다.
