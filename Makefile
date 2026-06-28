PROJECT_NAME := aiexpert
IMAGE_NAME := snuvgilab/${PROJECT_NAME}
SHM_SIZE := 64gb
CUDA_ARCHITECTURES ?= 86
DIR ?= user0
GPU_ID ?= 0
PORT ?= 9000
NERFSTUDIO_PORT ?= 10000
USER_ID ?= 0
SESSION ?= 3d-perception
SESSIONS := 3d-perception siren-and-nerf nerf-and-3dgs

all: build copy-materials run-users

run-users: run-user0 run-user1 run-user2 run-user3 run-user4 run-user5 run-user6 run-user7

build:
	docker build \
		--tag ${IMAGE_NAME}:latest \
		--build-arg USERNAME=$$(whoami) \
		--build-arg USER_UID=$$(id -u) \
		--build-arg USER_GID=$$(id -g) \
		--build-arg CUDA_ARCHITECTURES="${CUDA_ARCHITECTURES}" \
		-f Dockerfile .

run:
	if [ ! -d "${DIR}" ]; then \
		mkdir -p ${DIR}; \
	fi
	if [ ! -d "data" ]; then \
		mkdir -p data; \
	fi
	docker run \
		-itd \
		--rm \
		--shm-size ${SHM_SIZE} \
		--workdir="/app" \
		--gpus "device=${GPU_ID}" \
		--volume="$$(pwd)/${DIR}:/app" \
		--volume="$$(pwd)/data:/data" \
		-p ${PORT}:8888 \
		-p ${NERFSTUDIO_PORT}:${NERFSTUDIO_PORT} \
		-e NERFSTUDIO_PORT=${NERFSTUDIO_PORT} \
		--name aiexpert_${DIR} \
		${IMAGE_NAME}:latest


BLENDER_SCENES := chair drums ficus hotdog lego materials mic ship
BLENDER_MIRROR := https://huggingface.co/datasets/nerfbaselines/nerfbaselines-data/resolve/main/blender
download-data:
	mkdir -p data
	python3 -m pip install -q gdown
	# --- ScanNet (3D perception) ---
	if [ ! -f data/scannet_3d.zip ]; then \
		wget "https://cvg-data.inf.ethz.ch/openscene/data/scannet_processed/scannet_3d.zip" -O data/scannet_3d.zip; \
	fi
	python3 -m zipfile -e data/scannet_3d.zip data
	# --- NeRF synthetic ("blender"), from the Hugging Face mirror ---
	mkdir -p data/blender
	for s in ${BLENDER_SCENES}; do \
		if [ ! -d "data/blender/$$s/train" ]; then \
			wget "${BLENDER_MIRROR}/$$s.zip" -O data/blender/$$s.zip; \
			python3 -m zipfile -e data/blender/$$s.zip data/blender; \
			rm -f data/blender/$$s.zip; \
		fi; \
	done
	# --- mip-NeRF 360 (https://jonbarron.info/mipnerf360/), from Google Cloud Storage ---
	mkdir -p data/mipnerf360
	if [ ! -d "data/mipnerf360/bicycle" ]; then \
		wget "http://storage.googleapis.com/gresearch/refraw360/360_v2.zip" -O data/mipnerf360/360_v2.zip; \
		python3 -m zipfile -e data/mipnerf360/360_v2.zip data/mipnerf360; \
		rm -f data/mipnerf360/360_v2.zip; \
	fi
	if [ ! -d "data/mipnerf360/flowers" ]; then \
		wget "http://storage.googleapis.com/gresearch/refraw360/360_extra_scenes.zip" -O data/mipnerf360/360_extra_scenes.zip; \
		python3 -m zipfile -e data/mipnerf360/360_extra_scenes.zip data/mipnerf360; \
		rm -f data/mipnerf360/360_extra_scenes.zip; \
	fi
	# --- SIREN SDF point clouds (oriented .xyz: positions + unit normals) ---
	if [ ! -f data/thai_statue.xyz ]; then \
		gdown "1tkrHBciOzGLKZP0Pd9ye0Yz71JMdAtfl" -O data/thai_statue.xyz; \
	fi
	if [ ! -f data/interior_room.xyz ]; then \
		gdown "1SqlByPMwf6EcTJNrvlpRn5-GEO9DiYhI" -O data/interior_room.xyz; \
	fi

copy_materials_single:
	mkdir -p user${USER_ID}/${SESSION}
	tar --exclude='*_answer.ipynb' -C materials/${SESSION} -cf - . | tar -C user${USER_ID}/${SESSION} -xf -
	find user${USER_ID}/${SESSION} -name '*_answer.ipynb' -delete

copy_materials_user:
	for session in ${SESSIONS}; do \
		mkdir -p user${USER_ID}/$$session; \
		tar --exclude='*_answer.ipynb' -C materials/$$session -cf - . | tar -C user${USER_ID}/$$session -xf -; \
		find user${USER_ID}/$$session -name '*_answer.ipynb' -delete; \
	done

copy-materials:
	$(MAKE) copy_materials_user USER_ID=0
	$(MAKE) copy_materials_user USER_ID=1
	$(MAKE) copy_materials_user USER_ID=2
	$(MAKE) copy_materials_user USER_ID=3
	$(MAKE) copy_materials_user USER_ID=4
	$(MAKE) copy_materials_user USER_ID=5
	$(MAKE) copy_materials_user USER_ID=6
	$(MAKE) copy_materials_user USER_ID=7

run-user0:
	$(MAKE) run DIR=user0 GPU_ID=0 PORT=9000 NERFSTUDIO_PORT=10000

run-user1:
	$(MAKE) run DIR=user1 GPU_ID=1 PORT=9001 NERFSTUDIO_PORT=10001

run-user2:
	$(MAKE) run DIR=user2 GPU_ID=2 PORT=9002 NERFSTUDIO_PORT=10002

run-user3:
	$(MAKE) run DIR=user3 GPU_ID=3 PORT=9003 NERFSTUDIO_PORT=10003

run-user4:
	$(MAKE) run DIR=user4 GPU_ID=4 PORT=9004 NERFSTUDIO_PORT=10004

run-user5:
	$(MAKE) run DIR=user5 GPU_ID=5 PORT=9005 NERFSTUDIO_PORT=10005

run-user6:
	$(MAKE) run DIR=user6 GPU_ID=6 PORT=9006 NERFSTUDIO_PORT=10006

run-user7:
	$(MAKE) run DIR=user7 GPU_ID=7 PORT=9007 NERFSTUDIO_PORT=10007


.PHONY: all run run-users build download-data copy_materials_single copy_materials_user copy-materials run-user0 run-user1 run-user2 run-user3 run-user4 run-user5 run-user6 run-user7
