 FROM nvidia/cuda:11.3.1-devel-ubuntu20.04

# 환경변수 설정 (비인터랙티브 설치 + 타임존 지정)
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Seoul

# 시스템 패키지 설치
RUN apt-get update && apt-get install wget -yq
RUN apt-get install build-essential g++ gcc -y
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get install libgl1-mesa-glx libglib2.0-0 -y
RUN apt-get install openmpi-bin openmpi-common libopenmpi-dev libgtk2.0-dev git -y
RUN apt-get update && apt-get install python3.8 python3.8-dev python3-pip -y

# python / pip 심볼릭 링크
RUN rm -f /usr/bin/python && ln -s /usr/bin/python3.8 /usr/bin/python && \
    rm -f /usr/bin/pip && ln -s /usr/bin/pip3 /usr/bin/pip

# pip 업그레이드
RUN pip install --upgrade pip

# PyTorch + CUDA 11.3
RUN pip install torch==1.10.1+cu113 torchvision==0.11.2+cu113 torchaudio==0.10.1 \
    --extra-index-url https://download.pytorch.org/whl/cu113

# 기타 패키지
RUN pip install -U openmim "numpy<1.24"
RUN pip install Pillow==8.4.0
RUN pip install tqdm
RUN pip install torchpack
RUN pip install mmcv==1.4.0 mmcv-full==1.4.0 mmdet==2.20.0
RUN pip install nuscenes-devkit
RUN pip install mpi4py==3.0.3
RUN pip install numba==0.48.0

# 나머지 의존성
RUN pip install Pillow==8.4.0 tqdm torchpack nuscenes-devkit mpi4py==3.0.3 numba==0.48.0
RUN pip install --default-timeout=100 tensorflow==2.12.0 waymo-open-dataset-tf-2-12-0