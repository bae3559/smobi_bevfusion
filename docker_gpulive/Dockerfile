FROM nvidia/cuda:11.3.1-devel-ubuntu20.04
# 환경변수 설정 (비인터랙티브 설치 + 타임존 지정)
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Seoul

# root로 전환
USER root   

# 시스템 패키지 업데이트 및 필수 패키지 설치
# - openssh-server: SSH 서버 (원격 접속용)
# - supervisor: 여러 프로세스를 관리하는 데몬
# - sudo: 관리자 권한 실행을 위한 패키지
# - python3-pip: Python 패키지 관리자
RUN apt update \
    && apt install -y openssh-server \
    supervisor \
    sudo \
    python3-pip 

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
RUN pip install mmdet==2.20.0
RUN pip install nuscenes-devkit
RUN pip install mpi4py==3.0.3
RUN pip install numba==0.48.0
RUN pip install mmcv-full==1.4.0 -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.10.0/index.html
RUN pip install yapf==0.32.0
RUN pip install --default-timeout=100 tensorflow==2.12.0 waymo-open-dataset-tf-2-12-0

# 패키지 캐시 정리 (이미지 크기 최적화)
RUN apt clean

### SSH 공개키 설정 ###
# SSH 접속을 위한 공개키를 빌드 시점에 전달받습니다
# PEM 키에서 추출한 공개키 내용을 입력하세요
# 예시) ARG RSA_PUBLIC_KEY="ssh-rsa AAAAB3NEmoUDZ~~~"
ARG RSA_PUBLIC_KEY="ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAABAQC8L0fpQSd6sZCDTc2nN0LVXMGOt2531Ydq7gjxo7McskOLoNGkgF34+0vXsDUImcBa6RKLjzXICSgL1uH1Q7ZNsS8Pkfiip3fUu4I33TDWfMaPAAWgv5iUUM7+ekSCd/Zv16ttNEyGGyAtg9HtEdxdLg2CoxuEObLmYTJw9Cs+7lyvm3BbHFeIj0/s8+vyb2t9cDoHKaHsjQh+QJtaQg8F9PqdIewmfwF1GjYl/Ipdx61mCOVSLK3MHBf1RPGwJVY4/qtez1teYYOsMacWDn4JvAFDUk50S2hPB2850YoJ9kuNizoTsG7Qh1dhbForXur3ufxzFtAxbMcJ4TORz/ip"
##########################

### 사용자 계정 설정 ###
# 기본 ubuntu 계정을 삭제하고 새로 생성 (보안상 이유)
RUN if getent passwd ubuntu >/dev/null 2>&1; then \
      (command -v deluser >/dev/null 2>&1 && deluser --remove-home ubuntu) \
      || (command -v userdel >/dev/null 2>&1 && userdel -r ubuntu || true); \
    fi
# ubuntu 사용자 그룹 및 계정 생성
# - UID/GID: 1001 (일반적으로 첫 번째 사용자)
# - 홈 디렉토리 생성 (-m)
# - 기본 셸: bash (-s /bin/bash)
# - sudo 권한 부여 (패스워드 없이 모든 명령 실행 가능)
RUN groupadd -g 1001 ubuntu \
    && useradd -u 1001 -g 1001 -m -s /bin/bash ubuntu \
    && echo "ubuntu ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/ubuntu


### SSH 서버 설정 ###
# SSH 서버 설정 및 보안 강화
RUN mkdir -p /var/run/sshd \
    && echo "PermitRootLogin no" >> /etc/ssh/sshd_config \
    && echo "PasswordAuthentication no" >> /etc/ssh/sshd_config \
    && echo "PubkeyAuthentication yes" >> /etc/ssh/sshd_config \
    && echo "AuthorizedKeysFile .ssh/authorized_keys" >> /etc/ssh/sshd_config


# ubuntu 사용자의 SSH 공개키 등록
RUN mkdir -p /home/ubuntu/.ssh

# 공개키를 authorized_keys에 추가하고 적절한 권한 설정
# - chmod 700: .ssh 디렉토리 (소유자만 읽기/쓰기/실행)
# - chmod 600: authorized_keys 파일 (소유자만 읽기/쓰기)
RUN echo $RSA_PUBLIC_KEY >> /home/ubuntu/.ssh/authorized_keys \
    && chown -R ubuntu:ubuntu /home/ubuntu/.ssh \
    && chmod 700 /home/ubuntu/.ssh \
    && chmod 600 /home/ubuntu/.ssh/authorized_keys

### 추가 사용자 생성 (선택사항) ###
# 필요시 NEW_USER 변수에 원하는 사용자명을 입력하세요
# 추가 사용자 생성 및 설정
# - UID/GID: 1002 (두 번째 사용자)
# - ubuntu 그룹에 추가하여 ubuntu 사용자와 파일 공유 가능
# - sudo 권한 부여
##########################################
# ARG NEW_USER={새로운 사용자}
# RUN groupadd -g 1002 $NEW_USER \
#     && useradd -u 1002 -g 1002 -m -s /bin/bash $NEW_USER \
#     && echo "$NEW_USER ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/$NEW_USER \
#     && usermod -aG ubuntu $NEW_USER

# RUN mkdir -p /home/$NEW_USER/.ssh \
#     && echo "$RSA_PUBLIC_KEY" >> /home/$NEW_USER/.ssh/authorized_keys \
#     && chown -R $NEW_USER:ubuntu /home/$NEW_USER/.ssh \
#     && chmod 700 /home/$NEW_USER/.ssh \
#     && chmod 600 /home/$NEW_USER/.ssh/authorized_keys
##########################################

### Python 및 Jupyter 환경 설정 ###
# 데이터 분석 및 머신러닝을 위한 Python 라이브러리 설치
# --break-system-packages: 시스템 패키지와의 충돌을 무시하고 설치
RUN python3 -m pip install --no-cache-dir --break-system-packages \
    jupyterlab==4.2.5 \
    notebook==7.2.2 \
    ipykernel==6.29.5 \
    jupyter-client==8.6.0 \
    nbformat==5.10.4 \
    nbconvert==7.14.2 \
    ipywidgets==8.1.2

# pandas \            
# matplotlib==3.10.3 \       
# seaborn==0.13.2 \           
# scikit-learn==1.7.0 \        
# tqdm==4.67.1                 
### Supervisor 설정 ###
# 여러 서비스를 동시에 관리하기 위한 Supervisor 설정
RUN mkdir -p /var/log/supervisor

# Supervisor 설정 파일 생성
RUN tee /etc/supervisor/conf.d/services.conf > /dev/null <<EOF
# Supervisor 메인 설정
[supervisord]
nodaemon=true              # 데몬 모드 비활성화 (포그라운드 실행)
user=root                  # root 사용자로 실행
logfile=/var/log/supervisor/supervisord.log
pidfile=/var/run/supervisord.pid

# SSH 서버 설정
[program:sshd]
command=/usr/sbin/sshd -D  # SSH 데몬 실행
autostart=true             # 자동 시작
autorestart=true           # 자동 재시작
stderr_logfile=/var/log/sshd.err.log
stdout_logfile=/var/log/sshd.out.log
user=root                  # root로 실행

# Jupyter Notebook 설정 (포트 8888)
[program:jupyter-notebook]
command=jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --IdentityProvider.token="ubuntu" --ServerApp.allow_origin='*' --ServerApp.allow_remote_access=True --ServerApp.root_dir="/"
autostart=true             # 자동 시작
autorestart=true           # 자동 재시작
stderr_logfile=/dev/null   # 로그 출력 안함
stdout_logfile=/dev/null   # 로그 출력 안함
user=ubuntu                # ubuntu 사용자로 실행
environment=HOME="/home/ubuntu",USER="ubuntu",PATH="/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"

# Jupyter Lab 설정 (포트 8889)
[program:jupyter-lab]
command=jupyter lab --ip=0.0.0.0 --port=8889 --no-browser --IdentityProvider.token="ubuntu" --ServerApp.allow_origin='*' --ServerApp.allow_remote_access=True --ServerApp.root_dir="/"
autostart=true             # 자동 시작
autorestart=true           # 자동 재시작
stderr_logfile=/dev/null   # 로그 출력 안함
stdout_logfile=/dev/null   # 로그 출력 안함
user=ubuntu                # ubuntu 사용자로 실행
environment=HOME="/home/ubuntu",USER="ubuntu",PATH="/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
EOF


### 컨테이너 실행 설정 ###
# 컨테이너 시작 시 Supervisor를 실행하는 엔트리포인트 스크립트 생성
RUN echo "exec sudo supervisord -c /etc/supervisor/conf.d/services.conf" > /entrypoint.sh \
    && chmod +x /entrypoint.sh

# 외부에서 접근 가능한 포트 노출
# - 22: SSH 서버
# - 8888: Jupyter Notebook
# - 8889: Jupyter Lab
EXPOSE 22 8888 8889

# 기본 사용자를 ubuntu로 설정
USER ubuntu

# 컨테이너 시작 시 엔트리포인트 스크립트 실행
ENTRYPOINT [ "bash", "/entrypoint.sh" ]
