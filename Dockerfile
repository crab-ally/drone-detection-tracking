# ROS 2 Humble Desktop 기본 이미지 사용 (Ubuntu 22.04)
FROM osrf/ros:humble-desktop-full

# 빌드 시그널 방지 및 비인터랙티브 모드 설정
ENV DEBIAN_FRONTEND=non-interactive

# 시스템 패키지 업데이트 및 OpenCV 구동에 필요한 그래픽/비디오 라이브러리 설치
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip \
    python3-colcon-common-extensions \
    terminator \
    tree \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# 작업 디렉터리 설정
WORKDIR /workspace

# Python 패키지 설치
COPY requirements.txt .
RUN python3 -m pip install --no-cache-dir --upgrade pip && \
    python3 -m pip install --no-cache-dir --ignore-installed sympy -r requirements.txt

# 소스 코드 및 워크스페이스 디렉터리 복사
COPY . /workspace

# ROS 2 환경 자동 로딩을 위한 bashrc 설정
RUN echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc

# 기본 실행 명령어
CMD ["bash"]