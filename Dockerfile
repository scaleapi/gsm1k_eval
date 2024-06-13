# syntax=docker/dockerfile:1.3

FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-devel

# Default Environment Variables
ENV PYTHONFAULTHANDLER=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONHASHSEED=random

# Build-time Environment Variables
ARG PIP_NO_CACHE_DIR=off
ARG PIP_DISABLE_PIP_VERSION_CHECK=on
ARG PIP_DEFAULT_TIMEOUT=100
ARG DEBIAN_FRONTEND=noninteractive
ARG DEBCONF_NONINTERACTIVE_SEEN=true

# Install Debian packages
RUN apt-get -yq update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
  build-essential \
  apt-utils \
  dumb-init \
  git \
  gcc \
  ssh \
  htop \
  iftop \
  vim \
  apt-transport-https \
  ca-certificates \
  gnupg \
  curl \
  zlib1g-dev \
  libjpeg-dev \
  libsm6 \
  libxext6 \
  libxrender-dev \
  libgl1-mesa-glx \
  libglib2.0-0 \
  libgtk2.0-dev \
  libssl-dev \
  libbz2-dev \
  libreadline-dev \
  libsqlite3-dev \
  wget \
  llvm \
  libncurses5-dev \
  libncursesw5-dev \
  xz-utils \
  tk-dev \
  libffi-dev \
  liblzma-dev \
  python3-openssl \
  libcurl4-openssl-dev \
  libssl-dev \
  python3-dev \
  gcc \
  && apt-get autoremove -y \
  && rm -rf /var/lib/apt/lists/* \
  && apt-get clean

# Install s5cmd for fast reads from s3
RUN conda install -y -c conda-forge s5cmd
RUN pip3 install --upgrade pip --no-cache-dir

# CACHE INSTALL
COPY . /workspace/research/hugh/lm-evaluation-harness
WORKDIR /workspace/research/hugh/lm-evaluation-harness
RUN pip3 install -e .
RUN pip3 install -r requirements_for_docker.txt
RUN pip3 install transformers==4.41.2
RUN pip3 install flash-attn==2.5.8 --no-build-isolation

# set FORCE_CUDA because during `docker build` cuda is not accessible
ENV FORCE_CUDA="1"
ENV DD_SERVICE lm-evaluation-harness
ENV AWS_DEFAULT_REGION=us-west-2

WORKDIR /workspace/
