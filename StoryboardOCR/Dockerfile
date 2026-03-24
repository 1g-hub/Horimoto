# Dockerfile
# NVIDIA CUDA 公式イメージをベースにする
FROM nvidia/cuda:12.2.2-cudnn8-runtime-ubuntu22.04

# 環境変数の設定
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# 必要なシステムパッケージのインストール
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-dev \
    python3-pip \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 20260106 追加（PDF2PNG のため）
RUN apt-get update && apt-get install -y poppler-utils 

# Pythonのエイリアスを設定
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1
RUN update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

# 作業ディレクトリの設定
WORKDIR /app

# requirements.txtをコピーして依存関係をインストール
COPY requirements.txt .
# RUN pip install --no-cache-dir -r requirements.txt
RUN python3 -m pip install --no-cache-dir -U pip setuptools wheel
RUN python3 -m pip --version
RUN python3 -m pip install --no-cache-dir -r requirements.txt

# コンテナにコピー
COPY . .

# コンテナ起動時のデフォルトコマンド
# Makefileで実行スクリプトを指定するため、ここでは特に設定しない
CMD ["python", "--version"]