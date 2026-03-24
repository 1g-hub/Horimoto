    
# Makefile

# Dockerイメージ名
IMAGE_NAME := qwen3-vl-econt
TAG := latest

# 命名規則 : (環境名).(使用者の名前)
CONTAINER_NAME := qwen3-vl-econt.horimoto

.PHONY: all build run train clean

all: build run

build: ## Dockerイメージをビルドします
	docker build -t $(IMAGE_NAME):$(TAG) .

run: ## Dockerコンテナを起動し、bashシェルに入ります
	docker run --rm -it --gpus all \
		--name $(CONTAINER_NAME) \
		-v $(PWD):/app \
		-p 7801:7801 \
		$(IMAGE_NAME):$(TAG) bash

clean: ## Dockerイメージを削除します
	docker rmi $(IMAGE_NAME):$(TAG) || true

help: ## このMakefileのヘルプメッセージを表示します
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

  