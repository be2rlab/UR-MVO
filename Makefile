# ------------------------------------------------------------------------------
#                                ALIASES
# ------------------------------------------------------------------------------

MKFILE_PATH := $(abspath $(lastword $(MAKEFILE_LIST)))
MKFILE_DIR := $(dir $(MKFILE_PATH))
ROOT_DIR := $(MKFILE_DIR)

DOCKER_COMPOSE_FILES := \
	-f $(ROOT_DIR)/docker-compose.yaml

RENDER_DISPLAY := $(DISPLAY)
DATA_DIR?="/media/zein/Samsung_T5/SLAM/Underwater SLAM/"
USER_ID=1000#$(UID)
GROUP_ID=1000#$(GID)

BASE_PARAMETERS := \
	ROOT_DIR=$(ROOT_DIR) \
	DATA_DIR=$(DATA_DIR)

# ------------------------------------------------------------------------------
#                              BUILDING COMMANDS
# ------------------------------------------------------------------------------

build:
	@echo "Building UR_MVO"
	cd $(ROOT_DIR) && sudo ROOT_DIR=$(ROOT_DIR) docker compose $(DOCKER_COMPOSE_FILES) build --build-arg USER_ID=$(USER_ID) --build-arg GROUP_ID=$(GROUP_ID) UR_MVO
# ------------------------------------------------------------------------------
#                              RUNNING COMMANDS
# ------------------------------------------------------------------------------

run:
	@echo "Running VO"
	cd $(ROOT_DIR) && \
	export $(BASE_PARAMETERS) && \
	sudo ROOT_DIR=$(ROOT_DIR) DATA_DIR=$(DATA_DIR) docker compose $(DOCKER_COMPOSE_FILES) run UR_MVO

pre-visualization:
	xhost +local:
	DISPLAY=$(DISPLAY) xhost +
	RCUTILS_COLORIZED_OUTPUT=1

build-visualization:
	cd $(ROOT_DIR) && \
	export $(BASE_PARAMETERS) && \
	xhost local:docker  && \
	docker compose build visualization


run-visualization: pre-visualization
	cd $(ROOT_DIR) && \
	export $(BASE_PARAMETERS) && \
	docker compose run visualization

