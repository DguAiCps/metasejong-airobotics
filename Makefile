.PHONY: build run clean build-ws clean-ws check-env link-answer-sheet-folder

METASEJONG_PROJECT_PATH := ../metacom2025-metasejong
#METASEJONG_PROJECT_PATH := ../metacom2025

link-answer-sheet-folder:
	ln -s ${METASEJONG_PROJECT_PATH}/scenario-data/answer-sheets demo

# Docker operations
build:
	docker compose build

run:
	docker compose up

clean:
	docker compose down
	rm -rf metasejong_competitor_ws/build metasejong_competitor_ws/install

# Development operations
build-ws:
	cd metasejong_competitor_ws && colcon build --symlink-install

clean-ws:
	rm -rf metasejong_competitor_ws/build metasejong_competitor_ws/install

# Environment setup
check-env:
	@if [ -z "$(ENV_METASEJONG_TEAM_NAME)" ] || [ -z "$(ENV_METASEJONG_TEAM_TOKEN)" ] || [ -z "$(ENV_METASEJONG_TEAM_TARGET_STAGE)" ]; then \
		echo "Error: Required environment variables are not set"; \
		echo "Please set ENV_METASEJONG_TEAM_NAME, ENV_METASEJONG_TEAM_TOKEN, and ENV_METASEJONG_TEAM_TARGET_STAGE"; \
		exit 1; \
	fi 
