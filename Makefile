.PHONY: help install install-dev compile test lint docker-up docker-down \
        deploy-l1 deploy-l2 build-jacobi clean

# ── Default target ─────────────────────────────────────────────────────────────
help:
	@echo "INFT monorepo — available targets:"
	@echo ""
	@echo "  install        Install Python dependencies"
	@echo "  install-dev    Install with dev extras (pytest)"
	@echo "  compile        Compile both L1 and L2 Solidity contracts"
	@echo "  test           Run all Python tests"
	@echo "  lint           Run ruff linter across l1/ and l2/"
	@echo ""
	@echo "  docker-up      Start full stack (detached)"
	@echo "  docker-down    Stop full stack"
	@echo "  docker-obs     Start full stack + observability profile"
	@echo ""
	@echo "  deploy-l1      Deploy L1 InferenceToken to Base Sepolia"
	@echo "  deploy-l2      Deploy L2 contracts to Sepolia"
	@echo ""
	@echo "  build-jacobi   Build jacobi-server C++ binary (Jacobi parallel decoding)"
	@echo "  clean          Remove build artifacts and caches"

# ── Python ─────────────────────────────────────────────────────────────────────
install:
	pip install -e .

install-dev:
	pip install -e ".[dev,miner]"

# ── Contracts ──────────────────────────────────────────────────────────────────
compile:
	cd l1 && npx hardhat compile
	cd l2 && npx hardhat compile

compile-l1:
	cd l1 && npx hardhat compile

compile-l2:
	cd l2 && npx hardhat compile

# ── Tests ──────────────────────────────────────────────────────────────────────
test:
	python -m pytest l2/tests/ -v

test-contracts:
	cd l1 && npx hardhat test
	cd l2 && npx hardhat test

# ── Lint ───────────────────────────────────────────────────────────────────────
lint:
	ruff check l1/ l2/ --exclude='__pycache__'

# ── Docker ─────────────────────────────────────────────────────────────────────
docker-up:
	docker compose up -d

docker-down:
	docker compose down

docker-obs:
	docker compose --profile observability up -d

docker-logs:
	docker compose logs -f

# ── Deploy ─────────────────────────────────────────────────────────────────────
deploy-l1:
	cd l1 && npx hardhat run scripts/deploy.js --network baseSepolia

deploy-l2:
	cd l2 && npx hardhat run scripts/deploy_l1.js --network sepolia

# ── Jacobi fork ───────────────────────────────────────────────────────────────
build-jacobi:
	cd l2/jacobi && bash build.sh Release

clean-jacobi:
	rm -rf l2/jacobi/llama.cpp/build

# ── Clean ──────────────────────────────────────────────────────────────────────
clean: clean-jacobi
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name '*.egg-info' -exec rm -rf {} + 2>/dev/null || true
	find . -name '*.pyc' -delete 2>/dev/null || true
	rm -rf l1/artifacts l1/cache l2/artifacts l2/cache
