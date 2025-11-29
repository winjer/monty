.DEFAULT_GOAL := all

.PHONY: .cargo
.cargo: ## Check that cargo is installed
	@cargo --version || echo 'Please install cargo: https://github.com/rust-lang/cargo'

.PHONY: .pre-commit
.pre-commit: ## Check that pre-commit is installed
	@pre-commit -V || echo 'Please install pre-commit: https://pre-commit.com/'

.PHONY: install
install: .cargo .pre-commit ## Install the package, dependencies, and pre-commit for local development
	cargo check
	pre-commit install --install-hooks

.PHONY: lint-rs
lint-rs:  ## Lint Rust code with fmt and clippy
	@cargo fmt --version
	cargo fmt --all -- --check
	@cargo clippy --version
	cargo clippy --tests -- -D warnings -A incomplete_features -W clippy::dbg_macro

.PHONY: lint-py
lint-py: ## Lint Python code with ruff
	uv run ruff format
	uv run ruff check --fix --fix-only
	uv run basedpyright

.PHONY: lint
lint: lint-rs lint-py ## Lint the code with ruff and clippy

.PHONY: test
test: ## Run tests
	cargo test

.PHONY: test-ref-counting
test-ref-counting: ## Run tests with ref-counting enabled
	cargo test --features ref-counting

.PHONY: complete-tests
complete-tests: ## Fill in incomplete test expectations using CPython
	uv run scripts/complete_tests.py

.PHONY: bench
bench: ## Run benchmarks
	cargo bench --bench main

.PHONY: profile
profile: ## Profile the code with pprof and generate flamegraphs
	cargo bench --bench main --profile profiling -- --profile-time=5
	uv run scripts/flamegraph_to_text.py

.PHONY: all
all: lint test
