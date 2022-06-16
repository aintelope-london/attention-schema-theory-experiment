tests-local: ## run tests locally with active python environment
	pytest

tests-local-p: ## run tests locally without active python environment
	poetry run pytest

typecheck-local: ## local typechecking with active python environment
	mypy aintelope

typecheck-local-p: ## local typechecking without active python environment
	poetry run mypy aintelope

isort: ## sort python imports with active python environment
	isort .

isort-p: ## sort python imports without active python environment
	poetry run isort .
