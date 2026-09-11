PYTHON ?= python3
PROJECT_ROOT := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
DBT ?= $(PROJECT_ROOT)/.venv/bin/dbt
STREAMLIT ?= $(PROJECT_ROOT)/.venv/bin/streamlit
DBT_FLAGS := --project-dir "$(PROJECT_ROOT)/dbt" --profiles-dir "$(PROJECT_ROOT)/dbt"

.PHONY: silver benchmark test dbt-debug dbt-build dbt-clean dashboard

silver:
	$(PYTHON) -m src.transform.build_silver

benchmark:
	$(PYTHON) -m src.enrich.run_benchmark $(if $(MODEL),--model "$(MODEL)",) $(if $(LIMIT),--limit "$(LIMIT)",)

test:
	$(PYTHON) -m unittest discover -s tests -v

dbt-debug:
	BI_BENCHMARK_ROOT="$(PROJECT_ROOT)" $(DBT) debug $(DBT_FLAGS)

dbt-build:
	BI_BENCHMARK_ROOT="$(PROJECT_ROOT)" $(DBT) build $(DBT_FLAGS)

dbt-clean:
	BI_BENCHMARK_ROOT="$(PROJECT_ROOT)" $(DBT) clean $(DBT_FLAGS)

dbt-docs:
	BI_BENCHMARK_ROOT="$(PROJECT_ROOT)" $(DBT) docs generate $(DBT_FLAGS)

dashboard:
	cd "$(PROJECT_ROOT)" && "$(STREAMLIT)" run "$(PROJECT_ROOT)/streamlit_app.py"
