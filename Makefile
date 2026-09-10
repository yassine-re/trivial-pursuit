PYTHON ?= python3

.PHONY: silver benchmark test

silver:
	$(PYTHON) -m src.transform.build_silver

benchmark:
	$(PYTHON) -m src.enrich.run_benchmark $(if $(MODEL),--model "$(MODEL)",) $(if $(LIMIT),--limit "$(LIMIT)",)

test:
	$(PYTHON) -m unittest discover -s tests -v
