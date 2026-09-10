PYTHON ?= python3

.PHONY: silver benchmark test

silver:
	$(PYTHON) -m src.transform.build_silver

benchmark:
	@test -n "$(MODEL)" || (echo "Utilisation : make benchmark MODEL=nom [LIMIT=20]"; exit 1)
	$(PYTHON) -m src.enrich.run_benchmark --model "$(MODEL)" $(if $(LIMIT),--limit "$(LIMIT)",)

test:
	$(PYTHON) -m unittest discover -s tests -v
