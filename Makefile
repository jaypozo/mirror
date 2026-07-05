.PHONY: schema pull embed bot check

-include .env
export

PYTHON ?= python3

schema:
	psql "$$DATABASE_URL" -f db/schema.sql

pull:
	$(PYTHON) -m ingest.pull

embed:
	$(PYTHON) -m ingest.embed

embed-style:
	$(PYTHON) -m ingest.embed_style

bot:
	$(PYTHON) -m agent.bot

check:
	$(PYTHON) -m compileall ingest agent
