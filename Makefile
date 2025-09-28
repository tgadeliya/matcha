
RUFF_LINE_LENGTH := 88

fix: lint_docs ruff_fmt ruff_check

ruff_fmt:
	uvx ruff format 

ruff_check:
	uvx ruff check

lint_docs:
	uvx docformatter --in-place -r \
		--wrap-summaries=$(RUFF_LINE_LENGTH) \
		--wrap-descriptions=$(RUFF_LINE_LENGTH) \
		.