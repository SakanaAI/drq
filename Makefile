.PHONY: test test-prolog test-python clean

test: test-prolog test-python

test-prolog:
	cd tests/prolog && swipl -g "consult(run_all), run_tests" -t halt \
		test_archive.pl test_warrior.pl test_strategy.pl \
		test_constraints.pl test_drq.pl

test-python:
	uv run pytest tests/python/ -v

clean:
	find . -name "*.pyc" -delete
	find . -name "__pycache__" -delete
