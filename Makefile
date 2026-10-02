.PHONY: check test test-training test-evaluation paper package package-paper

check:
	python -B scripts/check_release.py

test: test-training

test-training:
	python -B -m pytest tests --ignore=tests/test_evaluation_verifiers.py

test-evaluation:
	python -B -m pytest tests/test_evaluation*.py tests/test_codereasoner*.py tests/test_continuation*.py tests/test_release_tools.py

paper:
	$(MAKE) -C paper

package: check
	python -B scripts/package_release.py

package-paper: check paper
	python -B scripts/package_release.py --include-paper
