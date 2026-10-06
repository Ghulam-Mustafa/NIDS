.PHONY: install prepare train evaluate all test
install:
	pip install -r requirements.txt
prepare:
	python -m nids prepare
train:
	python -m nids train
evaluate:
	python -m nids evaluate
all:
	python -m nids run-all
test:
	pytest -q
