.PHONY: all clean data

data: data/raw/val data/raw/test data/raw/train

clean:
	rm -rf data

data/metadata/train.in:
	mkdir -p data/metadata
	touch data/metadata/train.in
	poetry run python src/data/generate_splits.py

data/metadata/val.in:
	mkdir -p data/metadata
	touch data/metadata/val.in
	poetry run python src/data/generate_splits.py

data/metadata/test.in:
	mkdir -p data/metadata
	touch data/metadata/test.in
	poetry run python src/data/generate_splits.py

data/raw/train: data/metadata/train.in
	mkdir -p data/raw/train
	poetry run python src/data/downloader.py data/metadata/train.in data/raw/train

data/raw/val: data/metadata/val.in
	mkdir -p data/raw/val
	poetry run python src/data/downloader.py data/metadata/val.in data/raw/val

data/raw/test: data/metadata/test.in
	mkdir -p data/raw/test
	poetry run python src/data/downloader.py data/metadata/test.in data/raw/test
