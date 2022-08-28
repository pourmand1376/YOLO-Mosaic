## Convert images to mosaic format. This file converts each four image into one single image. 

.ONESHELL:

SHELL=/bin/bash
CONDA_ACTIVATE = source $$(conda info --base)/etc/profile.d/conda.sh ; conda activate ; conda activate

.PHONY:help
help:
	@egrep -h '\s##\s' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m  %-30s\033[0m %s\n", $$1, $$2}'

create_env: ## create a new environment for this project
	source $$(conda info --base)/etc/profile.d/conda.sh
	conda create -n mosaic python=3.10.4

.PHONY: install
install: ## Install required dependencies for this project
	$(CONDA_ACTIVATE) mosaic
	pip install -r requirements.txt

remove_log: ## Removes log file
	rm debug.log

help_main: ## Show main.py python help file
	$(CONDA_ACTIVATE) mosaic
	python main.py --help

temp_convert: remove_log ## convert dataset to new format
	$(CONDA_ACTIVATE) mosaic
	python main.py \
		'/media/amir/external_1TB/Dataset/Anevrism/output_folder2/images' \
		'/media/amir/external_1TB/Dataset/Anevrism/output_folder2/labels' \
		'/media/amir/external_1TB/Dataset/Anevrism/output_mosaic' \
		--output-height 1000 --output-width 1000

convert: remove_log ## convert dataset to new format
	$(CONDA_ACTIVATE) mosaic
	python main.py convert-database \
		'/mnt/new_ssd/projects/Anevrism/Data/brain_cta/output_folder/database.yaml' \
		'/mnt/new_ssd/projects/Anevrism/Data/brain_cta/output_mosaic/'