###########################################################################################################
## VARIABLES
###########################################################################################################
PYTHON=python
CONDA=conda
STREAMLIT=streamlit
CURRENT_DIR := $(PWD)
SRC_DIR=$(CURRENT_DIR)/src
MAIN_DIR=$(SRC_DIR)/main

###########################################################################################################
## SCRIPTS
###########################################################################################################
# Create conda env to run MM
create-env:
	$(CONDA) env update --file environment.yml

# Run Main Predict Pipeline
predict-sentiments:
	$(PYTHON) -m src.main.main_predict_sentiments --date='$(DATE)' --dailyfx='$(DAILYFX)' --econtimes='$(ECONTIMES)'  --financialtimes='$(FINANCIALTIMES)' --suffix='$(SUFFIX)' --pipeline=$(RUN_SCRAPER) --pipeline=$(RUN_PREDICTION)  --pipeline=$(PREPARE_STREAMLIT) 

run-streamlit:
	$(STREAMLIT) run app.py

# Scrape Investing scripts
scrape-investing:
	$(PYTHON) -m src.scripts.selenium_investing --date='$(DATE)' --suffix='$(SUFFIX)'

# Run Main Predict Pipeline
predict-stocks:
	$(PYTHON) -m src.main.main_predict_stock --date='$(DATE)' --stock='$(STOCK)' --suffix='$(SUFFIX)' --pipeline=$(RUN_SCRAPER)  --pipeline=$(RUN_SUMMARIZER) --pipeline=$(RUN_PREDICTION)  --pipeline=$(PREPARE_STREAMLIT) 

# Export Conda Environment
conda-export-env:
	$(PYTHON) conda_export_minimal.py --s_save="env.yml"

# Run Setup
setup-bart:
	$(PYTHON) -m src.main.main_setup

# Run Streamlit for Stock App
run-streamlit-stock:
	$(STREAMLIT) run app_stock.py