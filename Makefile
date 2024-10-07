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

scrape-investing:
	$(PYTHON) -m src.scripts.selenium_investing --date='$(DATE)' --suffix='$(SUFFIX)'

# Run Main Predict Pipeline
predict-stocks:
	$(PYTHON) -m src.main.main_predict_stock --date='$(DATE)' --stock='$(STOCK)' --suffix='$(SUFFIX)' --pipeline=$(RUN_SCRAPER)  --pipeline=$(RUN_SUMMARIZER) --pipeline=$(RUN_PREDICTION)  --pipeline=$(PREPARE_STREAMLIT) 

