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