# Flight Delay Prediction Pridiction

This challenge was designed specifically for the AI Tunisia Hack 2019, which takes place from 20 to 22 September. Welcome to the AI Tunisia Hack participants!

After AI Hack Tunisia, this competition will be re-opened as a Knowledge Challenge to allow others in the Zindi community to learn and test their skills.

Flight delays not only irritate air passengers and disrupt their schedules but also cause :

a decrease in efficiency
an increase in capital costs, reallocation of flight crews and aircraft
an additional crew expenses
As a result, on an aggregate basis, an airline's record of flight delays may have a negative impact on passenger demand.

This competition aims to predict the estimated duration of flight delays per flight

This solution proposes to build a flight delay predictive model using Machine Learning techniques. The accurate prediction of flight delays will help all players in the air travel ecosystem to set up effective action plans to reduce the impact of the delays and avoid loss of time, capital and resources.

from https://zindi.africa/competitions/ai-tunisia-hack-5-predictive-analytics-challenge-2

---
## Requirements and Environment

Requirements:
- pyenv with Python: 3.9.8

Environment: 

For installing the virtual environment you can either use the Makefile and run `make setup` or install it manually with the following commands: 

```Bash
pyenv local 3.9.8
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Usage

In order to train the model and store test data in the data folder and the model in models run:

```bash
#activate env
source .venv/bin/activate

python scripts/train_xgb.py data/Train.csv      
```

In order to make predictions for a new dataset:

```bash
python scripts/predict_xgb.py data/Test.csv 
```

## Limitations

Development libraries are part of the production environment, normally these would be separate as the production code should be as slim as possible.
