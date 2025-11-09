## Setup the Environment

```bash
python -m venv .venv
source .venv/bin/activate 
pip install -r requirements.txt
cd TransformerLens
pip install -e .
cd ..
```

## Download the Data

```bash
git lfs install
git lfs init
git lfs pull
```

## Run the Notebooks
Run the cells in run_patcher.ipynb to generate the patcher results.