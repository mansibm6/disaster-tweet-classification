# Disaster tweet classification

Compares a TF-IDF/Naive Bayes baseline, dense/recurrent/convolutional text networks, and a Universal Sentence Encoder classifier.

## Setup and execution

Use Python 3.11 and a separate virtual environment for this project. Run commands from this folder:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python check_inputs.py
python -m jupyterlab
```

On Windows, activate with `.venv\Scripts\activate`. For notebooks, select this environment’s kernel and execute from the top in a fresh kernel. The first code cell exposes `INPUT_PATHS` where old machine-specific paths existed. `PROJECT_DATA_DIR` changes the base for those configured paths; remaining relative paths use the notebook working directory.

## Notebooks

- [nlp_with_disaster_tweets.ipynb](nlp_with_disaster_tweets.ipynb)

## Inputs

Supply these inputs yourself; no dataset is downloaded automatically:

- `train.csv`
- `test.csv`

## Outputs

Model comparisons, TensorBoard logs, saved encoder classifiers, and error analysis.

## Implementation and limitations

Training CSV needs text and target; test CSV needs text. Supply these locally; if using the archive-extraction cell, also provide nlp_getting_started.zip or skip extraction. TensorFlow Hub loads the specified pretrained encoder separately. Its legacy KerasLayer requires the legacy tf-keras runtime; the setup cell selects that runtime before importing TensorFlow. Use a fresh kernel for this notebook. No encoder was downloaded during refactoring.

Reusable model/analysis functions live in `experiment_nlp_with_disaster_tweets.py`. Notebooks retain the experiment sequence, training, and visual narrative.

## Verification

The workspace regression suite checks source/notebook validity and selected behavior with bundled or synthetic inputs. Full experiment training and historical metric reproduction are not implied. See the workspace `REFACTOR_REPORT.md` for exact test coverage and unresolved runtime requirements.
