# Computing Coverage-Based Abductive Explanations for Tree-Based Models

## Supplementary Material

## Contents

This archive contains the following resources:

- **`proofs.pdf`**: Detailed proofs for all propositions presented in the paper.

- **`sources`**: The source code is built on the basis of a GIT clone of <https://github.com/crillab/pyxai>.

  **1. Implementation of Our Approach:**

  The core logic of our Coverage-Based Prime Implicant Explanations for Tree-Based Models is implemented in the following files:

  - For Decision Trees: `sources/pyxai/pyxai/sources/core/explainer/explainerDT.py`
  - For Random Forests: `sources/pyxai/pyxai/sources/core/explainer/explainerRF.py`
  - For Boosted Trees:
    - Explainer: `sources/pyxai/pyxai/sources/core/explainer/explainerBT.py`
    - Implicant test (MIP-based): `sources/pyxai/pyxai/sources/solvers/ORTOOLS/IsImplicantBT.py`

  **2. Experimental Scripts (Comparison):**

  Located in `sources/pyxai/pyxai/examples/RF/`, these scripts run the experiments comparing our approach against the state-of-the-art algorithm (Martin Cooper and Leila Amgoud):

  - `Coverage-Based_Abductive_Explanations_for_DT.py`: Main script to run the comparison for Decision Trees.
  - `Coverage-Based_Abductive_Explanations_for_RF.py`: Main script to run the comparison for Random Forests.
  - `Coverage-Based_Abductive_Explanations_for_BT.py`: Main script to run the comparison for Boosted Trees.
  - `cpi_explaine_DT.py`, `cpi_explaine_RF.py` and `cpi_explaine_BT.py`: Modules containing the implementation of the baseline algorithm (Cooper & Amgoud), imported by the main scripts.
  - `run_all_coverage_DT.py`, `run_all_coverage_RF.py` and `run_all_coverage_BT.py`: Scripts to batch execute the experiments on all datasets.

- **`dataset`**: The datasets converted and used in our experiments.

  More specifically, for each dataset, you can find:

  - `datasets/<dataset>.csv`: The converted dataset.
  - `datasets/<dataset>.types`: A JSON file containing all the information about the features.

- **`logs`**: The outputs produced by the algorithms run in the experiments.

  - `logs/cpi_xp_log_dt/`: Results obtained for Decision Trees experiments.
  - `logs/cpi_xp_log_rf/`: Results obtained for Random Forests experiments.
  - `logs/cpi_xp_log_bt/`: Results obtained for Boosted Trees experiments.
  - `logs/gen_latex_tables.py`: Python script to process JSON logs and automatically generate the LaTeX performance tables (Tables 2, 3, and 4) for the DT, RF, and BT models.

  Execution commands (requires the path to the logs directory):

  For Decision Trees logs:
  ```
  python3 logs/gen_latex_tables.py --path=logs/cpi_xp_log_dt/
  ```

  For Random Forests logs:
  ```
  python3 logs/gen_latex_tables.py --path=logs/cpi_xp_log_rf/
  ```

  For Boosted Trees logs:
  ```
  python3 logs/gen_latex_tables.py --path=logs/cpi_xp_log_bt/
  ```

- **`directory pyxai`**: A modified version of pyxai used to compute explanations.

---

## Setup

- Ensure you are using a Linux OS and Python version ≥ 3.12.7
- Install Pyxai. Follow these [instructions](https://www.cril.univ-artois.fr/pyxai/documentation/installation/github/).  
  Instead of cloning the software, please use the source provided in this archive.
- Install the required dependencies:
  ```
  python3 -m pip install numpy==2.0.2
  python3 -m pip install pandas==2.2.3
  python3 -m pip install scikit-learn==1.5.2
  python3 -m pip install xgboost==1.7.3
  ```
- To compile the modified version of pyxai in the pyxai directory:
  ```
  python3 -m pip install -e .
  ```

---

## How to Run the Experiments

For a given dataset, the specific scripts `Coverage-Based_Abductive_Explanations_for_DT.py`, `Coverage-Based_Abductive_Explanations_for_RF.py` and `Coverage-Based_Abductive_Explanations_for_BT.py` located in `sources/pyxai/pyxai/examples/RF/` execute the comparison between our approach and the baseline.

### Running the comparison (Our Approach vs. Cooper & Amgoud)

These scripts execute both algorithms and log the results.

For Decision Trees (DT):
```
python3 sources/pyxai/pyxai/examples/RF/Coverage-Based_Abductive_Explanations_for_DT.py -dataset="../../../../../datasets/breast-tumor_0"
```

For Random Forests (RF):
```
python3 sources/pyxai/pyxai/examples/RF/Coverage-Based_Abductive_Explanations_for_RF.py -dataset="../../../../../datasets/breast-tumor_0"
```

For Boosted Trees (BT):
```
python3 sources/pyxai/pyxai/examples/RF/Coverage-Based_Abductive_Explanations_for_BT.py -dataset="../../../../../datasets/breast-tumor_0"
```

> **Note:** Only the dataset name (with the directory) needs to be specified in the `-dataset` parameter.

### Running the comparison on all datasets (Batch Execution)

For Decision Trees:
```
python3 sources/pyxai/pyxai/examples/RF/run_all_coverage_DT.py
```

For Random Forests:
```
python3 sources/pyxai/pyxai/examples/RF/run_all_coverage_RF.py
```

For Boosted Trees:
```
python3 sources/pyxai/pyxai/examples/RF/run_all_coverage_BT.py
```