# JMLR2025
This article contains the code used to produce the experimental results of the article **Disaggregation Techniques for Faster Markov Decision Process Solving and State Abstraction Discovery**.

### Requirements

- Python 3.8+
- Install dependencies with:
  ```sh
  pip install -r requirements.txt
  ```
- For the notebook:
  ```
  pip install notebook
  ```

### Usage

To run experiments and reproduce results, use the main script:

```sh
python JMLR.py
```

The original experiment parameters (model, solver, etc.) are set directly in [JMLR.py](JMLR.py).

To run the notebook for plotting abstractions:

```sh
jupyter notebook plot_abstractions.ipynb
```

### Models

The `models/` directory contains various MDP environments, such as:
- [`barto_total.py`](models/barto_total.py)
- [`rooms.py`](models/rooms.py)
- [`rooms_total.py`](models/rooms_total.py)
- [`taxi.py`](models/taxi.py)
- and others.

### Results

- Results are shown directly in the prompt.
- Figures and additional outputs may be generated in the `figures/` directory (created automatically).

### Project Structure

- `JMLR.py`: Main script to run experiments.
- `models/`: MDP environment definitions.
- `solvers/`: MDP solvers and algorithms.
- `utils/`: Utility functions for data management and processing.
- `saved_models/`: Stores trained models and experiment outputs.

### Resetting the Environment

To clean all generated files and results, use the reset function in [`utils/data_management.py`](utils/data_management.py):

```python reset.py
```