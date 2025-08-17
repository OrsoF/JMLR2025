import sys
from IPython.core import ultratb

sys.excepthook = ultratb.FormattedTB(color_scheme="Linux", call_pdb=False)

import argparse

parser = argparse.ArgumentParser(description="Process experiment file.")
parser.add_argument("--model", help="Experiment file to run", required=True)
args = parser.parse_args()

from utils.experience_class import Experience


experience_parameters = {
    "model": args.model,
    "experience_name": args.model,
    "discounts": [0.8, 0.99],
    "precisions": [1e-2],
    "exact_value_function_precision": 1e-3,
    "measure_repetition": 1,  # Number of experience with same parameters to measure solving time
    "verbose": True,
}

exp = Experience(experience_parameters=experience_parameters)
exp.run()
