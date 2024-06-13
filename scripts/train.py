#!/opt/venv/bin/python

# script args
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-f", "--file", help="Path config file", type=str)
parser.add_argument("-pb", "--progress_bar", help="Use  progress bar for training", type=bool)
args = parser.parse_args()

# imports
import rl4fisheries
from rl4fisheries.utils import sb3_train, sb3_train_save_checkpoints

# hf login
from huggingface_hub import hf_hub_download, HfApi, login
# login()

import os

# transform to absolute file path
abs_filepath = os.path.abspath(args.file)

# change directory to script's directory (since io uses relative paths)
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

# train
save_id, options = sb3_train_save_checkpoints(abs_filepath)
fname = os.path.basename(save_id)

# hf upload
api = HfApi()
try:
    api.upload_file(
        path_or_fileobj=save_id,
        path_in_repo="sb3/rl4fisheries/"+fname,
        repo_id="boettiger-lab/rl4eco",
        repo_type="model",
    )
except Exception as ex:
    print("Couldn't upload to hf :(.")
    print(ex)

print(f"""
Finished training on input file {args.file}.

""")
