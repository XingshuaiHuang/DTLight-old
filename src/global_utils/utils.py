import json
import numpy as np
import random
import torch
import os


def insert_data_to_json(file_path, new_data, long_list=False):
    try:
        # Try to read existing JSON data from the file
        with open(file_path, "r") as file:
            existing_data = json.load(file)
    except (FileNotFoundError, json.JSONDecodeError):
        # If the file does not exist or is empty, initialize with an empty dictionary
        existing_data = {}

    # Merge new data into the existing JSON object
    existing_data.update(new_data)

    # Write the updated JSON data back to the file
    if not long_list:
        with open(file_path, "w") as file:
            json.dump(existing_data, file, cls=NpEncoder, indent=4)
    else:
        json_str = json.dumps(existing_data, ensure_ascii=False, indent=4, separators=(",", ": "))
        json_str = json_str.replace('[\n        ', '[')
        json_str = json_str.replace('\n        ', ' ')
        json_str = json_str.replace('\n    ]', ']')
        with open(file_path, "w") as file:
            file.write(json_str)


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


def set_seed_everywhere(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    