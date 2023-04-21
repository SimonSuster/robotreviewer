import csv
import json
import os
import base64

def rand_id():
    return base64.urlsafe_b64encode(os.urandom(16))[:21].decode('utf-8')

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


def read_csv(f, keys=None):
    with open(f) as csvfile:
        if keys is not None:
            reader = csv.DictReader(csvfile, fieldnames=keys)
        else:
            reader = csv.DictReader(csvfile)
        for row in reader:
            yield row


def save_json(obj, filename):
    with open(filename, "w") as out:
        json.dump(obj, out, separators=(',', ':'), indent=2, sort_keys=True)


def load_json(filename):
    with open(filename) as in_f:
        return json.load(in_f)

