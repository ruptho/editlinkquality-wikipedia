import csv
import os
import pickle

BASE_DIR = os.path.dirname(os.path.realpath(__file__))


def save_to_file_pickle(object_to_save, filename, path=""):
    pickle.dump(object_to_save, open(get_path(path, filename, 'pkl'), 'wb'), pickle.HIGHEST_PROTOCOL)


def load_from_file_pickle(filename, path):
    load_path = path + filename
    if os.path.exists(load_path):
        with open(load_path, 'rb') as f:
            return pickle.load(f)
    else:
        return None


def get_path(path, filename, ext, base_dir=BASE_DIR):
    return "{}/{}/{}.{}".format(base_dir, path, filename, ext)


def normalize_article(article):
    return article.replace(' ', '_')


def read_csv(filename, path, delimiter=',', escape=csv.QUOTE_MINIMAL):
    with open(get_path(path, filename, 'csv'), 'r', encoding='utf8') as csvfile:
        rows = list(csv.reader(csvfile, delimiter=delimiter, quoting=escape))
    return rows
