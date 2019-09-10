'''
Classify articles and compute metrics
'''
import numpy as np
from sklearn.ensemble import RandomForestClassifier

import utils


def train_random_forest(features, labels, n_estimators=750, max_depth=25, max_features=0.5, n_jobs=1):
    return RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, max_features=max_features,
                                  n_jobs=n_jobs).fit(features, labels)


def get_sorted_revisions(rev_and_features):
    return list(sorted(rev_and_features, key=lambda x: x[0]))


def label_article_folder(classifier, article_files, folder='', path='data/articles/'):
    full_path = path + folder
    labeled_articles = {}
    for f in article_files:
        labeled_articles[f] = label_article(classifier, f, full_path)
    return labeled_articles


def label_article(classifier, filename, path='data/articles/', padding_len=3):
    rev_ids, features = zip(*get_sorted_revisions((utils.load_from_file_pickle(filename, path))))
    faulty = []
    for i in range(len(features)):
        if not isinstance(features[i], list):
            faulty.append(i)

    filtered_feats = [features[i] for i in range(len(features)) if i not in faulty]
    data_y = classifier.predict(filtered_feats)

    if padding_len > 0:
        for f in faulty:
            data_y = np.insert(data_y, f, np.array([0.0] * padding_len), 0)
    return list(zip(rev_ids, data_y))


if __name__ == '__main__':
    features, labels = None, None
    random_forest = train_random_forest(features, labels)

    path = 'data/articles'
    folders = ['a', 'b', 'c', 'featured', 'good', 'cw']
    category_dict = {}
    for cat_folder in folders:
        articles = utils.get_article_files_pickle(path + cat_folder)  # this could be any article list you like
        labeled_articles = label_article_folder(random_forest, articles, cat_folder, path)
        category_dict[cat_folder] = labeled_articles
