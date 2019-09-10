'''
Classify articles and compute metrics
'''
import itertools
import random as rd
from collections import Counter, defaultdict

import numpy as np
from nltk import ngrams
from sklearn.ensemble import RandomForestClassifier

import utils

NO_LABEL_PAD = 'NoLabel'
SUPER_LABELS = ['Content', 'Format', 'WikiContext']
KEY_INT_COUNTS, KEY_INT_COUNTS_RATIO, KEY_REV_COUNT, KEY_ARTICLE_COUNT = 'counts', 'countsRatio', 'revs', 'articles'
KEY_MACRO_R, KEY_MICRO_R, KEY_TOTAL, KEY_REVSPERART = 'macroR', 'microR', 'overall', 'revsPerArt'


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


def shuffle_and_get_n_articles(articles, n):
    if n < 0:
        n = len(articles.keys())
    to_shuffle = list(articles.items())
    rd.shuffle(to_shuffle)
    return to_shuffle[:n]


def calculate_interaction_scores(labeled_data, labels, matrix_indices=None, dec=8):
    n_gram_labels = list(ngrams(labeled_data, n=2, pad_left=True, pad_right=True, pad_symbol=NO_LABEL_PAD))
    count_label_combinations = Counter(n_gram_labels)
    count_label_source = Counter([source for source, dest in n_gram_labels])
    scores = np.zeros((len(labels), len(labels)))
    if matrix_indices is None:
        for index, l in enumerate(labels):
            matrix_indices[l] = index

    for transmission, count in count_label_combinations.most_common():
        src, dest = transmission[0], transmission[1]
        if src in labels and dest in labels:
            scores[matrix_indices[src], matrix_indices[dest]] = (round(count / count_label_source[src], dec)) \
                if count / count_label_source[src] > 10 ** -dec else 10 ** -dec
    return scores


def binary_vector_to_match(y, label_names, no_label_pad=NO_LABEL_PAD):
    labeled_data = []
    for data in y:
        if all(d == 0 for d in data):
            labeled_data.append([no_label_pad])
        else:
            labeled_data.append(sorted([label_names[l_i] for l_i in range(len(label_names)) if data[l_i] == 1]))
    return labeled_data


def binary_vector_to_string(y, label_names, separator=', '):
    return [separator.join(sorted(l)) for l in binary_vector_to_match(y, label_names)]


def get_article_string_labels(article_revisions, labels):
    # make sure revisions are sorted accordingly before they are converted
    return binary_vector_to_string([d[1] for d in get_sorted_revisions(article_revisions)], labels)


def get_label_combinations(label_names, separator=', ', include_nolabel=True):
    if include_nolabel:
        combinations, comb_indices = [NO_LABEL_PAD], {NO_LABEL_PAD: 0}
    else:
        combinations, comb_indices = [], {}

    for i in range(1, len(label_names) + 1):
        for cont in itertools.combinations(label_names, i):
            combinations.append(separator.join(cont))
            comb_indices[separator.join(cont)] = len(combinations) - 1
    return combinations, comb_indices


def relative_frequency_per_category(articles, n):
    # iterate article_data stored in items: article -> interactions
    art_int_count = {}  # article interaction count
    considered_arts = shuffle_and_get_n_articles(articles, n)
    # first calc article results
    for article, interaction_data in considered_arts:
        interaction_count, interact_count_ratio, rev_count = defaultdict(int), defaultdict(int), len(interaction_data)
        interaction_count.update(Counter(binary_vector_to_string([d[1] for d in interaction_data], SUPER_LABELS)))
        interact_count_ratio.update({k: v / rev_count for k, v in interaction_count.items()})
        art_int_count[article] = {KEY_INT_COUNTS: interaction_count, KEY_INT_COUNTS_RATIO: interact_count_ratio,
                                  KEY_REV_COUNT: rev_count}
    # now calc category results
    cat_int, art_results = {}, art_int_count.values()
    cat_int[KEY_REV_COUNT] = sum(single_art[KEY_REV_COUNT] for single_art in art_results)
    cat_int[KEY_INT_COUNTS], cat_int[KEY_MACRO_R], cat_int[KEY_MICRO_R] = {}, {}, {}
    cat_int[KEY_ARTICLE_COUNT] = len(considered_arts)
    cat_int[KEY_REVSPERART] = cat_int[KEY_REV_COUNT] / cat_int[KEY_ARTICLE_COUNT]

    for label in get_label_combinations(SUPER_LABELS)[0]:
        cat_int[KEY_MACRO_R][label] = \
            sum(art_res[KEY_INT_COUNTS_RATIO][label] for art_res in art_results) / len(art_results)
        cat_int[KEY_INT_COUNTS][label] = sum(art_res[KEY_INT_COUNTS][label] for art_res in art_results)
        cat_int[KEY_MICRO_R][label] = cat_int[KEY_INT_COUNTS][label] / cat_int[KEY_REV_COUNT]
    return art_int_count, cat_int


def calculate_transition_probabilities(category_data, labels, matrix_indices=None, n=800, dec=8):
    category_scores = {}
    cat_article_scores = {}
    for category, articles in category_data.items():
        article_scores = {}
        article_data = shuffle_and_get_n_articles(articles, n)
        for article, data_y in article_data:
            string_labels = get_article_string_labels(data_y, labels)
            article_scores[article] = calculate_interaction_scores(string_labels, labels, matrix_indices, dec=dec)
        cat_article_scores[category] = article_scores
        category_scores[category] = np.mean(list(article_scores.values()), 0)
    return cat_article_scores, category_scores


def calculate_relative_frequencies(category_data, n=800):
    cat_int, art_cat_int = {}, {}
    for category, articles in category_data.items():
        print('Analyze %s' % category)
        art_cat_int[category], cat_int[category] = relative_frequency_per_category(articles, n)
    return art_cat_int, cat_int


if __name__ == '__main__':
    super_labels = ['Content', 'Format', 'WikiContext']
    features, y_labels = None, None
    random_forest = train_random_forest(features, y_labels)
    path = 'data/articles'
    folders = ['a', 'b', 'c', 'featured', 'good', 'cw']
    category_dict = {}
    for cat_folder in folders:
        # this could be any article list you like
        articles = utils.get_article_files_pickle(utils.get_path(path, cat_folder))
        labeled_articles = label_article_folder(random_forest, articles, cat_folder, path)
        category_dict[cat_folder] = labeled_articles

    calculate_transition_probabilities(category_dict, super_labels)
    calculate_relative_frequencies(category_dict)
