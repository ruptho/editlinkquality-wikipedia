from collections import defaultdict

import numpy as np


def load_counts_from_pkl(filename='cat_interaction_vals', path='data/articles'):
    cat_data = file_utils.load_from_file_pickle(filename, path)
    classes, interaction_counts, article_counts, revs_per_article = [], [], [], []
    for cat, values in cat_data.items():
        classes.append(cat)
        interaction_counts.append(list(values['counts'].values())[:-3])
        article_counts.append(values['articles'])
        article_counts.append(values['revsPerArt'])
    return classes, interaction_counts, article_counts, revs_per_article


def permutation_test_frequency(rf_articles, labels):
    article_values = defaultdict(list)
    for classes, articles in rf_articles.items():
        for article in articles.values():
            article_values[classes].append(np.asarray([article['countsRatio'][l] for l in labels]))
    return permutation_test(article_values)


def permutation_test(article_class_values):
    classes, articles = zip(*article_class_values.items())
    p_values = {}
    for c1 in range(0, len(classes) - 1):
        for c2 in range(c1 + 1, len(classes)):
            ratios1, ratios2, class1, class2 = articles[c1], articles[c2], classes[c1], classes[c2]
            p_array = permutation_significance(np.asarray(ratios1), np.asarray(ratios2), two_sided=False)
            p_values[(class1, class2)] = p_array
    return p_values


def permutation_significance(ratios1, ratios2, draws=10000, stringent=True, two_sided=True):
    count1, n_larger, ratios, diff = \
        len(ratios1), np.zeros(ratios1.shape[1:]), np.concatenate([ratios1, ratios2]), np.abs(
            np.mean(ratios1, axis=0) - np.mean(ratios2, axis=0))

    for i in range(draws):
        np.random.shuffle(ratios)
        delta = np.around(np.abs(np.mean(ratios[:count1], axis=0) - np.mean(ratios[count1:], axis=0)), 5)
        # stringent (weak inequality) vs non-stringent (strict inequality)
        delta_diff_comparison = np.less_equal(diff, delta) if stringent else np.less(diff, delta)
        n_larger = np.add(n_larger, delta_diff_comparison)
    return (n_larger / draws) * (2 if two_sided else 1)
