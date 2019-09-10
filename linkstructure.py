'''
Processing pipeline for article links and metrics
'''
import csv
from collections import defaultdict

import networkx as nx

import utils


def order_metric_by_category(metric_dict, category_dict):
    vals, not_found = [], defaultdict(set)
    for cat, cat_arts in category_dict.items():
        cat_vals = []
        for art in cat_arts:
            norm_art = utils.normalize_article(art)
            if norm_art not in metric_dict:
                cat_vals.append(metric_dict[norm_art])
            else:
                not_found[cat].add(norm_art)
        vals.append(cat_vals)
    return list(category_dict.keys()), vals, not_found


def resolve_redirects(wikilinkgraph_rows, redirect_rows):
    red_dict = {}
    for r in redirect_rows:
        red_dict[utils.normalize_article(r[1])] = utils.normalize_article(r[3])
    # one outgoing, one incoming links??
    # do we HAVE to resolve it?
    # look at results again
    new_content = set()
    for r in wikilinkgraph_rows:
        from_id, to_id = utils.normalize_article(r[1]), utils.normalize_article(r[3])
        resolved_from, resolved_to = red_dict[from_id] if from_id in red_dict else from_id, red_dict[
            to_id] if to_id in red_dict else to_id

        # if points at "same"
        if resolved_from != resolved_to:
            # have to take care of property that redirects are in dataset, remove those.
            new_content.add((resolved_from, resolved_to))
    return sorted(list(new_content), key=lambda x: x[0])


def build_graph_from_wikilinkgraphs(wikilinkgraphs):
    edges = [(from_t, to_t) for from_t, to_t in wikilinkgraphs]
    G = nx.DiGraph()
    G.add_edges_from(edges)
    return G


if __name__ == '__main__':
    # Optional postprocessing: resolve Wikipedia redirects and remove redirects (nodes with single outgoing link)
    # could also skip this and only feed wikilinkgraphs to
    wikilinkgraphs = resolve_redirects(
        utils.read_csv('<Wikipedia redirects>.csv', 'data', escape=csv.QUOTE_ALL),
        utils.read_csv('<WikiLinksGraph>.csv', 'data', delimiter='\t'))
    G = build_graph_from_wikilinkgraphs(wikilinkgraphs)
    # Specify which articles belong to which category - depends on current task
    articles_per_category = None  # change this variable accordingly
    # calculate example metric
    out_degree_dict = {utils.normalize_article(k): deg for k, deg in G.out_degree().items()}
    # use helper to pick apart article results per category
    categories, outdegree_per_cat, not_found = order_metric_by_category(out_degree_dict, articles_per_category)
