from itertools import chain, combinations
from collections import defaultdict
from time import time
import pandas as pd


def run(file_name, map_name, min_support, min_conf):
    t = time()
    print '=' * 50, file_name
    print 'min_support:', min_support, 'min_conf:', min_conf
    print 'Running ...'
    input_file = read_data(file_name)
    items, rules = start_algo(input_file, min_support, min_conf)
    name_map = read_name_map(map_name)
    print_items_rules(items, rules, ignore_one_item_set=True, name_map=name_map)
    print '\n', time() - t, 'sec\n\n'


def start_algo(infile, min_support, min_conf):
    """
    Run the Apriori algorithm. infile is a record iterator.
    Return:
        rtn_items: list of (set, support)
        rtn_rules: list of ((preset, postset), confidence)
    """
    one_cand_set, all_transactions = gen_one_item_cand_set(infile)
    N = len(all_transactions)

    set_count_map = defaultdict(int)  # maintains the count for each set

    one_freq_set, set_count_map = get_items_with_min_support(
        one_cand_set, all_transactions, min_support, set_count_map, N)

    freq_map, set_count_map = run_apriori_loops(
        one_freq_set, set_count_map, all_transactions, min_support, N)

    rtn_items = get_frequent_items(set_count_map, freq_map, N)
    rtn_rules = get_frequent_rules(set_count_map, freq_map, min_conf, N)

    return rtn_items, rtn_rules


def gen_one_item_cand_set(input_fileator):
    """
    Generate the 1-item candidate sets and a list of all the transactions.
    """
    all_transactions = list()
    one_cand_set = set()
    for record in input_fileator:
        transaction = frozenset(record)
        all_transactions.append(transaction)
        for item in transaction:
            one_cand_set.add(frozenset([item]))  # Generate 1-item_sets
    return one_cand_set, all_transactions


def get_items_with_min_support(item_set, all_transactions, min_support,
                               set_count_map, N):
    """
    item_set is a set of candidate sets.
    Return a subset of the item_set
    whose elements satisfy the minimum support.
    Update set_count_map.
    """
    rtn = set()
    local_set = defaultdict(int)

    for item in item_set:
        for transaction in all_transactions:
            if item.issubset(transaction):
                set_count_map[item] += 1
                local_set[item] += 1

    for item, count in local_set.items():
        support = float(count)/N
        if support>=min_support:
            rtn.add(item)
        #################### TODO
        #################### TODO

    return rtn, set_count_map


def run_apriori_loops(l_set, set_count_map, all_transactions,
                      min_support, N):
    """
    Return:
        freq_map: a dict
            {<length_of_set_l>: <set_of_frequent_itemsets_of_length_l>}
        set_count_map: updated set_count_map
    """
    freq_map = dict()
    i = 1
    while (l_set!= set([])):#################### TODO:
        freq_map[i] = l_set
        c_set = join_set(l_set, i + 1)
        l_set, set_count_map = get_items_with_min_support(
            c_set, all_transactions, min_support, set_count_map, N)
        i += 1
    return freq_map, set_count_map


def get_frequent_items(set_count_map, freq_map, N):
    """ Return frequent items as a list. """
    rtn_items = []
    for key, value in freq_map.items():
        rtn_items.extend(
            [(tuple(item), get_support(set_count_map, item, N))
             for item in value])
    return rtn_items


def get_frequent_rules(set_count_map, freq_map, min_conf, N):
    """ Return frequent rules as a list. """
    rtn_rules = []
    for key, value in freq_map.items()[1:]:
        for item in value:
            _subsets = map(frozenset, [x for x in subsets(item)])
            for element in _subsets:
                remain = item.difference(element)
                if len(remain) > 0:
                    union = item.union(element)
                    confidence = (get_support(set_count_map, union, N))/get_support(set_count_map, element, N)
                    if confidence >= min_conf:
                        rtn_rules.append(
                            ((tuple(element), tuple(remain)), confidence))
    return rtn_rules


def get_support(set_count_map, item, N):
    """ Return the support of an item. """
    return float(set_count_map[item])/N


def join_set(s, l):
    """
    Self-joining.
    Return a set whose elements are unions of sets in s
        whose length is equal to the parameter l.
    """
    rtn = set()
    for i in s:
        for j in s:
            if len(i.union(j)) == l:
                rtn.add(i.union(j))
    return rtn # TODO


def subsets(x):
    """ Return non-empty subsets of x. """
    return chain(*[combinations(x, i + 1) for i, a in enumerate(x)])


def print_items_rules(items, rules, ignore_one_item_set=False, name_map=None):
    print '\n------------------------ FREQUENT PATTERNS'
    cnt = 0
    for itemset, support in sorted(items, key=lambda (item, support): support):
        if len(itemset) == 1 and ignore_one_item_set:
            continue
        print 'Itemset: {} , {:.2%}'.format(
            convert_item_to_name(itemset, name_map), support)
        cnt += 1
    print('----> %s printed' % cnt)
    print '\n------------------------ RULES:'
    cnt = 0
    for rule, confidence in sorted(
            rules, key=lambda (rule, confidence): confidence):
        pre, post = rule
        print 'Rule: %s ==> %s , %.3f' % (
            convert_item_to_name(pre, name_map),
            convert_item_to_name(post, name_map),
            confidence)
        cnt += 1
    print('----> %s printed' % cnt)


def convert_item_to_name(item, name_map):
    """ Return the string representation of the item. """
    if name_map:
        return ', '.join(['"{}"'.format(name_map[int(id)]) for id in item])
    else:
        return ', '.join(['"{}"'.format(x) for x in item])


def read_data(fname):
    """ Read from the file and yield a generator. """
    file_iter = open(fname, 'rU')
    for line in file_iter:
        line = line.strip().rstrip(',')
        record = frozenset(line.split(','))
        yield record


def read_name_map(fname):
    if not fname:
        return {}
    """ Read from the file and return a dict mapping ids to names. """
    df = pd.read_csv(fname)
    return df.set_index('id')['movie'].to_dict()


if __name__ == '__main__':
    run('toy.txt', '', 0.6, 0.8)
    run('user_movies.txt', 'id_movie.csv', 0.26, 0.68)
    run('movie_tags.txt', '', 0.0028, 0.6)
