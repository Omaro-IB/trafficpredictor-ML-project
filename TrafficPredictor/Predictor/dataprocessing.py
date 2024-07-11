"""
Process data from .pcapng to .traffic and dictionary files
Also discretize the data so bins are consistent
"""
import Packets as Packets
from os import listdir
from os.path import isfile, join
import pickle
import kmedoids


def longest_cs(str1, str2):
    """
    Calculates the longest common substrings of two strings
    :param str1: str: the first string
    :param str2: str: the second string
    :return: int: length of the longest common substring
    """
    if not str1 or not str2:
        return 0
    m = len(str1)
    n = len(str2)
    dp = [[0] * (n+1) for _ in range(m+1)]

    for i in range(m + 1):
        dp[i][0] = 0
    for j in range(n + 1):
        dp[0][j] = 0

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if str1[i - 1] == str2[j - 1]:
                dp[i][j] = dp[i-1][j - 1] + 1
            else:
                if dp[i - 1][j] >= dp[i][j - 1]:
                    dp[i][j] = dp[i - 1][j]
                else:
                    dp[i][j] = dp[i][j - 1]

    return dp[m][n]


def levenshtein(str_a, str_b, caps=False):
    """
    Calculates similarity of two strings using their Levenshtein distance
    :param str_a: str: the first string
    :param str_b: str: the second string
    :param caps: bool (optional, default = False): caps-sensitive?
    :return: int: the Levenshtein distance
    """
    if not caps:
        return levenshtein(str_a.lower(), str_b.lower(), caps=True)

    m = len(str_a)
    n = len(str_b)
    d = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(1, m + 1):
        d[i][0] = i

    for j in range(1, n + 1):
        d[0][j] = j

    for j in range(1, n + 1):
        for i in range(1, m + 1):
            if str_a[i - 1] == str_b[j - 1]:
                cost = 0
            else:
                cost = 1
            d[i][j] = min(d[i - 1][j] + 1,  # deletion
                          d[i][j - 1] + 1,  # insertion
                          d[i - 1][j - 1] + cost)  # substitution

    return d[m][n]


def k_medoids_group_strings(str_list, name, k=-1, cachedir=None):
    """
    Group a list of similar strings into groups using K-medoids
    :param str_list: list of strings to group together
    :param name: str: name for caching purposes (can be anything)
    :param k: int: (optional, default = 87% * number of strings in str_list): number of clusters to use
    :param cachedir: str (optional, default = None): directory to save/load cached grouping, if None then don't cache
    :return: 2D list. Any strings found to be similar will be in the same inner list.
    """
    if cachedir is not None:
        cache = [f for f in listdir(cachedir) if isfile(join(cachedir, f))]
        if name in cache:
            return Packets.open_backup(cachedir + "\\" + name)
    if k == -1:
        k = int(0.87 * len(str_list))

    # Create distance matrix
    distance_matrix = [[0] * len(str_list) for _ in range(len(str_list))]
    for i in range(len(str_list)):
        for j in range(i + 1, len(str_list)):
            value = (levenshtein(str_list[i], str_list[j]))  # levenshtein metric
            distance_matrix[i][j] = value
            distance_matrix[j][i] = value

    # Use K-Medoids using distance matrix
    km = kmedoids.KMedoids(k, method='fasterpam')
    km.fit(distance_matrix)
    labels = km.labels_

    # Find all labels with multiple members
    lists = {}
    multiple = set()
    for i in range(len(labels)):
        try:
            lists[labels[i]].append(i)
            multiple.add(labels[i])
        except KeyError:
            lists[labels[i]] = [i]

    final_list = []
    for x in multiple:
        final_list.append([str_list[y] for y in lists[x]])
    if cachedir is not None:
        with open(cachedir+"\\"+name, "wb") as file:  # backup
            pickle.dump(final_list, file)
    return final_list


def find_bin(minmax, n, nbins, index=False):
    """
    Given an integer n and the minmax tuple, find which bin n belongs to
    :param minmax: tuple(int, int): the minmax tuple
    :param n: int: the integer
    :param nbins: int: the number of bins to split minmax into
    :param index: bool (optional, default=False): whether to return the index of the bin instead of the actual bin
    :return: str: the bin n belongs to
    """
    # Find equal split
    splitn = [minmax[0]]
    key_range = minmax[1] - minmax[0]
    if key_range % nbins == 0:  # # keys factors perfectly in # bins -> all bins are equally sized
        for i in range(nbins):
            splitn.append((key_range // nbins)+splitn[i])
    else:
        zp = nbins - (key_range % nbins)
        pp = key_range // nbins
        for i in range(nbins):
            if i >= zp:
                splitn.append(pp+splitn[i]+1)
            else:
                splitn.append(pp+splitn[i])
    splitn[-1] += 1

    for s in range(len(splitn) - 1):
        min_key = splitn[s]
        max_key = splitn[s + 1]
        if min_key <= n < max_key:
            if index:
                return s
            else:
                return f"{min_key}-{max_key - 1}"


def discretize_keys(dictionary, nbins, minmax=None):
    """
    Group a dictionary with int keys and int values into a new dictionary with nbins number of bins
    :param dictionary: dict: the original dictionary
    :param nbins: int: the number of bins to group into
    :param minmax: tuple (optional): Specify smallest and largest key to use for split.
                                     Each bin will be equally distributed from smallest to largest key.
                                     If not provided, the min/max of the dictionary keys will be used.
    :return: dict: dictionary with nbins number of keys for each bin. Values are the sum of all values of keys that
                   fall into the corresponding bins.
    """
    if minmax is None:
        minmax = (min(dictionary), max(dictionary))

    # Find equal split
    splitn = [minmax[0]]
    key_range = minmax[1] - minmax[0]
    if key_range <= nbins:  # # keys less than # bins -> return original
        return dictionary
    elif key_range % nbins == 0:  # # keys factors perfectly in # bins -> all bins are equally sized
        for i in range(nbins):
            splitn.append((key_range // nbins)+splitn[i])
    else:
        zp = nbins - (key_range % nbins)
        pp = key_range // nbins
        for i in range(nbins):
            if i >= zp:
                splitn.append(pp+splitn[i]+1)
            else:
                splitn.append(pp+splitn[i])
    splitn[-1] += 1

    # Bin into new bin dictionary
    binned_dict = {}
    # Initialize bins to 0
    for s in range(len(splitn) - 1):
        min_key = splitn[s]
        max_key = splitn[s + 1]
        binned_dict[f"{min_key}-{max_key - 1}"] = 0
    # Find which bin each k fits into
    for k in dictionary:
        for s in range(len(splitn) - 1):
            min_key = splitn[s]
            max_key = splitn[s+1]
            if min_key <= k < max_key:
                binned_dict[f"{min_key}-{max_key - 1}"] += dictionary[k]
                break
    return binned_dict


def group_keys(dictionary, group_list):
    """
    Group a dictionary with str keys and int values into a new dictionary with group_list as the new keys
    :param dictionary: dict: the original dictionary
    :param group_list: list: 2D list. Each inner list contains strings that are to be grouped together
    :return: dict: dictionary with grouped strings together. Values are the sum of all values of keys that
                   are the most similar to each group in group_list
    """
    # Initialize dictionary
    new_dictionary = dictionary.copy()

    # Group into new dictionary
    for group in group_list:
        for key in group[1:]:
            new_dictionary[group[0]] += dictionary[key]
            del new_dictionary[key]

    return new_dictionary


def discretize_group_stats(stats, group_list_dict, minmax_dict, nbins=5):
    """
    Discretize and group statistics by grouping integer dictionaries into nbins number of bins
                                   and grouping string dictionaries into similar words
    :param stats: dict: dictionary to discretize and group
    :param group_list_dict: dict: dictionary of 2D list of groups of words that are similar. The dictionary keys
                                  correspond to the statistic measures and the values are the 2D lists. Each inner list
                                  contains strings that are to be grouped together.
    :return: dict: dictionary with grouped strings together. Values are the sum of all values of keys that
    :param minmax_dict: dict: dictionary of tuple of smallest and largest keys. The dictionary keys correspond to the
                              statistic measures and the values are the min and max values this measure can take on
    :param nbins: int (optional, default = 5): the number of bins to group into
    :return: stats: dict: the discretized and grouped dictionary
    """
    new_stats = {}
    for s in stats:
        t = type(next(iter(stats[s])))
        if t is int or t is float:
            try:
                minmax = minmax_dict[s]
                new_stats[s] = discretize_keys(stats[s], nbins, minmax=minmax)
            except KeyError:
                new_stats[s] = stats[s]
        elif t is str:
            try:
                group_list = group_list_dict[s]
                new_stats[s] = group_keys(stats[s], group_list)
            except KeyError:
                new_stats[s] = stats[s]
        else:
            raise TypeError(f"Type of stats[{s}] keys must be integers or strings")
    return new_stats


def discretize_group_together(all_stat_dicts, int_fields, str_fields, custom_str_groups=None, nbins=5, k=-1, group_list_cachedir=None):
    """
    Discretize and group a list of statistics by grouping integer dictionaries into bins and grouping string
    dictionaries into similar words. Grouping them all together allows for consistent bins and grouping across
    statistic dictionaries.
    :param all_stat_dicts: list: list of dictionaries to discretize and group
    :param int_fields: list: list of keys that correspond to integer values that should be grouped
    :param str_fields: list: list of keys that correspond to string values that should be grouped
    :param custom_str_groups: dict: dictionary of custom string groupings. Keys are keys that should be grouped and
                                   values are 2D lists. Each inner list contains strings that are to be grouped together
    :param nbins: int (optional, default = 5): the number of bins to group into
    :param k: int: (optional, default = 87% * number of strings) number of clusters to use when grouping strings
    :param group_list_cachedir: str (optional, default = None): directory to save/load cached grouping, if None then don't cache
    :return: list: list in the same order as all_stat_dicts but each dictionary is now discretized and grouped together
    """
    # Minmax for each int field values can take on
    minmax_dict = {}
    group_list_dict = {}

    # Initialize minmax and group_list dictionaries
    for field in int_fields:
        minmax_dict[field] = [float("inf"), -float("inf")]
        group_list_dict[field] = set()

    for dic in all_stat_dicts:
        # Update minmax values for all int_fields for all dictionaries
        for int_field in int_fields:
            for val in dic[int_field]:
                if val < minmax_dict[int_field][0]:
                    minmax_dict[int_field][0] = val
                if val > minmax_dict[int_field][1]:
                    minmax_dict[int_field][1] = val
        # Update group_list values for all str_fields for all dictionaries
        for str_field in str_fields:
            str_field_vals = set()
            for val in dic[str_field]:
                str_field_vals.add(val)
            group_list_dict[str_field] = k_medoids_group_strings(sorted(list(str_field_vals)), str_field, k, group_list_cachedir)

    # Add custom string groups to group_list_dict
    if custom_str_groups:
        for custom in custom_str_groups:
            group_list_dict[custom] = custom_str_groups[custom]

    # Discretize and group all dictionaries and return
    all_stat_dicts_grouped_discretized = []  # Final list
    for dic in all_stat_dicts:
        all_stat_dicts_grouped_discretized.append(discretize_group_stats(dic, group_list_dict, minmax_dict, nbins))
    return all_stat_dicts_grouped_discretized, minmax_dict


def dict_list_to_probabilities_vertical(all_stat_dicts):
    """
    Convert a list of statistics dictionaries' values to probabilities. All keys must match.
    Probabilities are normalized vertically (between all other groups)
    :param all_stat_dicts: list: list of dictionaries to convert to probabilities
    :return: list: list in the same order as all_stat_dicts but each dictionary values are now the proportion of
                   each category of the entire list.
    """
    # Initialize totals dictionary
    totals = {field: dict() for field in all_stat_dicts[0]}
    for field in all_stat_dicts[0]:
        totals[field] = {key: 0 for key in all_stat_dicts[0][field]}

    # Add totals
    for dic in all_stat_dicts:
        for field in dic:
            for key in dic[field]:
                try:
                    totals[field][key] += dic[field][key]
                except KeyError:
                    totals[field][key] = 0

    final_list = []
    for dic in all_stat_dicts:
        current_dict = {field: dict() for field in dic}
        for field in dic:
            for key in dic[field]:
                if totals[field][key] == 0:
                    current_dict[field][key] = 0
                else:
                    current_dict[field][key] = dic[field][key] / totals[field][key]
        final_list.append(current_dict)

    return final_list


def dict_list_to_probabilities_horizontal(all_stat_dicts, smoothing=0):
    """
    Convert a list of statistics dictionaries' values to probabilities. All keys must match.
    Probabilities are normalized horizontally (within groups)
    :param all_stat_dicts: list: list of dictionaries to convert to probabilities
    :param smoothing: int (optional, default = 0): laplace smoothing hyperparameter k
    :return: dict: a single dictionary; values are now the proportion of each category within each group.
    """
    # Initialize totals dictionary
    totals = {field: dict() for field in all_stat_dicts[0]}
    for field in all_stat_dicts[0]:
        totals[field] = {key: 0 for key in all_stat_dicts[0][field]}

    field_totals = {field: 0 for field in all_stat_dicts[0]}
    # Add totals
    for dic in all_stat_dicts:
        for field in dic:
            for key in dic[field]:
                try:
                    totals[field][key] += dic[field][key]
                    field_totals[field] += dic[field][key]
                except KeyError:
                    totals[field][key] = 0

    for field in totals:
        for key in totals[field]:
            totals[field][key] = (totals[field][key] + smoothing) / (field_totals[field] + smoothing * len(totals[field]))

    return totals


if __name__ == "__main__":
    preprocess = False
    if preprocess:
        # Preprocessing file example - convert .pcapng to .traffic file and also export statistics dictionary as .pydict
        tname = "netflix1"
        tdir = "streaming\\netflix1"
        Packets.create_traffic_file(tname, tdir + ".pcapng", tdir + ".traffic")
