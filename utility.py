import math

"""
calculate the entropy of entire dataset """
# --------------------------------------- #

def base_entropy(dataset):
    target_col = dataset.iloc[:, -1]
    choices = list(set(target_col))

    p = 0; n = 0

    for x in target_col:
        if x == choices[0]:
            p = p + 1
        else:
            n = n + 1

    if p == 0 or n == 0:
        return 0
    elif p == n:
        return 1
    else:
        return - ( (p / (p + n)) * (math.log2(p / (p + n))) ) \
            - ( (n / (p + n)) * (math.log2(n / (p + n))) )

"""
calculate the entropy of attributes """
# ----------------------------------- #

def entropy(dataset, feature, attribute):
    target_col = dataset.iloc[:, -1]
    choices = list(set(target_col))

    p = 0; n = 0

    for x, y in zip(feature, target_col):
        if x == attribute:
            if y == choices[0]:
                p = p + 1
            elif y == choices[1]:
                n = n + 1

    if p == 0 or n == 0:
        return 0
    elif p == n:
        return 1
    else:
        return - ( (p / (p + n)) * (math.log2(p / (p + n))) ) \
            - ( (n / (p + n)) * (math.log2(n / (p + n))) )

"""
calculate the Information Gain """
# ------------------------------ #

def information_gain(dataset, feature):
    distinct_feature_values = list(set(feature))
    sigma = 0

    for x in distinct_feature_values:
        sigma = sigma + (feature.count(x) / len(feature)) * entropy(dataset, feature, x)

    info_gain = base_entropy(dataset) - sigma
    print("IG", feature, ": %.5f" % (info_gain))
    return info_gain

"""
calculate the maximum Information Gain """
# -------------------------------------- #

def max_information_gain_attr_index(dataset):
    max_ig = -1; attr_index = 0
    size = len(dataset.columns) - 1

    for x in range(size):
        feature = list(dataset.iloc[:, x])
        my_ig = information_gain(dataset, feature)

        if my_ig > max_ig:
            max_ig = my_ig
            attr_index = x

    return attr_index, max_ig

"""
check the purity and impurity of a child """
# ---------------------------------------- #

def counter(target_col, attr, i):
    p = 0; n = 0
    choices = list(set(target_col))

    for x, y in zip(target_col, attr):
        if y == i:
            if x == choices[0]:
                p = p + 1
            elif x == choices[1]:
                n = n + 1
    
    return p, n

"""
get the child(s) of an attribute """
# -------------------------------- #

def get_childs(dataset, attr_index):
    distinct_attr_values = list(set(dataset.iloc[:, attr_index]))
    childs = {}

    for x in distinct_attr_values:
        childs[x] = counter(dataset.iloc[:, -1], dataset.iloc[:, attr_index], x)

    return childs

"""
reduce the dataset in accordance with the impurity """
# -------------------------------------------------- #

def reduce_dataset(dataset, attr_index, feature, impure_child):
    sub_dataset = dataset[dataset[feature] == impure_child]
    del (sub_dataset[sub_dataset.columns[attr_index]])
    return sub_dataset
