import pandas as pd
from IPython.display import display
import numpy as np


def read_data(dataset, sep, names=None):
    df_read = pd.read_csv(dataset, sep=sep, names=names)
    df_read = pd.DataFrame(df_read)
    df_read = df_read.sample(frac=1, random_state=15).reset_index(drop=True)
    return df_read


def print_db(db):
    display(db)


def split_dataset(df_main, start_pos):
    mask_test = int(df_main.shape[0] * (20 / 100))
    test_start = start_pos
    test_end = test_start + mask_test
    test_set_sp = df_main[test_start:test_end]
    train_set_sp = pd.concat([df_main[:test_start], df_main[test_end:]])
    return train_set_sp, test_set_sp


# Press the green button in the gutter to run the script.

def calc_prob(d: pd.DataFrame, lst1, col, pname):
    d["count"] = d.groupby(lst1)[col].transform('count')
    d_len = len(d)
    d[pname] = d["count"] / d_len
    return d


def add_trainset_prob_to_testset(test_df: pd.DataFrame, prob: pd.DataFrame, on_l):
    df_merge = pd.merge(test_df, prob, on=on_l, how='left')
    return df_merge


if __name__ == '__main__':
    df = read_data('Skin_NonSkin.txt', '\t', ['B', 'G', 'R', 'Skin', ])
    split_times = 5
    pos = 0
    for x in range(split_times):
        train_set, test_set = split_dataset(df, pos)
        pos += int(df.shape[0] / split_times)
        skin_db = train_set.loc[train_set['Skin'] == 2]
        skin_db = pd.DataFrame(skin_db).reset_index(drop=True)
        skin_db = calc_prob(skin_db, ['B', 'G'], "R", "skin_prob")

        non_skin_db = train_set.loc[train_set['Skin'] == 1]
        non_skin_db = pd.DataFrame(non_skin_db).reset_index(drop=True)
        non_skin_db = calc_prob(non_skin_db, ['B', 'G'], "R", "non_skin_prob")

        test_set = add_trainset_prob_to_testset(test_set, skin_db[['B', 'G', 'R', 'skin_prob']], ['B', 'G', 'R'])
        test_set = add_trainset_prob_to_testset(test_set, non_skin_db[['B', 'G', 'R', 'non_skin_prob']],
                                                ['B', 'G', 'R'])
        test_set = test_set.fillna(0)
        test_set['result'] = np.where(test_set['skin_prob'] > test_set['non_skin_prob'], 2, 1)

        match_sum = (test_set['result'] == test_set['Skin']).sum()
        test_sum = len(test_set)
        print("test set len = " + str(test_sum))
        print("match set len = " + str(match_sum))

        accuracy = (match_sum * 1.0 / test_sum) * 100
        print("Accuracy of set " + str(x + 1) + " is " + str(accuracy))
