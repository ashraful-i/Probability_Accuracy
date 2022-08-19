# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import pandas as pd
from IPython.display import display


def read_data(dataset, sep, names=None):
    df = pd.read_csv(dataset, sep=sep, names=names)
    df = pd.DataFrame(df)
    df = df.sample(frac=1, random_state=123).reset_index(drop=True)
    return df


def print_db(db):
    display(db)


def split_datasetx(df, x, total_split):
    mask = int(df.shape[0] * (100 / total_split) / 100)
    test_start = x * mask
    test_end = test_start + mask
    test_set = df[test_start:test_end]
    train_set = pd.concat([df[:test_start], df[test_end:]])
    # print(test_set,'\n', train_set)
    return train_set, test_set


def split_dataset(df, x, total_split, pos):
    mask_train = int(df.shape[0] * (80 / 100))
    mask_test = int(df.shape[0] * (20 / 100))
    test_start = pos
    test_end = test_start + mask_test
    # print(test_start, test_end)
    test_set = df[test_start:test_end]
    train_set = pd.concat([df[:test_start], df[test_end:]])
    return train_set, test_set


# Press the green button in the gutter to run the script.

def calc_prob():
    d = pd.DataFrame()
    d['A'] = [1, 1, 2, 3, 4, 5, 6, 4]
    d['B'] = [1, 1, 3, 4, 5, 6, 7, 3]
    d['C'] = [1, 1, 4, 5, 6, 7, 8, 4]
    d['Skin'] = [1, 1, 1, 1, 1, 1, 1, 2]
    skin_db = d.loc[d['Skin'] == 1]
    skin_db = pd.DataFrame(skin_db).reset_index(drop=True)
    # d['Total Occ'] = d.groupby(['A','B','C']).transform('count')

    # group = d.groupby(['A','B','C'])
    # group = d.groupby(['A','B','C']).size().to_frame('size')

    skin_db["count"] = skin_db.groupby(['A', 'B'])["C"].transform('count')
    skin_len = len(skin_db)

    skin_db["prob_skin"] = skin_db["count"]/skin_len
    print(skin_db)

    non_skin_db = d.loc[d['Skin'] == 2]
    non_skin_db = pd.DataFrame(non_skin_db).reset_index(drop=True)

    non_skin_db["count"] = non_skin_db.groupby(['A', 'B'])["C"].transform('count')
    non_skin_len = len(non_skin_db)

    non_skin_db["prob_nonskin"] = non_skin_db["count"] / non_skin_len
    print(non_skin_db)

    # d2 = d.merge(d2,left_on=(['A','B','C']), right_index=True)
    # print(d2)


if __name__ == '__main__':
    calc_prob()
    '''df = read_data('Skin_NonSkin.txt', '\t', ['B', 'G', 'R', 'Skin', ])
    #print_db(df.shape[0])
    split_times = 5
    pos = 0
    for x in range(split_times):
        train_set, test_set = split_dataset(df, x, split_times, pos)
        pos += int(df.shape[0] / split_times)
        calc_prob(train_set)'''

    # print(train_set.loc[df['Skin'] == 1].shape[0])
    # skin_db = train_set.loc[df['Skin'] == 1]

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
