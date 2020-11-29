import pandas as pd
from pathlib import Path
from utils import read_file


def load_data(data_folder, train_test_split=False):

    assert isinstance(data_folder, str) or isinstance(data_folder, Path), "data_folder must be a string or a Path variable"
    data_folder = Path(data_folder)

    try:
        train_data = load_tsv_data(data_folder / "train", train_test_split=train_test_split)
    except FileNotFoundError:
        print("FileNotFoundError while trying to load: '" + str(data_folder) + "/train/all_sentences.tsv'")
        create_tsv_files()
        train_data = load_tsv_data(data_folder / "train", train_test_split=train_test_split)

    try:
        test_data = load_tsv_data(data_folder / "test-gold", train_test_split=train_test_split)
    except FileNotFoundError:
        print("FileNotFoundError while trying to load: '" + str(data_folder) + "/test-gold/all_sentences.tsv'")
        create_tsv_files()
        test_data = load_tsv_data(data_folder / "test-gold", train_test_split=train_test_split)

    return train_data, test_data


def load_tsv_data(data_folder, train_test_split=False):
    path = data_folder / "all_sentences.tsv"
    all_data = pd.read_csv(path, sep='\t', index_col=0)
    if train_test_split:
        label_column = all_data.columns[-1]
        return all_data.drop(label_column, axis=1), pd.DataFrame(all_data[label_column])
    return all_data


def create_tsv_files():
    print("creating tsv files...")
    subfolders = list(Path('data').glob('*/'))
    for folder in subfolders:
        print("...")
        build_tsv(folder)


def build_tsv(data_folder):
    data_folder = Path(data_folder)
    gs_files = sorted(data_folder.glob('STS.gs.[!ALL]*'))
    sentence_files = sorted(data_folder.glob('STS.input.[!ALL]*'))

    assert len(gs_files) == len(sentence_files), "gs and input files are not of equal length"

    full_dataframe = pd.DataFrame(columns=["S1", "S2", "Gs"])
    for text_file, gs_file in zip(sentence_files, gs_files):
        text_data = read_file(text_file, delimiter='\t')
        gs_data = read_file(gs_file, delimiter='\t')
        gs_data = [line[0] for line in gs_data]
        file_data = pd.DataFrame(text_data, columns=["S1", "S2"])
        file_data["Gs"] = gs_data
        full_dataframe = full_dataframe.append(file_data, ignore_index=True)

    full_dataframe.to_csv(data_folder / "all_sentences.tsv", sep='\t')
    return
