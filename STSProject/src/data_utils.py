import pandas as pd
from pathlib import Path
from utils import read_file


def load_csv_data(data_folder):
    all_data = pd.read_csv(data_folder / "all_sentences.tsv", sep='\t', index_col=0)
    label_column = all_data.columns[-1]
    return all_data.drop(label_column, axis=1), pd.DataFrame(all_data[label_column])


def load_data(data_folder):

    assert isinstance(data_folder, str) or isinstance(data_folder, Path), f"data_foler must be a string or a Path variable"
    data_folder = Path(data_folder)

    X_train, Y_train = load_csv_data(data_folder / "train")
    X_test, Y_test = load_csv_data(data_folder / "test-gold")

    return X_train, Y_train, X_test, Y_test


def build_csv(data_folder):
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


if __name__ == "__main__":
    subfolders = list(Path('data').glob('*/'))
    for folder in subfolders:
        build_csv(folder)