import pytest

from sheepy.common.df_ops import (
    SOURCE_FILE,
    read_csv,
    read_csv_text_classifier,
    resample_multilabel_positives,
    resample_positives,
)


def test_read_csv_text_classifier():
    path = "tests/resources/dummy_dataset.csv"
    text_col = "text"
    label_col = "label"

    with pytest.raises(ValueError, match=r".*not a valid text column.*"):
        read_csv_text_classifier(path, text_col="invalid", label_cols=label_col)

    with pytest.raises(ValueError, match=r".*not a valid label column.*"):
        read_csv_text_classifier(path, text_col=text_col, label_cols="invalid")

    df_evaluate = read_csv_text_classifier(
        path, text_col=text_col, label_cols="invalid", evaluate=True
    )
    assert SOURCE_FILE in df_evaluate.columns

    df_train = read_csv_text_classifier(path, text_col=text_col, label_cols=label_col)
    assert text_col in df_train.columns
    assert label_col in df_train.columns
    assert SOURCE_FILE not in df_train.columns


def test_read_csv():
    path = "tests/resources/dummy_dataset.csv"
    text_col = "text"
    label_col = "label"

    df = read_csv(path)
    assert text_col in df.columns
    assert SOURCE_FILE in df.columns
    assert label_col in df.columns
    assert df.shape == (4, 6)


def test_read_csv_with_filter():
    path = "tests/resources/dummy_dataset.csv"
    text_col = "text"
    label_col = "label"

    column_list = ["text", "label"]

    df = read_csv(path, column_list)
    assert text_col in df.columns
    assert SOURCE_FILE in df.columns
    assert label_col in df.columns
    assert "group_id" not in df.columns
    assert df.shape == (4, 3)


def test_resample_positives():
    path = "tests/resources/dummy_dataset.csv"
    text_col = "text"
    label_col = "label"
    pos_label = "1"

    df = read_csv_text_classifier(path, text_col=text_col, label_cols=label_col)

    # Verify things are unmodified if resample rate is 1
    resample_rate = 1
    resampled_1_df = resample_positives(df, resample_rate, label_col, pos_label, reshuffle=False)
    assert df.shape == resampled_1_df.shape
    assert (
        df[df[label_col] == pos_label].shape
        == resampled_1_df[resampled_1_df[label_col] == pos_label].shape
    )

    # Verify correct number of rows and correct rows added if not equal to one
    resample_rate = 3
    resampled_3_df = resample_positives(df, resample_rate, label_col, pos_label, reshuffle=False)
    assert df.shape[1] == resampled_3_df.shape[1]
    assert (
        df[df[label_col] == pos_label].shape[0] * 3
        == resampled_3_df[resampled_3_df[label_col] == pos_label].shape[0]
    )
    assert (
        df[df[label_col] != pos_label].shape
        == resampled_3_df[resampled_3_df[label_col] != pos_label].shape
    )


def test_resample_positives_multilabel():
    path = "tests/resources/dummy_dataset_multilabel.csv"
    text_col = "text"
    text_id_col = "text_id"
    label_cols = ["label1", "label2", "label3"]
    pos_label = "1"

    df = read_csv_text_classifier(
        path, text_col=text_col, label_cols=label_cols, additional_cols=["text_id"]
    )

    resample_rate_dict = {"label1": 1, "label2": 1, "label3": 1}

    # Verify things are unmodified if resample rate is 1
    rdf = resample_multilabel_positives(df, resample_rate_dict, pos_label, reshuffle=False)
    assert df.shape == rdf.shape

    # Verify correct number of rows and correct rows added if a single is not equal to one
    resample_rate_dict = {"label1": 1, "label2": 1, "label3": 4}
    rdf = resample_multilabel_positives(df, resample_rate_dict, pos_label, reshuffle=False)
    assert rdf.shape[0] == 17
    assert df.shape[1] == rdf.shape[1]

    # Heres the original dataframe
    # ,group_id,text_id,text,label1,label2,label3
    # 0,123,123_0,This conference will now be recorded.,0,0,0
    # 1,123,123_1,potato,1,0,1
    # 2,123,123_2," potato but in quotes",0,1,1
    # 3,123,123_3,the end of all things potato,0,0,1
    # 4,123,123_4,final potato,1,1,1

    # Rows to be added for label1: 4
    # 5,123,123_1,potato,1,0,1
    # 6,123,123_2," potato but in quotes",0,1,1
    # 7,123,123_3,the end of all things potato,0,0,1
    # 8,123,123_4,final potato,1,1,1
    # 9,123,123_1,potato,1,0,1
    # 10,123,123_2," potato but in quotes",0,1,1
    # 11,123,123_3,the end of all things potato,0,0,1
    # 12,123,123_4,final potato,1,1,1
    # 13,123,123_1,potato,1,0,1
    # 14,123,123_2," potato but in quotes",0,1,1
    # 15,123,123_3,the end of all things potato,0,0,1
    # 16,123,123_4,final potato,1,1,1
    # Check each text id individually for how many copies it should have
    assert rdf[rdf[text_id_col] == "123_0"].shape[0] == 1
    assert rdf[rdf[text_id_col] == "123_1"].shape[0] == 4
    assert rdf[rdf[text_id_col] == "123_2"].shape[0] == 4
    assert rdf[rdf[text_id_col] == "123_3"].shape[0] == 4
    assert rdf[rdf[text_id_col] == "123_4"].shape[0] == 4

    # Verify correct number of rows and correct rows added if multiple are unequal to 1
    resample_rate_dict = {"label1": 2, "label2": 2, "label3": 3}

    # Heres the original dataframe
    # ,group_id,text_id,text,label1,label2,label3
    # 0,123,123_0,This conference will now be recorded.,0,0,0
    # 1,123,123_1,potato,1,0,1
    # 2,123,123_2," potato but in quotes",0,1,1
    # 3,123,123_3,the end of all things potato,0,0,1
    # 4,123,123_4,final potato,1,1,1

    # Rows to be added for label1: 2
    # 5,123,123_1,potato,1,0,1
    # 6,123,123_4,final potato,1,1,1

    # Rows to be added for label2: 2
    # 7,123,123_2," potato but in quotes",0,1,1
    # 8,123,123_4,final potato,1,1,1

    # Rows to be added for label3: 3
    # 9,123,123_1,potato,1,0,1
    # 10,123,123_2," potato but in quotes",0,1,1
    # 11,123,123_3,the end of all things potato,0,0,1
    # 12,123,123_4,final potato,1,1,1
    # 13,123,123_1,potato,1,0,1
    # 14,123,123_2," potato but in quotes",0,1,1
    # 15,123,123_3,the end of all things potato,0,0,1
    # 16,123,123_4,final potato,1,1,1

    rdf = resample_multilabel_positives(df, resample_rate_dict, pos_label, reshuffle=False)
    assert df.shape[1] == rdf.shape[1]
    assert rdf.shape[0] == 17

    # #Check each text id individually for how many copies it should have
    # assert rdf[rdf['text_id']=='123_0'].shape[0] == 1
    # assert rdf[rdf['text_id']=='123_1'].shape[0] == 4
    # assert rdf[rdf['text_id']=='123_2'].shape[0] == 4
    # assert rdf[rdf['text_id']=='123_3'].shape[0] == 3
    # assert rdf[rdf['text_id']=='123_4'].shape[0] == 5

    assert rdf[rdf[text_id_col] == "123_0"].shape[0] == 1
    assert rdf[rdf[text_id_col] == "123_1"].shape[0] == 4
    assert rdf[rdf[text_id_col] == "123_2"].shape[0] == 4
    assert rdf[rdf[text_id_col] == "123_3"].shape[0] == 3
    assert rdf[rdf[text_id_col] == "123_4"].shape[0] == 5
