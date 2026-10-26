from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd
from modelops.data.prepare import prepare_data


def test_prepare_data_creates_dataframe(tmp_path):
    raw_path = tmp_path / "raw" / "iris.csv"
    output_path = tmp_path / "processed" / "iris_processed.csv"

    df = prepare_data(
        raw_path=raw_path,
        output_path=output_path,
    )

    assert output_path.exists()
    assert len(df) == 150
    assert "species" in df.columns
    assert df.columns.is_unique
    assert df["species"].notna().all()

    assert set(df["species"].unique()) == {
        "setosa",
        "versicolor",
        "virginica",
    }

    saved_df = pd.read_csv(output_path)
    assert list(saved_df.columns) == list(df.columns)
    assert len(saved_df) == len(df)
