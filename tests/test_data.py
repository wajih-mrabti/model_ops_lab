from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from modelops.data.prepare import prepare_data


def test_prepare_data_creates_dataframe(tmp_path):
    raw_path = tmp_path / "raw" / "iris.csv"
    output_path = tmp_path / "processed" / "iris_processed.csv"

    df = prepare_data(raw_path=raw_path, output_path=output_path)

    assert "species" in df.columns
    assert output_path.exists()
    assert len(df) > 0
