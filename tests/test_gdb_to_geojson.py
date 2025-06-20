from unittest.mock import patch, MagicMock

from src.dataProcessing.gdb_to_geojson import extract_turning_spaces_layers


@patch("src.dataProcessing.gdb_to_geojson.fiona.listlayers")
@patch("src.dataProcessing.gdb_to_geojson.gpd.read_file")
def test_extract_turning_spaces_layers(mock_read_file, mock_listlayers, tmp_path):
    """
    Tester at funksjonen henter ut lag med snuplasser fra en GDB-fil
    og lagrer dem som GeoJSON.
    """
    mock_listlayers.return_value = [
        "Snuplasser_areal_1",
        "snuplasser_areal_2",
        "Other_Layer",
    ]

    mock_gdf = MagicMock()
    mock_gdf.to_crs.return_value = mock_gdf
    mock_read_file.return_value = mock_gdf

    gdb_path = "dummy/path/to.gdb"
    output_path = tmp_path / "turning_spaces.geojson"

    # Kjør funksjonen
    extract_turning_spaces_layers(gdb_path, str(output_path))

    # Valider at kun de riktige lagene ble behandlet
    assert mock_read_file.call_count == 2
    assert mock_gdf.to_file.call_count == 2
    mock_gdf.to_file.assert_called_with(str(output_path), driver="GeoJSON")
