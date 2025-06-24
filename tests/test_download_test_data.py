import pytest
from pathlib import Path
from unittest.mock import AsyncMock, patch

import src.config as config
from src.dataProcessing.download_test_data import main


@pytest.mark.asyncio()
@patch("src.dataProcessing.download_test_data.download_image", new_callable=AsyncMock)
@patch("pathlib.Path.exists", return_value=False)
@patch("pathlib.Path.mkdir")
async def test_main_downloads_correct_tiles(mock_mkdir, _, mock_download_image):
    """
    Tester at main-funksjonen laster ned bilder for riktig antall fliser
    og at de har forventet størrelse og plassering.
    """
    test_bbox = [0, 0, 200, 200]
    image_size = [500, 500]  # px
    resolution = 0.2  # m/px → 100x100m tile
    expected_tiles = [
        [0, 0, 100.0, 100.0],
        [0, 100.0, 100.0, 200.0],
        [100.0, 0, 200.0, 100.0],
        [100.0, 100.0, 200.0, 200.0],
    ]

    with patch.object(config, "TEST_BBOX", test_bbox), patch.object(
        config, "IMAGE_SIZE", image_size
    ), patch.object(config, "RESOLUTION", resolution):

        await main()

        assert mock_download_image.call_count == len(expected_tiles)

        for call, expected_bbox in zip(
            mock_download_image.call_args_list, expected_tiles
        ):
            called_bbox, called_path = call.args
            assert called_bbox == expected_bbox
            assert isinstance(called_path, Path)
            assert "image_" in str(called_path)

        # Sjekk at mappen ble forsøkt laget
        mock_mkdir.assert_called()
