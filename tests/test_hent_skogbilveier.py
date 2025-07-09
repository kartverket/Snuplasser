import unittest
import pandas as pd
from src.dataProcessing.download_skogsbilveg import hent_skogsbilveier_og_noder


class TestSkogsbilveierHenting(unittest.TestCase):

    def test_henter_data_for_kommune(self):
        kommune_id = "0301"  
        df = hent_skogsbilveier_og_noder(kommune_id)

        
        self.assertIsInstance(df, pd.DataFrame)

        self.assertIn("nodeid", df.columns)
        self.assertIn("veglenkesekvensid", df.columns)

      
        self.assertGreater(len(df), 0)

        print("\n📦 Eksempeldata (first 4 raws):\n", df.head())


if __name__ == "__main__":
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
