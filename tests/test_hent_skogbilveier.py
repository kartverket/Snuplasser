import unittest
import pandas as pd
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src', 'dataProcessing')))
from download_skogsbilveg import hent_skogbilveier_fra_vegnett


class TestRealNVDBAPI(unittest.TestCase):

    def test_hent_skogbilveier_real_api(self):
        df = hent_skogbilveier_fra_vegnett("0301") 
        print(df.head())  

        self.assertIsInstance(df, pd.DataFrame)
        self.assertGreater(len(df), 0)
        self.assertIn("veglenkesekvensid", df.columns)

if __name__ == "__main__":
    unittest.main()
