import unittest
import pandas as pd
import sys
import os

sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "src", "dataProcessing")
    )
)
from download_skogsbilveg import hent_skogsbilveier_og_noder
from endepunkt import filtrer_ekte_endepunkter

class TestEkteEndepunkterDataframe(unittest.TestCase):

    def test_ekte_endepunkter_dataframe(self):
     
        df = hent_skogsbilveier_og_noder("0301")
        self.assertIsInstance(df, pd.DataFrame)
        self.assertIn("nodeid", df.columns)
        self.assertGreater(len(df), 0)

       
        ekte_df = filtrer_ekte_endepunkter(df)
        print("Eksempel på ekte endepunkt-dataframe:\n", ekte_df.head())

       
        self.assertIsInstance(ekte_df, pd.DataFrame)
        self.assertIn("nodeid", ekte_df.columns)
        self.assertIn("wkt", ekte_df.columns)
        self.assertIn("x", ekte_df.columns)
        self.assertIn("y", ekte_df.columns)

    
        self.assertGreaterEqual(len(ekte_df), 0)
        if len(ekte_df) > 0:
     
            row = ekte_df.iloc[0]
            self.assertTrue(pd.notnull(row["wkt"]))
            self.assertTrue(pd.notnull(row["x"]))
            self.assertTrue(pd.notnull(row["y"]))

if __name__ == "__main__":
    unittest.main()
