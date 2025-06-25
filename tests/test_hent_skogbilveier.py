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

class TestSkogsbilveierOgEndepunkter(unittest.TestCase):

    def test_hent_skogsbilveier_og_noder(self):
        # Henter alle skogsbilveier og noder for kommune 301
        df = hent_skogsbilveier_og_noder("0301")
        print("Eksempel på dataframe (før filtrering):\n", df.head())

        
        self.assertIsInstance(df, pd.DataFrame)
        self.assertIn("veglenkesekvensid", df.columns)
        self.assertIn("nodeid", df.columns)
        self.assertGreater(len(df), 0)

    def test_filtrer_ekte_endepunkter(self):
       
        df = hent_skogsbilveier_og_noder("0301")
        
        ekte_endepunkter = filtrer_ekte_endepunkter(df)
        print("Eksempel på ekte endepunkter:\n", ekte_endepunkter.head())

        
        self.assertIsInstance(ekte_endepunkter, pd.DataFrame)
        
        self.assertTrue(set(ekte_endepunkter["nodeid"]).issubset(set(df["nodeid"])))
       
        self.assertGreaterEqual(len(ekte_endepunkter), 0)

if __name__ == "__main__":
    unittest.main()
