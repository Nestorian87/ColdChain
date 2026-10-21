from pathlib import Path
import unittest
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
class DatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw=pd.read_csv(ROOT/'data/raw/telemetry.csv')
        cls.f=pd.read_csv(ROOT/'data/processed/shipment_features.csv')
    def test_count_and_unique_keys(self):
        self.assertEqual(len(self.raw),86400)
        self.assertFalse(self.raw.duplicated(['shipment_id','sample_index']).any())
        self.assertTrue(self.raw.groupby('shipment_id').size().eq(144).all())
    def test_no_group_leakage(self):
        g={k:set(v.shipment_id) for k,v in self.f.groupby('split')}
        self.assertFalse(g['train'] & g['validation'] or g['train'] & g['test'] or g['validation'] & g['test'])
        self.assertLess(max(g['train']),min(g['validation']))
        self.assertLess(max(g['validation']),min(g['test']))
    def test_feature_schema(self):
        self.assertNotIn('latent_final_exposure',self.f.columns)
        self.assertEqual(len(self.f),600)
        self.assertTrue(self.f['simulated_spoilage'].isin([0,1]).all())
        self.assertTrue(self.f['missing_fraction'].between(0,1).all())
        # A prefix cannot contain more than 720 observed overheating minutes.
        self.assertTrue(self.f['minutes_above8'].between(0,720).all())
if __name__=='__main__': unittest.main()
