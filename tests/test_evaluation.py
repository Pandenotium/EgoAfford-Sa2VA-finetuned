import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from PIL import Image
from egoafford import ROLES
from egoafford.metrics import Metrics
from egoafford.data import relative_file, read_jsonl, load_masks, validate_manifest
from egoafford.__main__ import evaluate


class EvaluationTests(unittest.TestCase):
    def test_empty_roles(self):
        m = Metrics(); z = np.zeros((3, 2, 2), bool); m.update(z, z)
        r = m.compute()
        self.assertEqual(r['gIoU'], 1); self.assertEqual(r['cIoU'], 1)
        self.assertIsNone(r['component_wise']['instrument']['NE_mIoU'])
        self.assertIsNone(r['presence_macro_F1'])

    def test_false_positive_and_missing_present_role(self):
        g = np.zeros((3, 2, 2), bool); p = g.copy()
        g[0, 0, 0] = True; p[1, 0, 0] = True
        m = Metrics(); m.update(p, g); r = m.compute()
        self.assertEqual(r['cIoU'], 0); self.assertAlmostEqual(r['gIoU'], 1/3)
        self.assertEqual(r['component_wise']['direct_object']['NE_mIoU'], 0)
        self.assertEqual(r['component_wise']['instrument']['FP'], 1)

    def test_global_ciou_is_not_mean_sample_iou(self):
        m = Metrics()
        m.update_counts([1,0,0],[1,0,0],[True,False,False],[True,False,False])
        m.update_counts([0,0,0],[9,0,0],[True,False,False],[False,False,False])
        r = m.compute(); self.assertAlmostEqual(r['cIoU'], .1)
        self.assertAlmostEqual(r['gIoU'], 5/6)
        self.assertAlmostEqual(r['component_wise']['direct_object']['NE_mIoU'], .5)

    def test_role_swaps_are_not_reassigned(self):
        g = np.zeros((3,2,2),bool); g[0,0,0]=True; g[1,1,1]=True
        p = g[[1,0,2]]; m = Metrics(); m.update(p,g)
        self.assertEqual(m.compute()['cIoU'], 0)

    def test_invalid_inputs(self):
        with self.assertRaises(ValueError): Metrics().compute()
        with self.assertRaises(ValueError): Metrics().update(np.zeros((3,2,2)), np.zeros((3,2,2)))
        with self.assertRaises(ValueError): Metrics().update_counts([2,0,0],[1,0,0],[True]*3,[True]*3)
        with self.assertRaises(ValueError): Metrics().update_counts([0]*3,[0]*3,[True]*3,[False]*3)

    def test_manifest_and_cli_contract(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); masks={}
            for i,role in enumerate(ROLES):
                a=np.zeros((2,3),np.uint8)
                if i==0:a[0,0]=255
                Image.fromarray(a).save(root/f'{role}.png');masks[role]=f'{role}.png'
            Image.new('RGB',(3,2)).save(root/'image.png')
            row=dict(id='sample',action='Move the cup',source='synthetic',group_id='task1',image='image.png',masks=masks)
            (root/'gt.jsonl').write_text(json.dumps(row)+'\n');(root/'p.jsonl').write_text(json.dumps(dict(id='sample',masks=masks))+'\n')
            self.assertEqual(validate_manifest(root/'gt.jsonl',root,True)['num_samples'],1)
            self.assertEqual(evaluate(root/'gt.jsonl',root/'p.jsonl',root,root)['cIoU'],1)
            (root/'p.jsonl').write_text(json.dumps(dict(id='other',masks=masks))+'\n')
            with self.assertRaises(ValueError): evaluate(root/'gt.jsonl',root/'p.jsonl',root,root)
            with self.assertRaises(ValueError): relative_file(root,'../outside')
            (root/'gt.jsonl').write_text((json.dumps(row)+'\n')*2)
            with self.assertRaises(ValueError): read_jsonl(root/'gt.jsonl')

    def test_predicted_null_means_explicit_empty(self):
        p=load_masks(dict(id='x',masks=dict.fromkeys(ROLES)),'.',(2,3),True)
        self.assertFalse(p.any())


if __name__ == '__main__': unittest.main()
