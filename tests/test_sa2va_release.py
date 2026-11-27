import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from egoafford_sa2va.common import rows, ground_truth, statistics, aggregate, ROLES
from egoafford_sa2va.config import configure
from egoafford_sa2va.protocol import prompt, target, map_prediction
from egoafford.metrics import Metrics


class Sa2VAReleaseTests(unittest.TestCase):
    def test_role_mapping_uses_text_and_preserves_recoverable_masks(self):
        masks = np.zeros((3, 5, 7), dtype=bool)
        masks[0, :2, :2] = True
        masks[2, 3:, 4:] = True
        # Malformed box, but each mask still has an identifiable role line.
        answer = ('Components:\ndirect_object: cup; box: bad [SEG]\n'
                  'instrument: None; box: [0, 0, 0, 0] [SEG]\n'
                  'destination: tray; box: [4, 3, 6, 4] [SEG]')
        result, valid, fmt = map_prediction(answer, masks, (5, 7), 'emit_empty')
        np.testing.assert_array_equal(result, masks)
        self.assertEqual(valid, [True, True, True])
        self.assertFalse(fmt)
        self.assertNotIn('[SEG]', prompt('Move the cup', 'full_rule', 'emit_empty'))

    def test_png_and_packed_gt_have_identical_targets(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            masks = np.zeros((3, 5, 7), dtype=bool)
            masks[0, :2, :2] = True
            masks[2, 3:, 4:] = True
            paths = {}
            for role, mask in zip(ROLES, masks):
                paths[role] = role + '.png'
                Image.fromarray(mask.astype('uint8') * 255).save(root / paths[role])
            nonempty = masks[[0, 2]]
            np.savez(root / 'packed.npz', shape=np.array((1, 3, 5, 7)),
                     nonempty_idx=np.array([[0, 0], [0, 2]]),
                     nonempty_shape=nonempty.shape, packed=np.packbits(nonempty))
            row = dict(id='test', image='image.jpg', action='Move the cup onto the tray',
                       source='synthetic', names=dict(zip(ROLES, ['cup', 'None', 'tray'])))
            png = dict(row, masks=paths)
            packed = dict(row, mask_path='packed.npz', mask_index=0)
            manifest = root / 'examples.jsonl'
            manifest.write_text(json.dumps(png) + '\n' + json.dumps(packed) + '\n')
            with patch.dict(os.environ, {'EGOAFFORD_DATA_ROOT': str(root)}):
                a, b = rows(manifest)
            self.assertEqual(a['data_source'], 'synthetic')
            np.testing.assert_array_equal(ground_truth(a), masks)
            np.testing.assert_array_equal(ground_truth(a), ground_truth(b))
            ta, ma, ia = target(a, 'emit_empty')
            tb, mb, ib = target(b, 'emit_empty')
            self.assertEqual(ta, tb)
            self.assertEqual(ia, ib)
            np.testing.assert_array_equal(ma, mb)

    def test_native_and_portable_metrics_agree(self):
        gt = np.zeros((3, 5, 7), dtype=bool)
        gt[0, :2, :2] = True
        gt[2, 3:, 4:] = True
        pred = gt.copy()
        pred[2] = False
        record = dict(id='scene/image', format_valid=True, **statistics(pred, gt))
        native = aggregate([record])
        portable = Metrics()
        portable.update(pred, gt)
        values = portable.compute()
        self.assertAlmostEqual(native['cIoU'], values['cIoU'])
        self.assertAlmostEqual(native['gIoU'], values['gIoU'])
        for role in ROLES:
            self.assertEqual(native['component_wise'][role]['NE_mIoU'],
                             values['component_wise'][role]['NE_mIoU'])

    def test_public_named_npz(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            masks = np.zeros((3, 5, 7), dtype=bool)
            masks[0, :2, :2] = True
            np.savez_compressed(root / 'roles.npz', **dict(zip(ROLES, masks)))
            row = dict(id='synthetic/example', schema_version='functional_role_annotation/1.0',
                       mask='roles.npz', image=None, height=5, width=7, action='Lift the cup',
                       names=dict(zip(ROLES, ['cup', None, None])))
            (root / 'test.jsonl').write_text(json.dumps(row) + '\n')
            with patch.dict(os.environ, {'EGOAFFORD_DATA_ROOT': str(root/'legacy-root')}):
                parsed = rows(root / 'test.jsonl')[0]
            self.assertEqual(parsed['data_source'], 'synthetic')
            np.testing.assert_array_equal(ground_truth(parsed), masks)
            from egoafford.data import load_masks
            np.testing.assert_array_equal(load_masks(row, root), masks)
            from egoafford.data import validate_manifest
            self.assertEqual(validate_manifest(root/'test.jsonl', root)['missing_group_ids'],1)
            answer, result, _ = target(parsed, 'emit_empty')
            self.assertIn('instrument: None;', answer)
            np.testing.assert_array_equal(result, masks)

    def test_recipe_resolves_without_private_paths(self):
        root = Path(__file__).resolve().parents[1]
        with patch.dict(os.environ):
            recipe = configure(root / 'configs/sa2va_8b.json')
        self.assertEqual(recipe['initialization'], 'components')
        self.assertEqual(recipe['effective_batch'], 32)
        self.assertEqual(recipe['epochs'], 40)
        self.assertTrue(Path(recipe['upstream']).is_dir())
        self.assertNotIn('/inspire/', (root / 'configs/sa2va_8b.json').read_text())
        self.assertEqual(recipe['upstream'], str(root / 'third_party/sa2va'))


if __name__ == '__main__':
    unittest.main()
