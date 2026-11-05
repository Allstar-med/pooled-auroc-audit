"""Small synthetic tests; no source-dataset records are included."""
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import oea_core as core
from audit_all import exclude_test_frames, validate_predictions


def explicit_pairs(unit, y, score):
    """Independent enumeration of positive-negative pairs for tiny inputs."""
    pairs = [(u, v, float(score[i] > score[j]) + .5 * (score[i] == score[j]))
             for i, u in enumerate(unit) if y[i] == 1
             for j, v in enumerate(unit) if y[j] == 0]
    within = [x for u, v, x in pairs if u == v]
    cross = [x for u, v, x in pairs if u != v]
    per_unit = [[x for u, v, x in pairs if u == v == k] for k in np.unique(unit)]
    mean = lambda x: float(np.mean(x)) if len(x) else np.nan
    return dict(E1=mean([x for _, _, x in pairs]), W=mean(within), X=mean(cross),
                E2=mean([mean(x) for x in per_unit if x]), alpha=len(within) / len(pairs))


class CoreTests(unittest.TestCase):
    def test_pair_enumeration_with_ties(self):
        u = np.repeat(np.arange(3), 4)
        y = np.tile([0, 1, 0, 1], 3)
        s = np.array([.1, .1, .3, .8, .4, .8, .1, .2, .9, .7, .1, .7])
        actual = core.estimands_point(u, y, s)
        for key, expected in explicit_pairs(u, y, s).items():
            self.assertAlmostEqual(actual[key], expected)

    def test_repeated_bootstrap_units_match_expanded_sample(self):
        u = np.repeat(np.arange(3), 4)
        y = np.tile([0, 1, 0, 1], 3)
        s = np.array([.1, .1, .3, .8, .4, .8, .1, .2, .9, .7, .1, .7])
        for counts in ([2, 0, 1], [0, 2, 1], [1, 1, 1]):
            expanded_u, expanded_y, expanded_s = [], [], []
            for copy, source in enumerate(np.repeat(np.arange(3), counts)):
                ix = u == source
                expanded_u.extend([copy] * int(ix.sum()))
                expanded_y.extend(y[ix]); expanded_s.extend(s[ix])
            expected = explicit_pairs(expanded_u, expanded_y, expanded_s)
            actual = core.from_counts(core.prepare(np.arange(3), u, y, s), counts)
            for key in expected:
                self.assertAlmostEqual(actual[key][0], expected[key])

    def test_cross_auc_when_within_auc_is_undefined(self):
        result = core.estimands_point([0, 0, 1, 1], [0, 0, 1, 1], [.1, .5, .5, .9])
        self.assertTrue(np.isnan(result["W"]))
        self.assertAlmostEqual(result["X"], .875)
        self.assertEqual(result["alpha"], 0)

    def test_invalid_statistical_input_is_rejected(self):
        for units, labels, scores in [([0, 1], [0, 2], [.1, .2]),
                                      ([0, 1], [0, 1], [.1, np.nan]),
                                      ([0], [0, 1], [.1, .2])]:
            with self.subTest(units=units, labels=labels, scores=scores), self.assertRaises(ValueError):
                core.prepare([0, 1], units, labels, scores)


def predictions():
    rows = []
    scores = {"a": [.95, .2, .8, .6], "b": [.1, .7, .6, .9], "c": [.2, .3, .7, .8]}
    for split in ("val", "test"):
        for unit in range(3):
            for model, values in scores.items():
                for frame, y in enumerate([0, 0, 1, 1]):
                    rows.append(dict(dataset="Endoscapes", split=split, unit=f"synthetic_{unit}",
                                     frame_id=frame, t=frame * 5, label="y_C1", y=y,
                                     score=values[frame] + unit * .01, model=model, backbone="synthetic",
                                     head="causal" if model == "c" else "frame", C=1,
                                     sel_E1=int(model == "a"), sel_W=int(model == "b"), sel_head=int(model != "b")))
    return pd.DataFrame(rows)


class InputTests(unittest.TestCase):
    def test_duplicate_predictions_are_rejected(self):
        table = predictions()
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            validate_predictions(pd.concat([table, table.iloc[:1]]))

    def test_conflicting_frame_labels_are_rejected(self):
        table = predictions(); table.loc[0, "y"] = 1
        with self.assertRaisesRegex(ValueError, "disagree"):
            validate_predictions(table)

    def test_exclusion_is_label_specific(self):
        table = predictions()
        screen = table.iloc[:1][["dataset", "unit", "frame_id", "label"]].copy()
        screen["label"] = "y_C2"
        self.assertEqual(len(exclude_test_frames(table, screen)), len(table))

    def test_cli_applies_exclusions_to_main_selection_and_heads(self):
        table = predictions()
        screen = pd.DataFrame([dict(dataset="Endoscapes", unit="synthetic_0", frame_id=0, label="y_C1")])
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            table.to_csv(folder / "pred.csv", index=False); screen.to_csv(folder / "screen.csv", index=False)
            run = subprocess.run([sys.executable, str(ROOT / "src/audit_all.py"), str(folder / "out"),
                                  str(folder / "pred.csv"), "--zoo", "all", "--B", "30", "--exclude",
                                  str(folder / "screen.csv")], capture_output=True, text=True)
            self.assertEqual(run.returncode, 0, run.stderr)
            selected = pd.read_csv(folder / "out/selection.csv").iloc[0]
            heads = pd.read_csv(folder / "out/heads.csv").set_index("estimand")
            main = pd.read_csv(folder / "out/estimands.csv").set_index("model")
            expected = {}
            for model in ("a", "b", "c"):
                t = table[(table.split == "test") & (table.model == model)]
                t = t[~((t.unit == "synthetic_0") & (t.frame_id == 0))]
                expected[model] = explicit_pairs(t.unit.to_numpy(), t.y.to_numpy(), t.score.to_numpy())
                self.assertAlmostEqual(main.loc[model, "W"], expected[model]["W"], places=5)
            self.assertAlmostEqual(selected.test_W_selE1, expected["a"]["W"], places=5)
            self.assertAlmostEqual(selected.test_W_selW, expected["b"]["W"], places=5)
            self.assertAlmostEqual(heads.loc["W", "diff"], expected["c"]["W"] - expected["a"]["W"], places=5)


if __name__ == "__main__":
    unittest.main()
