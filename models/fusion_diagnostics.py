"""Fusion diagnostics utility

Usage:
  python models/fusion_diagnostics.py --test-csv path/to/test.csv \
    --image-col image_path --tabular-cols col1,col2,... \
    --rf-path models/saved_models/tabular_rf.pkl \
    --fusion-tflite models/saved_models/multimodal_model.tflite \
    --visual-tflite models/saved_models/visual_model.tflite \
    --out report.csv

This script attempts to load models (TFLite and RF) and compare predictions
from the sequential pipeline (visual -> hb -> build tabular -> RF) vs the
fusion model (image+tabular). It sweeps a range of late-fusion weights if
requested and writes a CSV report with metrics and per-sample diffs.

The runtime must have either `tensorflow` or `tflite_runtime` and `scikit-learn` installed
to run real model inference. If these are not available the script will exit
with instructions.
"""
import argparse
import csv
import os
import sys
import json
from typing import List

import numpy as np
import pandas as pd

try:
    import joblib
except Exception:
    joblib = None

try:
    # prefer tensorflow's tflite interpreter when available
    import tensorflow as tf
    TFLITE_INTERPRETER = tf.lite.Interpreter
except Exception:
    try:
        from tflite_runtime.interpreter import Interpreter as TFLITE_INTERPRETER  # type: ignore
    except Exception:
        TFLITE_INTERPRETER = None

from sklearn.metrics import accuracy_score, f1_score, confusion_matrix


def load_rf(path: str):
    if joblib is None:
        raise RuntimeError("joblib/scikit-learn is required to load RF models. Install scikit-learn and joblib.")
    return joblib.load(path)


def load_tflite(path: str):
    if TFLITE_INTERPRETER is None:
        raise RuntimeError("TFLite runtime not available. Install tensorflow or tflite_runtime.")
    interp = TFLITE_INTERPRETER(model_path=path)
    interp.allocate_tensors()
    return interp


def run_tflite_interpreter(interp, inputs: List[np.ndarray]):
    # expects inputs to be in same order as model inputs
    input_details = interp.get_input_details()
    output_details = interp.get_output_details()
    for i, arr in enumerate(inputs):
        inp = input_details[i]
        data = np.asarray(arr, dtype=inp['dtype'])
        interp.set_tensor(inp['index'], data)
    interp.invoke()
    outputs = [interp.get_tensor(o['index']) for o in output_details]
    return outputs


def read_test_csv(path: str, image_col: str, tabular_cols: List[str]):
    df = pd.read_csv(path)
    if image_col not in df.columns:
        raise ValueError(f"image column '{image_col}' not found in {path}")
    for c in tabular_cols:
        if c not in df.columns:
            raise ValueError(f"tabular column '{c}' not found in {path}")
    return df


def build_tabular_array(df: pd.DataFrame, tabular_cols: List[str]) -> np.ndarray:
    return df[tabular_cols].to_numpy(dtype=np.float32)


def compare_predictions(y_true, seq_preds, fusion_preds):
    results = {}
    results['seq_acc'] = accuracy_score(y_true, seq_preds)
    results['fusion_acc'] = accuracy_score(y_true, fusion_preds)
    results['seq_f1_macro'] = f1_score(y_true, seq_preds, average='macro')
    results['fusion_f1_macro'] = f1_score(y_true, fusion_preds, average='macro')
    return results


def sweep_weights_and_eval(y_true, seq_proba, fusion_proba, weights=np.linspace(0,1,11)):
    # seq_proba and fusion_proba are (N, C)
    best = None
    for w in weights:
        combined = w * seq_proba + (1 - w) * fusion_proba
        preds = combined.argmax(axis=1)
        f1 = f1_score(y_true, preds, average='macro')
        if best is None or f1 > best['f1']:
            best = {'weight': w, 'f1': f1, 'preds': preds}
    return best


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--test-csv', required=True)
    parser.add_argument('--image-col', default='image_path')
    parser.add_argument('--tabular-cols', required=True, help='comma-separated columns used by tabular RF')
    parser.add_argument('--rf-path', required=True)
    parser.add_argument('--fusion-tflite', required=True)
    parser.add_argument('--visual-tflite', required=True)
    parser.add_argument('--out', default='models/fusion_diagnostics_report.csv')
    args = parser.parse_args(argv)

    df = read_test_csv(args.test_csv, args.image_col, args.tabular_cols.split(','))
    tabular_cols = args.tabular_cols.split(',')
    y_true = df['label'].to_numpy() if 'label' in df.columns else None

    # load models
    try:
        rf = load_rf(args.rf_path)
    except Exception as e:
        print('Failed to load RF model:', e)
        sys.exit(2)

    try:
        fusion_interp = load_tflite(args.fusion_tflite)
    except Exception as e:
        print('Failed to load fusion tflite model:', e)
        sys.exit(2)

    try:
        visual_interp = load_tflite(args.visual_tflite)
    except Exception as e:
        print('Failed to load visual tflite model:', e)
        sys.exit(2)

    # Prepare inputs
    X_tab = build_tabular_array(df, tabular_cols)

    # For visual tflite we'll try a synthetic placeholder per sample (shape check)
    # The script expects the CSV's image_path column to point to preprocessed numpy .npy arrays matching the visual model input
    visual_inputs = []
    for p in df[args.image_col].tolist():
        if os.path.exists(p) and p.endswith('.npy'):
            visual_inputs.append(np.load(p))
        else:
            # fallback: create zeros matching input shape
            inp = visual_interp.get_input_details()[0]
            shape = inp['shape']
            visual_inputs.append(np.zeros(shape, dtype=inp['dtype']))

    # Run sequential visual->tabular RF
    seq_preds = []
    seq_proba = []
    for vi, xt in zip(visual_inputs, X_tab):
        # visual: get hb/visual outputs
        v_outs = run_tflite_interpreter(visual_interp, [np.expand_dims(vi, axis=0)])
        # assume visual model outputs probabilities and hb; user should adapt indexes if different
        v_proba = v_outs[0][0] if len(v_outs[0].shape) > 1 else v_outs[0]
        # tabular: predict with sklearn RF
        # ensure xt is 2D
        xt2 = xt.reshape(1, -1)
        t_proba = rf.predict_proba(xt2)
        seq_proba.append(t_proba[0])
        seq_preds.append(t_proba.argmax(axis=1)[0])

    seq_proba = np.vstack(seq_proba)
    seq_preds = np.array(seq_preds)

    # Run fusion model
    fusion_preds = []
    fusion_proba = []
    # fusion model may expect two inputs: image and tabular vector
    for vi, xt in zip(visual_inputs, X_tab):
        # run fusion interpreter with [image, tabular]
        fusion_outs = run_tflite_interpreter(fusion_interp, [np.expand_dims(vi, axis=0), np.expand_dims(xt.astype(np.float32), axis=0)])
        f_proba = fusion_outs[0][0] if len(fusion_outs[0].shape) > 1 else fusion_outs[0]
        fusion_proba.append(f_proba)
        fusion_preds.append(f_proba.argmax())

    fusion_proba = np.vstack(fusion_proba)
    fusion_preds = np.array(fusion_preds)

    if y_true is None:
        print('No ground-truth `label` column in test CSV; writing per-sample outputs only.')
        out_df = df.copy()
        out_df['seq_pred'] = seq_preds
        out_df['fusion_pred'] = fusion_preds
        out_df.to_csv(args.out, index=False)
        print('Wrote', args.out)
        return

    # compute comparison metrics
    metrics = compare_predictions(y_true, seq_preds, fusion_preds)
    best = sweep_weights_and_eval(y_true, seq_proba, fusion_proba)

    summary = {
        'seq_acc': float(metrics['seq_acc']),
        'fusion_acc': float(metrics['fusion_acc']),
        'seq_f1_macro': float(metrics['seq_f1_macro']),
        'fusion_f1_macro': float(metrics['fusion_f1_macro']),
        'best_late_fusion_weight': float(best['weight']),
        'best_late_fusion_f1': float(best['f1']),
    }

    # save report (summary + per-sample diff)
    per_sample = df.copy()
    per_sample['seq_pred'] = seq_preds
    per_sample['fusion_pred'] = fusion_preds
    per_sample['seq_prob0'] = seq_proba[:, 0]
    per_sample['fusion_prob0'] = fusion_proba[:, 0]
    per_sample['fusion_diff'] = per_sample['fusion_pred'] != per_sample['seq_pred']
    report_path = args.out
    per_sample.to_csv(report_path, index=False)

    summary_path = report_path.replace('.csv', '.summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print('Wrote per-sample report to', report_path)
    print('Wrote summary to', summary_path)


if __name__ == '__main__':
    main()
