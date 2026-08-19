"""Environment shim to run the OFFICIAL VectorInstitute/laftr (TF 1.9-era)
unmodified under modern TensorFlow (2.21, tf.compat.v1) / Python 3.12 / numpy 2.

The LAFTR source tree is NOT edited. Four environment-level aliases:
  1. sys.modules['tensorflow'] -> tf.compat.v1, with v2 behavior disabled.
     This is the standard TF1-code migration shim, no semantic change.
  2. tf.contrib.layers.xavier_initializer -> tf.glorot_uniform_initializer.
     These are the SAME initializer (Glorot/Xavier uniform), and tf.contrib
     was removed in TF2. Mathematically identical.
  3. collections.Iterator -> collections.abc.Iterator, and
  4. collections.Mapping -> collections.abc.Mapping. Both ABCs moved to
     collections.abc in Python 3.10. The objects themselves are unchanged.

Usage:  python utils/laftr_shim.py <script-under-laftr/src> [laftr args...]
        (cwd must be the laftr repo root, as their README requires. The
        launching venv needs modern tensorflow + jinja2, e.g. python3.12 with
        tensorflow 2.21. See experiments/laftr_official.py.)
"""
import collections
import collections.abc
import sys
import types

collections.Iterator = collections.abc.Iterator
collections.Mapping = collections.abc.Mapping

import tensorflow.compat.v1 as tf1  # noqa: E402

tf1.disable_v2_behavior()

# tf.contrib.layers.xavier_initializer(uniform=True) == glorot uniform
_layers = types.SimpleNamespace(
    xavier_initializer=lambda uniform=True, seed=None, dtype=None: (
        tf1.glorot_uniform_initializer(seed=seed)
        if uniform else tf1.glorot_normal_initializer(seed=seed)))
tf1.contrib = types.SimpleNamespace(layers=_layers)

sys.modules['tensorflow'] = tf1

if __name__ == '__main__':
    import runpy
    script = sys.argv[1]
    sys.argv = sys.argv[1:]
    sys.path.insert(0, 'src')
    runpy.run_path(script, run_name='__main__')
