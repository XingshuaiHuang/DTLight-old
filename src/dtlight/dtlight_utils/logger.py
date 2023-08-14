"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the CC BY-NC license found in the
LICENSE.md file in the root directory of this source tree.
"""

from datetime import datetime
import os


class Logger:
    def __init__(self, variant, signal):
        self.signal = signal
        self.log_path = self.create_log_path(variant)
        os.makedirs(self.log_path, exist_ok=True)
        # print(f"Experiment log path: {self.log_path}")

    def create_log_path(self, variant):
        now = datetime.now().strftime("%Y.%m.%d/%H%M%S")
        exp_name = variant["exp_name"]
        prefix = variant["save_dir"]
        map = variant["map"]
        # return f"{prefix}/{now}-{exp_name}"
        return f"{prefix}/{map}/{self.signal}"

    def log_metrics(self, outputs, iter_num, total_transitions_sampled, writer, verbose=True):
        print("=" * 50)
        print(f"Iteration {iter_num}")
        for k, v in outputs.items():
            print(f"{k}: {v}")
            if writer:
                writer.add_scalar(k, v, iter_num)
                if k == "evaluation/return_mean_gm":
                    writer.add_scalar(
                        "evaluation/return_vs_samples",
                        v,
                        total_transitions_sampled,
                    )
