"""Build the human-readable landing page for checkpoint-quality artifacts."""

from __future__ import annotations

import argparse
import html
import json
import math
from pathlib import Path


BASELINES = {
    "text": {"uniform_loss": math.log(176), "uniform_accuracy": 1 / 176,
             "unigram_loss": 4.2628446, "unigram_accuracy": 0.0735628},
    "image": {"uniform_loss": math.log(512), "uniform_accuracy": 1 / 512,
              "unigram_loss": 5.6287699, "unigram_accuracy": 0.0118953},
}

# Taken from the W&B paired validation at the selected epoch (512 held-out
# pairs).  These are the valid runs after the mixed-sequence routing fix.
INSTRUCTION_METRICS = {
    "dense_instruction (6rata6go, epoch 8)": {
        "text_to_image": (4.2731150, 0.1009674, 0.2132421, 0.4657234),
        "image_to_text": (3.7578921, 0.1107863, 0.2293902, 0.1615654),
    },
    "Tri-LoRA token-routed (a6nn0atl, epoch 8)": {
        "text_to_image": (4.2716207, 0.1012980, 0.2196662, 0.4489414),
        "image_to_text": (3.7654322, 0.1115017, 0.2190181, 0.2482561),
    },
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pretraining-metrics", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--marginal-samples", default="marginal_samples/index.html")
    parser.add_argument("--instruction-samples", default="instruction_samples/index.html")
    return parser.parse_args()


def number(value):
    return f"{value:.3f}"


def percent(value):
    return f"{100 * value:.1f}%"


def main():
    cli = parse_args()
    metrics = json.loads(Path(cli.pretraining_metrics).read_text())
    rows = []
    for modality in ("text", "image"):
        baseline = BASELINES[modality]
        rows.append(
            f"<tr><td>{modality.title()}</td><td>Uniform random</td><td>{number(baseline['uniform_loss'])}</td><td>{percent(baseline['uniform_accuracy'])}</td></tr>"
        )
        rows.append(
            f"<tr><td>{modality.title()}</td><td>Frequency-only unigram</td><td>{number(baseline['unigram_loss'])}</td><td>{percent(baseline['unigram_accuracy'])}</td></tr>"
        )
        for name, model in metrics["models"].items():
            value = model["marginal"][modality]
            rows.append(
                f"<tr><td>{modality.title()}</td><td>{html.escape(name)}</td><td>{number(value['loss'])}</td><td>{percent(value['accuracy'])}</td></tr>"
            )

    conditional = []
    for name, model in metrics["models"].items():
        paired = model["paired"]
        for direction in ("text_to_image", "image_to_text"):
            prefix = f"val/paired/{direction}/t1"
            conditional.append(
                "<tr>"
                f"<td>{html.escape(name)}</td><td>{direction.replace('_', ' → ')}</td>"
                f"<td>{number(paired[prefix + '/matched_loss'])}</td>"
                f"<td>{number(paired[prefix + '/shuffle_gap'])}</td>"
                f"<td>{number(paired[prefix + '/context_gain'])}</td></tr>"
            )

    instruction = []
    for name, values in INSTRUCTION_METRICS.items():
        for direction, (loss, accuracy, shuffle_gap, context_gain) in values.items():
            instruction.append(
                "<tr>"
                f"<td>{html.escape(name)}</td><td>{direction.replace('_', ' → ')}</td>"
                f"<td>{number(loss)}</td><td>{percent(accuracy)}</td>"
                f"<td>{number(shuffle_gap)}</td><td>{number(context_gain)}</td></tr>"
            )

    page = f'''<!doctype html><html><head><meta charset="utf-8"><title>CLEVR checkpoint-quality report</title>
<style>body{{font-family:system-ui,sans-serif;max-width:1250px;margin:32px auto;padding:0 22px;color:#1f2933}}table{{border-collapse:collapse;width:100%;margin:14px 0 38px}}th,td{{border:1px solid #cbd5e1;padding:9px;text-align:left}}th{{background:#e8eef7}}tr:nth-child(even){{background:#f8fafc}}.callout{{padding:14px 16px;border-left:4px solid #2563eb;background:#eff6ff;line-height:1.45}}a{{color:#0759bb;font-weight:600}}</style></head><body>
<h1>CLEVR checkpoint-quality report</h1>
<p class="callout"><b>How to read this.</b> Lower NLL is better; it measures the probability placed on the true masked token. Higher top-1 accuracy is better. “Unigram” uses only overall token frequency, so it is a meaningful non-neural baseline. For paired conditioning, a positive shuffle gap or context gain means the correct other modality reduces error; this is evidence of grounding rather than just a good marginal model.</p>
<h2>Held-out marginal denoising — 75% of target tokens masked</h2>
<p>Evaluation set: {metrics['num_samples']} paired validation scenes. Models receive no other modality for these rows.</p>
<table><thead><tr><th>Modality</th><th>System</th><th>NLL ↓</th><th>Top-1 accuracy ↑</th></tr></thead><tbody>{''.join(rows)}</tbody></table>
<h2>Held-out paired conditioning — 100% of target modality masked</h2>
<p>Matched context is the true paired modality. Shuffled context uses a different scene, and null context removes it. Positive gaps are better. These pretraining checkpoints were trained on deliberately unpaired modalities, so a near-zero matched-versus-shuffled gap (and no matched gain over null) is the expected result, not a failure of the evaluator.</p>
<table><thead><tr><th>Model</th><th>Direction</th><th>Matched NLL ↓</th><th>Matched vs shuffled NLL gap ↑</th><th>Matched vs null NLL gain ↑</th></tr></thead><tbody>{''.join(conditional)}</tbody></table>
<h2>Instruction tuning — W&amp;B validation at selected epoch</h2>
<p>Evaluation set: 512 held-out paired scenes; target modality fully masked. These are the valid corrected runs only. The older standard/adversarial instruction-LoRA runs are intentionally excluded because they used the earlier incorrect mixed-token routing.</p>
<table><thead><tr><th>Model</th><th>Direction</th><th>Matched NLL ↓</th><th>Top-1 accuracy ↑</th><th>Matched vs shuffled NLL gap ↑</th><th>Matched vs null NLL gain ↑</th></tr></thead><tbody>{''.join(instruction)}</tbody></table>
<p class="callout"><b>Practical reading.</b> The valid dense and token-routed Tri-LoRA instruction checkpoints are essentially tied numerically. The image-to-text examples are often attribute- and object-level plausible, while the 25-step text-to-image samples are visibly sparse and do not yet consistently satisfy requested object count, attributes, or relations. Treat text-to-image as early qualitative evidence, not a strong generation result.</p>
<h2>Qualitative samples</h2><ul>
<li><a href="{html.escape(cli.marginal_samples)}">Pretraining: unconditional text-only and image-only samples</a></li>
<li><a href="{html.escape(cli.instruction_samples)}">Instruction tuning: text → image and image → text samples</a></li>
</ul>
<p class="callout"><b>Important limitation.</b> Both pretraining checkpoints can denoise a partly visible image well, but their fully unconditional image samples are mostly background. This is expected from a masking objective with little/no completely masked-image exposure, and shows that marginal NLL is not a free-generation metric. Instruction tuning adds full-mask cases and produces some objects, but the text → image examples remain weak.</p>
</body></html>'''
    output = Path(cli.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(page)
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
