"""Standalone scientific figure from frozen support-audit tables."""

import argparse
from pathlib import Path

from jerqazyna.data.emit import write_json
from jerqazyna.data.emit_coverage import checked_manifest, read_json
from jerqazyna.features.cache import Digests, fingerprint


def plot_support(plan, assessment):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from matplotlib.patches import Rectangle

    plan, assessment = Path(plan), Path(assessment)
    d = Digests()
    checked_manifest(plan, "manifest.json", d)
    checked_manifest(assessment, "manifest.json", d)
    spec = {"plan_sha256": d.file(plan / "manifest.json"), "assessment_sha256": d.file(assessment / "manifest.json"),
            "implementation_sha256": d.file(Path(__file__)), "matplotlib": matplotlib.__version__}
    output = plan.parent / ("figures_" + fingerprint(spec)[:16])
    if (output / "manifest.json").exists():
        checked_manifest(output, "manifest.json", d)
        return output
    output.mkdir(parents=True, exist_ok=True)
    summary, audit = read_json(plan / "summary.json"), read_json(assessment / "summary.json")
    frame = pd.read_csv(plan / "labels_geometric_support.csv")
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), constrained_layout=True)
    folds = np.arange(len(summary["selected_positive_support"]["folds"]))
    total = [next(r["baseline_positive"] for r in audit["groups"]["fold"] if r["group"] == str(f)) for f in folds]
    available = [row["positive_labels"] for row in summary["all_catalog_date_support_after_known_constraints"]["folds"]]
    selected = [row["positive_labels"] for row in summary["selected_positive_support"]["folds"]]
    for shift, counts, color, label in [(-.25, total, "#a7b1ba", "Baseline: все положительные"),
                                       (0, available, "#287fb8", "≥2 дат в каталоге + известные ограничения"),
                                       (.25, selected, "#dc8840", "Замороженный выбор дат и рамок")]:
        axes[0].bar(folds + shift, counts, width=.24, color=color, label=label)
    minimum = read_json(plan / "manifest.json")["spec"]["config"]["minimum_positive_per_fold"]
    axes[0].axhline(minimum, color="#b83643", linestyle="--", label=f"Минимум: {minimum} объектов / fold")
    axes[0].set(xticks=folds, xlabel="Пространственный fold", ylabel="Положительных меток",
                title="Поддержка сравнения моделей на train")
    axes[0].legend(loc="upper right", fontsize=8)
    axes[0].set_ylim(0, max(total) + 3)
    positive = frame.baseline_positive
    masks = [(positive & ~frame.planning_target, "x", "#9b4b55", "Ограничение известной маски"),
             (positive & frame.planning_target & frame.all_catalog_complete_date_count.lt(2), "o", "#a7b1ba", "Менее двух дат"),
             (positive & frame.planning_target & frame.all_catalog_complete_date_count.ge(2)
              & ~frame.potential_two_dates_selected, "o", "#287fb8", "≥2 дат, вне замороженного выбора"),
             (positive & frame.potential_two_dates_selected, "o", "#dc8840", "В замороженном выборе"),
             (~positive, "+", "#7d858b", "Неизвестный статус")]
    for mask, marker, color, label in masks:
        rows = frame[mask]
        axes[1].scatter(rows.lon, rows.lat, marker=marker, c=color, s=38, label=label, zorder=3)
    for item in summary["pilot_configs"]:
        w, s, e, n = item["area"]["bbox"]
        axes[1].add_patch(Rectangle((w, s), e-w, n-s, fill=False, lw=.7, color="#dc8840", alpha=.6))
    axes[1].set(xlabel="Долгота, °E", ylabel="Широта, °N", title="Геометрия наблюдений и план рамок; QA неизвестно")
    axes[1].legend(loc="lower left", fontsize=8)
    axes[1].grid(alpha=.2)
    fig.suptitle("EMIT: пригодность расширения для пяти fold пока не подтверждена\n"
                 "Летние даты 09.08.2022–19.10.2026; новые научные файлы не загружались", fontsize=14)
    fig.savefig(output / "train_support.png", dpi=160)
    plt.close(fig)
    write_json(output / "manifest.json", {"spec": spec, "files": {"train_support.png": d.file(output / "train_support.png")}})
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("plan")
    parser.add_argument("assessment")
    args = parser.parse_args()
    print(plot_support(args.plan, args.assessment))


if __name__ == "__main__":
    main()
