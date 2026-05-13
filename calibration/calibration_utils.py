from __future__ import annotations
from pathlib import Path
import pandas as pd
from typing import Dict, Iterable, Any, Tuple

from calibration import thermistor_calibration

def _to_pair(offset_result: Iterable[Any]) -> Tuple[float, float]:
    vals = list(offset_result)
    if len(vals) >= 3:
        return float(vals[0]), float(vals[2])  # (black, ..., white)
    if len(vals) >= 2:
        return float(vals[0]), float(vals[1])  # (black, white)
    raise ValueError("Unexpected offset return format")

def compute_and_save_offsets(
    logger_files: Dict[int, str],
    reference_by_logger: Dict[int, Any],
    *,
    out_csv: str | Path,
    plots_dir: str | Path | None = None
) -> pd.DataFrame:
    # Lazy import to avoid circulars at module import time
    from src.thermistor_plotting import ThermistorDataPlotter

    rows = []
    plots_path = Path(plots_dir) if plots_dir else None
    if plots_path:
        plots_path.mkdir(parents=True, exist_ok=True)

    for lg, fpath in sorted(logger_files.items()):
        ref = reference_by_logger.get(lg)
        if ref is None:
            raise ValueError(f"No reference dataset provided for logger {lg}")
        plotter = ThermistorDataPlotter(fpath, delimiter=",")
        title = f"Logger #{lg} - 0deg offset in ice bath"
        offsets = plotter.plot_ntc_icebath_calibration(
            ref, savepath=(str(plots_path) + "/") if plots_path else None, title=title
        )
        black, white = _to_pair(offsets)
        rows.append({"Logger": lg, "Black Probe Offset": black, "White Probe Offset": white})

    df = pd.DataFrame(rows).sort_values("Logger")
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    return df