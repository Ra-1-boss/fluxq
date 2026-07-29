"""
Data loaders for real generator datasets.

Currently supports RTS-GMLC (Reliability Test System - Grid Modernization
Laboratory Consortium), a real, publicly available power system test dataset
built by NREL/GMLC: https://github.com/GridMod/RTS-GMLC

See docs/glossary.md for the full derivation of every formula used here, the
unit-convention check that confirmed heat rate is in Btu/kWh (not MMBTU/MWh),
and the reasoning behind which fuel types are included or excluded from
FluxQ's binary on/off Unit Commitment model.
"""
from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Dict, List, Sequence, Union

from .grid_builder import Generator


# ---------------------------------------------------------------------------
# RTS-GMLC
# ---------------------------------------------------------------------------

# Fuel types treated as dispatchable (a real on/off commitment decision) in
# FluxQ's Phase 0 binary UC model. Everything else in gen.csv (Nuclear,
# Hydro, Solar, Wind, Sync_Cond, Storage) is excluded -- see docs/glossary.md
# for the reasoning behind each exclusion.
DISPATCHABLE_FUELS = ("Oil", "Coal", "NG")


def load_rts_gmlc_generators(
    gen_csv_path: Union[str, Path],
    fuel_types: Sequence[str] = DISPATCHABLE_FUELS,
    verbose: bool = False,
) -> List[Generator]:
    """
    Load dispatchable generators directly from a real RTS-GMLC gen.csv file.

    Applies the exact same derivation formulas that were worked out and
    verified by hand for a 5-generator sample
    (``grid_builder.make_rts_gmlc_sample()``), generalized here to every
    matching row in the file -- removing the manual-transcription risk of
    hand-copying values for more than a handful of generators.

        c_fuel  = HR_avg_0 (Btu/kWh) * Fuel Price ($/MMBTU) / 1000
        SU_cost = Start Heat Cold (MBTU) * Fuel Price ($/MMBTU)
                  + Non Fuel Start Cost ($)
        SD_cost = Non Fuel Shutdown Cost ($)
        MUT/MDT = ceil(Min Up/Down Time Hr)  -- rounded UP, never down, so a
                  generator's real commitment constraint is never
                  under-represented.

    See docs/glossary.md for the full derivation, including the unit-
    convention sanity check that confirmed these formulas are correct.

    Parameters
    ----------
    gen_csv_path : str or Path
        Path to RTS-GMLC's ``RTS_Data/SourceData/gen.csv``. This file is
        not bundled with FluxQ -- clone the dataset separately from
        github.com/GridMod/RTS-GMLC and point this at the real file.
    fuel_types : sequence of str, optional
        Which ``Fuel`` column values count as dispatchable and get turned
        into Generator objects. Default: ``("Oil", "Coal", "NG")``.
        Nuclear, Hydro, Solar, Wind, Sync_Cond, and Storage are excluded
        by default -- see docs/glossary.md for why each one is excluded.
    verbose : bool, optional
        If True, print a summary of how many rows were read, kept, and
        excluded (broken down by fuel type). Default False.

    Returns
    -------
    list of Generator

    Raises
    ------
    FileNotFoundError
        If ``gen_csv_path`` does not exist.
    ValueError
        If a row's numeric fields can't be parsed as numbers -- the error
        message identifies the offending ``GEN UID`` and field name so
        it's easy to trace back to the exact row in the source CSV.
    """
    path = Path(gen_csv_path)
    if not path.exists():
        raise FileNotFoundError(
            f"RTS-GMLC gen.csv not found at {path}. "
            f"Clone the dataset from github.com/GridMod/RTS-GMLC and "
            f"point gen_csv_path at RTS_Data/SourceData/gen.csv."
        )

    fuel_set = set(fuel_types)
    generators: List[Generator] = []
    excluded_counts: Dict[str, int] = {}
    total_rows = 0

    # utf-8-sig handles an optional BOM (common in Windows-exported CSVs)
    # gracefully whether or not one is actually present.
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            total_rows += 1
            fuel = row["Fuel"].strip()

            if fuel not in fuel_set:
                excluded_counts[fuel] = excluded_counts.get(fuel, 0) + 1
                continue

            uid = row["GEN UID"]

            def _float(field: str) -> float:
                raw = row[field]
                try:
                    return float(raw)
                except (ValueError, TypeError):
                    raise ValueError(
                        f"Generator '{uid}': could not parse field "
                        f"'{field}' = {raw!r} as a number."
                    )

            p_max = _float("PMax MW")
            p_min = _float("PMin MW")
            fuel_price = _float("Fuel Price $/MMBTU")
            heat_rate = _float("HR_avg_0")
            start_heat_cold = _float("Start Heat Cold MBTU")
            non_fuel_start = _float("Non Fuel Start Cost $")
            non_fuel_shutdown = _float("Non Fuel Shutdown Cost $")
            mut_hr = _float("Min Up Time Hr")
            mdt_hr = _float("Min Down Time Hr")

            c_fuel = heat_rate * fuel_price / 1000.0
            su_cost = start_heat_cold * fuel_price + non_fuel_start
            sd_cost = non_fuel_shutdown

            generators.append(
                Generator(
                    name=uid,
                    P_max=p_max,
                    P_min=p_min,
                    c_fuel=c_fuel,
                    SU_cost=su_cost,
                    SD_cost=sd_cost,
                    MUT=math.ceil(mut_hr),
                    MDT=math.ceil(mdt_hr),
                )
            )

    if verbose:
        print(f"RTS-GMLC loader: {total_rows} total rows in {path.name}")
        print(
            f"  Kept {len(generators)} dispatchable generators "
            f"(fuel types: {', '.join(sorted(fuel_set))})"
        )
        if excluded_counts:
            print("  Excluded:")
            for fuel, count in sorted(excluded_counts.items(), key=lambda kv: -kv[1]):
                print(f"    {fuel:<12} {count}")

    return generators
