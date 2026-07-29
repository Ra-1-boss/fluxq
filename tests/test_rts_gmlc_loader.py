"""
Tests for fluxq.data.loader.load_rts_gmlc_generators().

Since gen.csv is a large external dataset (not bundled with FluxQ), these
tests look for a local RTS-GMLC clone and skip gracefully if it isn't
found -- rather than failing on machines that haven't downloaded it.

Set the RTS_GMLC_PATH environment variable to override the default search
location, e.g. (PowerShell):
    $env:RTS_GMLC_PATH = "C:\\path\\to\\RTS-GMLC\\RTS_Data\\SourceData\\gen.csv"

Run with:
    python -m pytest tests/test_rts_gmlc_loader.py -v
"""
import os
import sys
from pathlib import Path
from typing import Optional

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fluxq.data.loader import load_rts_gmlc_generators
from fluxq.data.grid_builder import make_rts_gmlc_sample


def _find_gen_csv() -> Optional[Path]:
    """
    Look for a local RTS-GMLC gen.csv, in order:
      1. RTS_GMLC_PATH environment variable, if set and it exists.
      2. The default sibling-folder location used in this project's real
         setup: ../RTS-GMLC/RTS_Data/SourceData/gen.csv, relative to the
         repo root (i.e. RTS-GMLC cloned alongside fluxq, not inside it).
    Returns None if not found anywhere -- tests skip rather than fail.
    """
    env_path = os.environ.get("RTS_GMLC_PATH")
    if env_path and Path(env_path).exists():
        return Path(env_path)

    repo_root = Path(__file__).resolve().parent.parent
    default_path = repo_root.parent / "RTS-GMLC" / "RTS_Data" / "SourceData" / "gen.csv"
    if default_path.exists():
        return default_path

    return None


GEN_CSV_PATH = _find_gen_csv()

pytestmark = pytest.mark.skipif(
    GEN_CSV_PATH is None,
    reason=(
        "RTS-GMLC gen.csv not found. Clone github.com/GridMod/RTS-GMLC "
        "as a sibling of this repo, or set the RTS_GMLC_PATH environment "
        "variable, to run these tests."
    ),
)


class TestLoadRTSGMLCGenerators:
    def test_loads_without_error(self):
        gens = load_rts_gmlc_generators(GEN_CSV_PATH)
        assert len(gens) > 0

    def test_returns_72_dispatchable_generators(self):
        """
        72 was independently confirmed by hand via PowerShell
        (Where-Object Fuel -in Oil,Coal,NG | Count) *before* this loader
        was written -- see docs/research-log.md, 2026-07-26 entry. This
        is a real, independently-verified number, not something the
        loader itself invented.
        """
        gens = load_rts_gmlc_generators(GEN_CSV_PATH)
        assert len(gens) == 72

    def test_all_generators_pass_validation(self):
        """
        Every Generator's own __post_init__ already validates P_max>0,
        c_fuel>=0, MUT>=1, etc. If any row produced an invalid Generator,
        construction itself would already have raised before this test
        even runs. This test documents that expectation explicitly.
        """
        gens = load_rts_gmlc_generators(GEN_CSV_PATH)
        for g in gens:
            assert g.P_max > 0
            assert g.c_fuel >= 0
            assert g.SU_cost >= 0
            assert g.SD_cost >= 0
            assert g.MUT >= 1
            assert g.MDT >= 1

    def test_cross_validated_against_hand_verified_sample(self):
        """
        The 5 generators in make_rts_gmlc_sample() were derived and
        checked by hand independently on 2026-07-26 -- including catching
        and correcting a real arithmetic slip before it reached code.

        The general loader, applied to the real file, must reproduce
        those exact same 5 generators' values. This is the core
        validation strategy for this loader: cross-check it against
        already-verified ground truth, rather than hand-checking all
        72 rows (which would just reintroduce the transcription risk
        this loader exists to remove).

        Note: P_min is intentionally NOT compared here. The hand-verified
        sample never captured P_min values by hand, so it defaults to 0.0
        there, while the general loader correctly reads real P_min values
        from the CSV. That's a known, deliberate gap -- see the research
        log -- not a bug in either function.
        """
        loaded = {g.name: g for g in load_rts_gmlc_generators(GEN_CSV_PATH)}
        hand_verified = {g.name: g for g in make_rts_gmlc_sample()}

        assert set(hand_verified.keys()).issubset(set(loaded.keys())), (
            "All 5 hand-verified generators should appear in the full load."
        )

        for name, expected in hand_verified.items():
            actual = loaded[name]
            assert abs(actual.P_max - expected.P_max) < 1e-6, (
                f"{name}: P_max mismatch"
            )
            assert abs(actual.c_fuel - expected.c_fuel) < 0.01, (
                f"{name}: c_fuel expected ~{expected.c_fuel:.4f}, "
                f"loader gave {actual.c_fuel:.4f}"
            )
            assert abs(actual.SU_cost - expected.SU_cost) < 0.01, (
                f"{name}: SU_cost expected ~{expected.SU_cost:.2f}, "
                f"loader gave {actual.SU_cost:.2f}"
            )
            assert abs(actual.SD_cost - expected.SD_cost) < 1e-6, (
                f"{name}: SD_cost mismatch"
            )
            assert actual.MUT == expected.MUT, (
                f"{name}: MUT expected {expected.MUT}, loader gave {actual.MUT}"
            )
            assert actual.MDT == expected.MDT, (
                f"{name}: MDT expected {expected.MDT}, loader gave {actual.MDT}"
            )

    def test_excludes_non_dispatchable_fuel_types(self):
        """
        Sanity check that filtering actually happened -- total dispatchable
        count (72) should be well below the full file's 158 generators.
        """
        gens = load_rts_gmlc_generators(GEN_CSV_PATH)
        assert len(gens) < 158

    def test_custom_fuel_type_filter_coal_only(self):
        """
        Filtering to Coal alone should return exactly 16 -- independently
        confirmed by hand via PowerShell Group-Object earlier in the
        research log, same source of truth as the 72-generator count.
        """
        gens = load_rts_gmlc_generators(GEN_CSV_PATH, fuel_types=("Coal",))
        assert len(gens) == 16

    def test_raises_clear_error_on_missing_file(self):
        with pytest.raises(FileNotFoundError):
            load_rts_gmlc_generators("this/path/does/not/exist/gen.csv")
