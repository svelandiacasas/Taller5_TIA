"""Corre los diagnósticos del agente base y deja figuras + CSVs en `results/`.

Uso:
    python scripts/run_diagnostics.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from diagnostics import run_all_diagnostics  # noqa: E402


def main() -> None:
    paths = run_all_diagnostics(
        seeds=list(range(10)),
        episodes=10_000,
        eval_every=500,
        eval_episodes=200,
        output_dir=ROOT / "results",
    )
    print()
    print("Figuras generadas:")
    for name, p in paths.items():
        print(f"  {name}: {p.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
