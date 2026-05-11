"""Corrida completa del ablation study.

5 variantes × 10 seeds × 50k episodios + 4 evaluaciones × 500 partidas.
Tiempo estimado en máquina con 8-10 cores: ~25 min.
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = str(ROOT / "src")
sys.path.insert(0, SRC)
existing = os.environ.get("PYTHONPATH", "")
if SRC not in existing.split(os.pathsep):
    os.environ["PYTHONPATH"] = f"{SRC}{os.pathsep}{existing}" if existing else SRC

from new.ablations import run_ablation_study  # noqa: E402


def main() -> None:
    summary = run_ablation_study(
        seeds=list(range(10)),
        episodes=50_000,
        eval_games=500,
        agent_role=1,
        n_workers=10,
        output_dir=ROOT / "results",
    )
    print()
    print("=== Summary (UTF-8 forced) ===")
    # Windows console usa cp1252 por defecto y rompe con caracteres como
    # U+2212 ("MINUS SIGN") presente en variant names ("- decay" etc.).
    # Re-emitir con encoding explicito.
    sys.stdout.buffer.write(
        summary.to_string(index=False).encode("utf-8", errors="replace") + b"\n"
    )


if __name__ == "__main__":
    main()
