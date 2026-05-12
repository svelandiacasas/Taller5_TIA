"""Re-corre el pipeline completo desde cero, midiendo tiempos por etapa.

Uso:
    python scripts/reproduce_pipeline.py

Salida: imprime una tabla de tiempos al final, también guardada en
`results/logs/pipeline_times.json`.

Útil para Fase 8 (validación de reproducibilidad end-to-end) y para que un
colaborador clonando el repo regenere todos los artefactos en orden.
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = str(ROOT / "src")
os.environ["PYTHONPATH"] = f"{SRC}{os.pathsep}{os.environ.get('PYTHONPATH', '')}"

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

STEPS = [
    ("pytest",                 [sys.executable, "-m", "pytest", "-q"]),
    ("train_base",             [sys.executable, "-u", "scripts/train_base.py"]),
    ("train_improved",         [sys.executable, "-u", "scripts/train_improved.py"]),
    ("train_companero_agents", [sys.executable, "-u", "scripts/train_companero_agents.py"]),
    ("build_training_times",   [sys.executable, "-u", "scripts/build_training_times.py"]),
    ("run_diagnostics",        [sys.executable, "-u", "scripts/run_diagnostics.py"]),
    ("run_ablations",          [sys.executable, "-u", "scripts/run_ablations.py"]),
    ("run_robustness",         [sys.executable, "-u", "scripts/run_robustness.py"]),
    # nbconvert: usar el comando directo, no `python -m jupyter` (que no funciona en
    # esta versión del entorno conda — el módulo `jupyter` no es importable, hay que
    # invocar el script de consola).
    ("nbconvert_notebook",     ["jupyter", "nbconvert",
                                 "--to", "notebook", "--execute",
                                 "notebooks/taller_rl_triqui.ipynb",
                                 "--output", "taller_rl_triqui.ipynb"]),
]


def _per_step_log_path(step_name: str) -> Path:
    return ROOT / "results" / "logs" / f"reproduce_{step_name}.log"


def main() -> None:
    times: dict[str, float] = {}
    failures: list[str] = []
    overall_start = time.time()
    print(f"=== reproduce_pipeline.py — {len(STEPS)} etapas ===\n", flush=True)

    (ROOT / "results" / "logs").mkdir(parents=True, exist_ok=True)
    for name, cmd in STEPS:
        print(f"--- [{name}] {' '.join(str(x) for x in cmd[2:])}", flush=True)
        t0 = time.time()
        log_path = _per_step_log_path(name)
        # Pipeamos a un log por-etapa además de stdout. Permite que el notebook
        # parsee, p.ej., reproduce_train_companero_agents.log para sus métricas.
        with open(log_path, "w", encoding="utf-8", errors="replace") as f:
            p = subprocess.Popen(cmd, cwd=str(ROOT),
                                  stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                  text=True, encoding="utf-8", errors="replace",
                                  bufsize=1)
            for line in p.stdout:
                sys.stdout.write(line)
                sys.stdout.flush()
                f.write(line)
            p.wait()
        elapsed = time.time() - t0
        times[name] = round(elapsed, 2)
        ok = p.returncode == 0
        status = "OK " if ok else "FAIL"
        print(f"--- [{name}] {status} en {elapsed:.1f}s ({elapsed/60:.2f} min)\n", flush=True)
        if not ok:
            failures.append(name)
            # Continúa con las demás etapas para tener un cuadro completo.

    overall = time.time() - overall_start
    summary = {
        "total_seconds": round(overall, 2),
        "total_minutes": round(overall / 60, 2),
        "steps": times,
        "failures": failures,
    }
    out = ROOT / "results" / "logs" / "pipeline_times.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))

    print("=" * 60)
    print(f"TOTAL: {overall:.1f}s ({overall/60:.2f} min)")
    print("=" * 60)
    print(f"{'Etapa':<25s} {'Tiempo (s)':>12s} {'Tiempo (min)':>14s}")
    print("-" * 60)
    for name, t in times.items():
        marker = " ←FAIL" if name in failures else ""
        print(f"{name:<25s} {t:>12.2f} {t/60:>14.2f}{marker}")
    print("-" * 60)
    print(f"\nResumen JSON: {out.relative_to(ROOT)}")
    if failures:
        print(f"\n!! FALLARON: {failures}")
        sys.exit(1)


if __name__ == "__main__":
    main()
