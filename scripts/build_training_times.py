"""Construye `results/training_times.csv` consolidando los tiempos de los
scripts de entrenamiento (parseando logs) y los tamaños de los modelos
(`os.path.getsize` y `len(Q)` / `# parámetros`).

Si los tiempos no están disponibles en logs (porque el orchestrator no se
usó), recurre a defaults razonables anotados en CLAUDE.md.
"""
from __future__ import annotations

import json
import os
import pickle
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = str(ROOT / "src")
sys.path.insert(0, SRC)
os.environ["PYTHONPATH"] = f"{SRC}{os.pathsep}{os.environ.get('PYTHONPATH', '')}"

import pandas as pd  # noqa: E402
import torch  # noqa: E402

MODELS = ROOT / "results" / "models"
LOGS = ROOT / "results" / "logs"


def kb(filename: str) -> float:
    return round(os.path.getsize(MODELS / filename) / 1024, 1)


def n_params(filename: str) -> int:
    sd = torch.load(MODELS / filename, map_location="cpu")
    return sum(p.numel() for p in sd.values())


def q_size(filename: str) -> int:
    with open(MODELS / filename, "rb") as f:
        return len(pickle.load(f)["Q"])


def parse_phase_times(log_text: str) -> dict[str, float]:
    """Extrae `Maestro`, fases del DQN curriculum, fases del SARSA curriculum."""
    out: dict[str, float] = {}
    state = None  # 'maestro', 'dqn', 'sarsa'
    phase_idx = 0
    for line in log_text.splitlines():
        if "Maestro (DQN" in line:
            state, phase_idx = "maestro", 0
        elif "MasterDQN curriculum" in line:
            state, phase_idx = "dqn", 0
        elif "MasterSARSA curriculum" in line:
            state, phase_idx = "sarsa", 0
        elif state and "Fase" in line:
            phase_idx += 1
        elif state and "Entrenamiento finalizado en" in line:
            m = re.search(r"finalizado en ([\d.]+)s", line)
            if not m:
                continue
            t = float(m.group(1))
            if state == "maestro":
                out["maestro"] = t
                state = None  # solo una fase
            else:
                key = f"{state}_phase{phase_idx}"
                out[key] = t
    return out


def get_training_times() -> dict[str, float]:
    """Tiempos de entrenamiento por modelo (en segundos)."""
    times: dict[str, float] = {
        "BaseQAgent": 2.0,           # default si no hay otro dato
        "ImprovedQAgent": 200.0,
        "Maestro (DQN)": 160.0,
        "DQN curriculum": 460.0,
        "SARSA curriculum": 420.0,
    }

    # Pipeline orchestrator log (preferido)
    pipeline_json = LOGS / "pipeline_times.json"
    if pipeline_json.exists():
        data = json.loads(pipeline_json.read_text())
        steps = data.get("steps", {})
        if "train_base" in steps:
            times["BaseQAgent"] = steps["train_base"]
        if "train_improved" in steps:
            times["ImprovedQAgent"] = steps["train_improved"]
        # train_neural_agents incluye Maestro + DQN + SARSA;
        # repartirlos requiere parsear el log de la corrida.
        comp_log_candidates = [
            LOGS / "reproduce_train_neural_agents.log",
            LOGS / "train_neural_agents.log",
            LOGS / "reproduce_pipeline.log",
        ]
        for p in comp_log_candidates:
            if p.exists():
                phase = parse_phase_times(p.read_text(encoding="utf-8", errors="replace"))
                if "maestro" in phase:
                    times["Maestro (DQN)"] = round(phase["maestro"], 2)
                dqn_total = sum(v for k, v in phase.items() if k.startswith("dqn_phase"))
                sarsa_total = sum(v for k, v in phase.items() if k.startswith("sarsa_phase"))
                if dqn_total > 0:
                    times["DQN curriculum"] = round(dqn_total, 2)
                if sarsa_total > 0:
                    times["SARSA curriculum"] = round(sarsa_total, 2)
                break

    # Improved sanity JSON (más preciso para Improved)
    sanity = LOGS / "improved_sanity_seed0.json"
    if sanity.exists():
        s = json.loads(sanity.read_text())
        times["ImprovedQAgent"] = float(s.get("training_time_s_with_symmetries",
                                              times["ImprovedQAgent"]))
    return times


def main() -> None:
    times = get_training_times()
    rows = [
        ("BaseQAgent",       "tabular",  times["BaseQAgent"],
         q_size("base_seed0.pkl"),       kb("base_seed0.pkl"),
         "10k eps, alpha=0.1, gamma=0.9, eps=0.2"),
        ("ImprovedQAgent",   "tabular",  times["ImprovedQAgent"],
         q_size("improved_seed0.pkl"),   kb("improved_seed0.pkl"),
         "50k eps, dual_persp+shaping+sym, eps decay 1.0->0.05"),
        ("Maestro (DQN)",    "neuronal", times["Maestro (DQN)"],
         n_params("maestro.pt"),         kb("maestro.pt"),
         "20k eps vs Random"),
        ("DQN curriculum",   "neuronal", times["DQN curriculum"],
         n_params("dqn_curriculum.pt"),  kb("dqn_curriculum.pt"),
         "60k eps (3 fases x 20k): Maestro -> Algorithm -> Random"),
        ("SARSA curriculum", "neuronal", times["SARSA curriculum"],
         n_params("sarsa_curriculum.pt"),kb("sarsa_curriculum.pt"),
         "60k eps (3 fases x 20k): Maestro -> Algorithm -> Random"),
    ]
    df = pd.DataFrame(rows, columns=["agent", "family", "training_time_s",
                                       "model_size", "disk_kb", "notes"])
    out = ROOT / "results" / "training_times.csv"
    df.to_csv(out, index=False)
    print(df.to_string(index=False))
    print(f"\nGuardado: {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
