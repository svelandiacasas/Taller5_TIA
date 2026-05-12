# Taller de Aprendizaje por Refuerzo — Triqui

Entrega del **Taller de Aprendizaje por Refuerzo** del curso *Técnicas de Inteligencia
Artificial* (Ing. Mecatrónica · UNAL Bogotá · abril 2026).

## Descripción del proyecto

Este repositorio responde la pregunta del taller: *¿qué enfoque de aprendizaje
permite aprender mejor a jugar triqui y por qué?* A partir del agente Q-Learning
tabular de referencia presentado en la sesión del curso, el trabajo
**diagnostica sus tres patologías estructurales** (sección 5 del notebook) y
construye un agente mejorado (`ImprovedQAgent`) que las corrige. Compara
sistemáticamente — multi-semilla, con intervalos de confianza bootstrap, ablation
study completo y análisis de robustez — contra los agentes neuronales DQN y
SARSA con curriculum learning, y contra el oráculo perfecto Minimax con
memoización.

**Tesis del trabajo**: el factor crítico para aprender bien triqui no es la
familia del algoritmo (tabular vs neuronal) sino la corrección de la asignación
de crédito en self-play. La patología del agente de referencia — **100 % de las
jugadas ganadoras de O quedan con `Q < 0`** por una asimetría de signo en la
recompensa terminal — se resuelve por construcción con dual-perspective (Bellman
tipo negamax). El ablation cuantifica que `−dual_perspective` es la variante
devastadora (60 % de derrotas vs Minimax) mientras que `−shaping`, `−symmetries`
y `−decay` tienen impactos marginales o no significativos. `ImprovedQAgent`
tabular alcanza **0 % de derrotas vs Minimax** en 50 000 episodios, igualando a
DQN/SARSA pero a ~3× menor costo computacional.

## Estructura del repositorio

```
.
├── README.md                   # este archivo
├── conftest.py                 # bootstrap de pytest (sys.path + PYTHONPATH)
├── requirements.txt            # dependencias pip
│
├── notebooks/
│   └── taller_rl_triqui.ipynb  # ENTREGABLE PRINCIPAL (14 secciones)
│
├── scripts/
│   ├── train_base.py             # entrena BaseQAgent
│   ├── train_improved.py         # entrena ImprovedQAgent + sanity
│   ├── train_neural_agents.py    # entrena Maestro + DQN + SARSA (curriculum)
│   ├── build_training_times.py   # consolida tiempos en results/training_times.csv
│   ├── run_diagnostics.py        # 3 figuras de patologías del Base
│   ├── run_ablations.py          # 5 variantes × 10 seeds + bootstrap
│   ├── run_robustness.py         # 4 agentes × 3 oponentes × K∈{0,2,4} × 2 roles
│   ├── measure_parallel_speedup.py
│   └── reproduce_pipeline.py     # corre todo en orden con cronómetro
│
├── src/
│   ├── __init__.py
│   ├── triqui.py                  # motor del juego
│   ├── triqui_algorithm.py        # agente heurístico de referencia (Algorithm)
│   ├── master_RL.py               # MasterDQN/SARSA + MotorRL + ReplayBuffer
│   ├── triqui_train.py            # entorno gym-like (Train)
│   ├── triqui_championship.py     # torneo round-robin (Championship)
│   ├── seeds.py                   # set_seed
│   ├── rng_utils.py               # isolated_rng() context manager
│   ├── minimax.py                 # MinimaxAgent + memoización (oráculo)
│   ├── symmetry.py                # canonicalización D₄
│   ├── base_q_agent.py            # BaseQAgent + TabularToTorchAdapter
│   ├── improved_q_agent.py        # ImprovedQAgent (con todos los toggles)
│   ├── diagnostics.py             # las 3 patologías del Base
│   ├── evaluation.py              # eval funcs + análisis de robustez
│   ├── ablations.py               # orquestador del ablation study
│   ├── multi_seed.py              # paralelización con multiprocessing
│   └── bootstrap.py               # IC bootstrap + paired bootstrap test
│
├── tests/
│   ├── test_triqui.py             # Simulator (no pytest)
│   ├── test_minimax.py            # 10 tests
│   ├── test_symmetry.py           # 9 tests
│   ├── test_base_q_agent.py       # 17 tests (incluye fidelidad bit a bit)
│   ├── test_improved_q_agent.py   # 18 tests
│   ├── test_diagnostics.py        # 7 tests (smoke con tmp_path)
│   ├── test_multi_seed.py         # 5 tests (paralelo == secuencial)
│   ├── test_bootstrap.py          # 13 tests (incl. calibración bajo H0)
│   └── test_ablations.py          # 6 tests (smoke 30s)
│
└── results/                       # generado por los scripts; .gitkeep en git
    ├── models/                    # *.pkl (tabulares) + *.pt (neuronales) — gitignored
    ├── logs/                      # CSVs y JSONs de trazas
    ├── figures/                   # PNGs (incluye los del notebook)
    ├── ablation_raw.csv
    ├── ablation_summary.csv
    ├── robustness_raw.csv
    └── training_times.csv
```

## Instalación

Probado en **Windows 11** con conda Python 3.10. El motor del juego y los
agentes tabulares son pure-Python; DQN/SARSA usan PyTorch en CPU (no GPU — el
tamaño del problema lo hace contraproducente).

```bash
conda create -n triqui-rl python=3.10 -y
conda activate triqui-rl
pip install -r requirements.txt
```

`requirements.txt` incluye `torch torchvision torchaudio numpy pandas
scikit-learn matplotlib seaborn pytest` y, opcionalmente, `--extra-index-url
https://download.pytorch.org/whl/cu124` para tener disponible la opción GPU
(este proyecto no la usa).

## Reproducción end-to-end

Ejecutar el pipeline completo desde un clon limpio del repo:

```bash
python scripts/reproduce_pipeline.py
```

Este script encadena las 9 etapas en el orden correcto y mide tiempos por
etapa. Tiempos consolidados en una corrida limpia desde `results/` vacío
(Intel i7-14700K · CPU only · Windows 11 · Python 3.10, datos en
`results/logs/pipeline_times.json`):

| # | Etapa | Comando | Tiempo (s) | Tiempo (min) |
|---|-------|---------|-----------:|-------------:|
| 1 | Tests (85 tests) | `pytest -q`                                              |   139.6 |  2.33 |
| 2 | BaseQAgent (1 seed, 10k eps) | `scripts/train_base.py`                       |     2.3 |  0.04 |
| 3 | ImprovedQAgent (1 seed, 50k eps) | `scripts/train_improved.py`               |   236.9 |  3.95 |
| 4 | Maestro + DQN + SARSA curriculum | `scripts/train_neural_agents.py`          |   859.2 | 14.32 |
| 5 | Tabla de eficiencia | `scripts/build_training_times.py`                      |     3.0 |  0.05 |
| 6 | Diagnósticos del Base (10 seeds) | `scripts/run_diagnostics.py`              |    56.1 |  0.94 |
| 7 | Ablation 5×10 seeds × 50k eps + bootstrap | `scripts/run_ablations.py`       |   953.1 | 15.88 |
| 8 | Robustez 4×3×5×6×200 partidas | `scripts/run_robustness.py`                  |    55.1 |  0.92 |
| 9 | Ejecutar el notebook | `jupyter nbconvert --execute …`                       |    19.3 |  0.32 |
| **TOTAL** | | | **2 324.6** | **38.74** |

> Los tiempos exactos de cada corrida quedan en
> `results/logs/pipeline_times.json` después de ejecutar
> `scripts/reproduce_pipeline.py`.

Las dos etapas dominantes son `train_neural_agents` y `run_ablations`. Esta
última usa `multiprocessing.Pool` con 10 workers internamente (50 jobs de 50k
eps cada uno, ~17 min en paralelo); secuencial sería ~140 min, así que el
speedup observado es ~9× (cercano al máximo de 10 workers).

Si **no** se quiere regenerar todo (proceso completo ~40 min), se pueden
ejecutar solo subconjuntos:

```bash
python scripts/train_base.py        # ~2 s          → results/models/base_seed0.pkl
python scripts/run_diagnostics.py   # ~1 min        → 3 figuras + 3 CSVs
```

Estos dos generan los datos para las secciones 4 y 5 del notebook (las más
importantes desde el punto de vista del taller — el diagnóstico de las
patologías del agente base).

### Si solo se quiere ver el notebook

El notebook `notebooks/taller_rl_triqui.ipynb` se commitea **ejecutado** (con
outputs visibles). Abrirlo con:

```bash
jupyter notebook notebooks/taller_rl_triqui.ipynb
```

Para **re-ejecutarlo** desde cero (rápido, asume que `results/` está poblado):

```bash
jupyter nbconvert --to notebook --execute notebooks/taller_rl_triqui.ipynb --output taller_rl_triqui.ipynb
```

Si `results/` no está poblado, la celda 0 (Setup) lanza un `FileNotFoundError`
con instrucciones de qué scripts ejecutar (en orden).

## Suite de tests

```bash
python -m pytest -v
```

85 tests verificando:

- Determinismo (misma seed → mismos resultados, paralelo == secuencial bit a bit).
- **Fidelidad bit a bit con el agente de referencia** (test crítico:
  `tests/test_base_q_agent.py::test_base_q_agent_matches_professor_code_exactly`).
- Optimalidad de minimax (Minimax-vs-Minimax = 100 % empates).
- Invariancia D₄ de la canonicalización.
- Calibración del bootstrap test bajo H₀ (rejection rate ~5 % en 50 trials).
- Reproducibilidad de las corridas paralelas (Q-tables paralelas == secuenciales
  con `set_seed(seed)`).

## Cita

Si se usa este código o sus resultados, citar:

> Velandia, S., & Díaz Luna, J. D. (2026). *Taller de Aprendizaje por Refuerzo:
> diagnóstico y mejora del Q-Learning tabular sobre triqui*. Universidad
> Nacional de Colombia, Facultad de Ingeniería Mecatrónica, curso *Técnicas de
> Inteligencia Artificial*, abril 2026.

## Referencias bibliográficas

- Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction*
  (2nd ed.). MIT Press.
- Ng, A. Y., Harada, D., & Russell, S. (1999). Policy invariance under reward
  transformations: Theory and application to reward shaping. *ICML*.
- Mnih, V. *et al.* (2015). Human-level control through deep reinforcement
  learning. *Nature*, 518(7540), 529–533.
