# Contexto del proyecto — Taller RL Triqui

## Qué es esto

Entrega grupal del **Taller de Aprendizaje por Refuerzo** del curso *Técnicas de Inteligencia Artificial* (Ing. Mecatrónica — UNAL Bogotá, Prof. Flavio Prieto, abril 2026).

El taller pide partir de un agente Q-Learning tabular básico (provisto por el profesor en un notebook de clase) y **superarlo** mediante una técnica alternativa o una mejora significativa, respondiendo con análisis teórico y evidencia experimental a la pregunta: *¿qué enfoque de aprendizaje permite aprender mejor a jugar triqui y por qué?*

**Entregable**: notebook `.ipynb` + video ≤ 15 min.

## División del trabajo dentro del grupo

Un miembro del grupo (**el compañero**, repo `AxumII/Programacion/Python/TIA/L5v2`) ya construyó:
- El motor del juego (`triqui.py`).
- Un agente heurístico fuerte basado en reglas (`triqui_algorithm.py`).
- Abstracciones de DQN y SARSA con replay buffer y target network (`master_RL.py`).
- Entorno gym-like con evaluación detallada por iniciativa/turno/posición (`triqui_train.py`).
- Sistema de torneo con heatmaps (`triqui_championship.py`).
- Orquestador con curriculum learning (`triqui_general.py`).

**Esta parte se reutiliza tal cual**. No la reescribas; impórtala. Es código del grupo.

**Lo que falta y debes implementar**:
1. `BaseQAgent` — Q-Learning tabular fiel al notebook del profesor (línea base obligatoria del taller).
2. `ImprovedQAgent` — Q-Learning tabular bien hecho (self-play correcto + ε decay + reward shaping + canonicalización por simetrías opcional).
3. `MinimaxAgent` — oráculo perfecto (triqui es un juego resuelto, con alpha-beta basta).
4. **Diagnóstico empírico del agente base**: demostrar las tres patologías (no converge, asignación de crédito errónea por self-play degenerado, exploración mal calibrada).
5. **Análisis estadístico riguroso**: multi-semilla en paralelo (≥ 10 semillas), intervalos de confianza bootstrap, ablation study, opcional grid search de hiperparámetros.
6. **Notebook final** `taller_rl_triqui.ipynb` que integre todo con narrativa, justificación teórica, gráficas y discusión.

## Hardware disponible

- CPU: Intel i7-14700K (20 cores / 28 threads).
- GPU: NVIDIA RTX 5080.
- Triqui es trivial computacionalmente: no usar GPU salvo para DQN (que ya está implementado por el compañero). El uso real del hardware es **paralelizar semillas con `multiprocessing`**.

## Convenciones de código

- Python 3.10+.
- Type hints en funciones públicas.
- Docstrings estilo NumPy/Google (cortos pero claros).
- `numpy`, `pandas`, `matplotlib`, `torch` (heredado del compañero), `seaborn` (heredado), `pytest`.
- No introducir nuevas dependencias pesadas.
- Mantener compatibilidad con el código existente del compañero — usar los mismos nombres de clases/parámetros cuando se interactúe con su API (`Game`, `Train`, `Championship`, etc.).

## Reproducibilidad

- Toda corrida debe aceptar `seed` como parámetro.
- Función `set_seed(seed)` que fija `random`, `numpy.random`, `torch.manual_seed`, `torch.cuda.manual_seed_all`.
- Semillas reportadas: `[0, 1, ..., 9]` mínimo (10 semillas paralelas).
- Logs en CSV con una fila por episodio: `(seed, episode, reward, win_rate_ma, length, epsilon)`.

## Política sobre el agente base (importante)

`BaseQAgent` debe replicar **fielmente** el código del notebook de clase, incluyendo sus defectos. No es para "ganar", es el strawman a diagnosticar. Concretamente:
- `Q` como diccionario único compartido entre X y O (self-play degenerado).
- `epsilon = 0.2` constante, sin decaimiento.
- `alpha = 0.1`, `gamma = 0.9`.
- 10.000 episodios por defecto.
- Recompensas solo terminales: `+1 / -1 / 0`.

## Política sobre el agente mejorado

`ImprovedQAgent` debe incluir:
- **Self-play correcto** con perspectiva del jugador-a-mover (negar reward para el oponente o usar dos Q-tables separadas).
- **ε decay**: 1.0 → 0.05 exponencial.
- **Reward shaping potential-based** (justificar con teorema de Ng, Harada, Russell 1999).
- **Más episodios**: 50.000.
- **Canonicalización D₄** como flag (`use_symmetries=True/False`) — para ablation study.
- Recompensas terminales más informativas si conviene (justificar).

## Sobre el reward de movimiento ilegal en el código del compañero

El entorno `Train` del compañero usa `reward = -10` para acciones ilegales en el agente DQN/SARSA, sin enmascarar acciones durante entrenamiento. Es una decisión de diseño deliberada (que el agente aprenda legalidad internamente). **No la cambies.** En el notebook se justifica teóricamente como reward shaping para incentivar exploración acotada al espacio legal.

## Estructura esperada

```
taller-rl-triqui/
├── README.md
├── CLAUDE.md                     ← este archivo
├── requirements.txt
├── src/
│   ├── triqui.py                 ← del compañero, intacto
│   ├── triqui_algorithm.py       ← del compañero, intacto
│   ├── master_RL.py              ← del compañero, intacto
│   ├── triqui_train.py           ← del compañero, intacto
│   ├── triqui_championship.py    ← del compañero, intacto
│   └── new/                      ← código nuevo de este aporte
│       ├── __init__.py
│       ├── base_q_agent.py       ← BaseQAgent + wrapper torch-compatible
│       ├── improved_q_agent.py   ← ImprovedQAgent
│       ├── minimax.py            ← MinimaxAgent con alpha-beta
│       ├── symmetry.py           ← canonicalización D4
│       ├── diagnostics.py        ← análisis del agente base
│       ├── multi_seed.py         ← runner paralelo de semillas
│       ├── bootstrap.py          ← IC bootstrap + tests estadísticos
│       ├── ablations.py          ← ablation study
│       └── seeds.py              ← set_seed()
├── notebooks/
│   └── taller_rl_triqui.ipynb    ← entregable final
├── scripts/
│   ├── train_base.py
│   ├── train_improved.py
│   ├── run_diagnostics.py
│   ├── run_ablations.py
│   └── run_tournament.py
├── results/
│   ├── models/                   ← .pkl (tabulares) + .pt (heredados de DQN/SARSA)
│   ├── figures/                  ← .png exportadas del notebook
│   └── logs/                     ← CSV por episodio
└── tests/
    ├── test_triqui.py            ← del compañero, intacto
    ├── test_base_q_agent.py      ← nuevo
    ├── test_minimax.py           ← nuevo
    └── test_symmetry.py          ← nuevo
```

## Wrapper crucial: integración tabular ↔ torneo del compañero

El sistema del compañero (`Championship`, `evaluate_agent`) asume agentes que son `nn.Module` y se llaman como `q_vals = model(state_tensor)`.

`BaseQAgent` y `ImprovedQAgent` son tabulares (diccionarios). Para que entren al torneo del compañero **sin tocar su código**, crea un wrapper `TabularToTorchAdapter(nn.Module)`:
- Hereda de `nn.Module`.
- Internamente guarda la Q-table.
- En `forward(state_tensor)` devuelve un tensor `[1, 9]` con los Q-values del estado correspondiente.
- Estados no vistos → devuelve tensor de ceros (o valores optimistas si es `ImprovedQAgent`).

Eso permite enchufar agentes tabulares al `Championship` sin modificar `master_RL.py`.

## Métricas obligatorias (del enunciado del taller)

1. **Win Rate** — vs oponente aleatorio y vs agente base.
2. **Distribución W/D/L**.
3. **Velocidad de aprendizaje** — curva vs episodios + episodios hasta estabilización.
4. **Calidad de la política** — capacidad de evitar derrotas y aprovechar errores. Métrica adicional propuesta: **distancia a Minimax** (porcentaje de jugadas que coinciden con la política óptima del oráculo).
5. **Estabilidad** — varianza entre semillas (boxplot del win rate final).
6. **Eficiencia** — tiempo de entrenamiento + memoria.

## Protocolo experimental (obligatorio)

- 50.000 episodios de entrenamiento.
- ≥ 10.000 partidas de evaluación contra cada oponente (sin aprendizaje, ε=0 o equivalente).
- ≥ 10 semillas para todos los reportes principales.
- Variación de condiciones iniciales: agente inicia / oponente inicia / tablero parcialmente lleno.

## Orden de implementación

1. `seeds.py` y `minimax.py` primero (utilidades sin dependencias).
2. `base_q_agent.py` con wrapper torch-compatible.
3. `diagnostics.py`: correr `BaseQAgent` y graficar las patologías.
4. `symmetry.py` y `improved_q_agent.py`.
5. `multi_seed.py` y `bootstrap.py`.
6. `ablations.py`.
7. Notebook final integrando todo.

## Cosas importantes que NO hacer

- No reescribir el código del compañero.
- No introducir dependencias nuevas pesadas (PyTorch Lightning, Ray, Wandb, etc.).
- No "mejorar" el agente base para que gane: el agente base existe para mostrar las patologías. Mejorarlo derrota el propósito del taller.
- No omitir el análisis multi-semilla aunque "se vea bien" con una sola corrida.
- No usar la GPU para los agentes tabulares (es contraproducente).

## Cuando estés listo para empezar

Lee este archivo + `PLAN_DE_TRABAJO.md` y antes de escribir código, propone un plan más fino paso a paso para validación.
