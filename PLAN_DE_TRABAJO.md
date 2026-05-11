# Plan de trabajo — Taller RL Triqui

> Especificación técnica para los componentes nuevos. Lo que reutilizamos del compañero está documentado en `CLAUDE.md`.

---

## 1. `seeds.py` — control de semillas

```python
def set_seed(seed: int) -> None:
    """Fija seeds de random, numpy y torch (incluida CUDA si está disponible)."""
```

Util mínima. Sin dependencias circulares.

---

## 2. `minimax.py` — oráculo perfecto

Triqui tiene ~5.478 estados legales. Resoluble exhaustivamente.

```python
class MinimaxAgent:
    def __init__(self, player_id: int, cache: bool = True): ...
    def select_action(self, game: Game) -> tuple[int, int]:
        """Devuelve la jugada óptima para el jugador actual."""
    def optimal_actions(self, game: Game) -> list[tuple[int, int]]:
        """Devuelve TODAS las jugadas óptimas empatadas en valor (para tiebreak random)."""
```

Implementación: minimax con alpha-beta + memoización por estado canónico (clave en `state.tobytes()`).

Test obligatorio (`test_minimax.py`):
- Minimax contra Minimax → siempre empate.
- Minimax (X) contra Random → nunca pierde (≥ 1.000 partidas).
- Minimax (O) contra Random → nunca pierde (≥ 1.000 partidas).

---

## 3. `base_q_agent.py` — agente base + adapter

### 3.1 Fiel al notebook del profesor

```python
class BaseQAgent:
    """Réplica del Q-Learning del notebook de clase. Defectos intencionales."""
    def __init__(self, alpha=0.1, gamma=0.9, epsilon=0.2):
        self.Q = {}  # diccionario único compartido (defecto intencional)
        ...
    def get_Q(self, state_tuple, action): ...
    def choose_action(self, state, available_actions, training=True): ...
    def update_Q(self, state, action, reward, next_state): ...
    def train(self, episodes=10000, seed=None):
        """Self-play degenerado del notebook. Una sola Q para ambos jugadores."""
    def select_action_eval(self, game): ...   # para usarlo como oponente
```

El método `train` replica la lógica exacta del notebook (`current_player *= -1`, misma `Q` para ambos, ε constante, no decay).

### 3.2 Wrapper torch-compatible

```python
class TabularToTorchAdapter(nn.Module):
    """Permite que un agente tabular se pase al Championship del compañero."""
    def __init__(self, tabular_agent, default_value=0.0):
        super().__init__()
        self.agent = tabular_agent
        self.default = default_value
        # parámetro dummy para que .to(device), .eval(), etc. no fallen
        self.dummy = nn.Parameter(torch.zeros(1), requires_grad=False)
    def forward(self, state_tensor):
        """state_tensor: [B, 9] → return [B, 9] con Q-values."""
        # Convertir cada fila a tupla, mirar self.agent.Q, rellenar con default
        ...
```

Test: encadenar `TabularToTorchAdapter(BaseQAgent())` con el `Championship` del compañero y verificar que no rompe.

---

## 4. `symmetry.py` — canonicalización D₄

Grupo D₄ tiene 8 elementos (4 rotaciones × 2 por reflexión).

```python
def canonical_state(state: np.ndarray) -> tuple:
    """Devuelve el representante canónico del estado bajo D4."""
def canonical_action(state: np.ndarray, action: int) -> int:
    """Acción rotada/reflejada al mismo marco canónico que canonical_state."""
def restore_action(state: np.ndarray, canonical_action: int) -> int:
    """Inversa: dada la acción en el marco canónico, devolverla en el marco original."""
```

Test (`test_symmetry.py`):
- 8 transformaciones de un estado dado producen el mismo `canonical_state`.
- `restore_action(state, canonical_action(state, a)) == a` para todo `a` legal.

---

## 5. `improved_q_agent.py` — Q-Learning mejorado

```python
class ImprovedQAgent:
    def __init__(
        self,
        alpha=0.1,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay=0.9995,
        use_symmetries=True,
        reward_shaping=True,
        optimistic_init=0.5,
        dual_perspective=True,   # Q desde el punto de vista del jugador-a-mover
    ): ...
    def train(
        self,
        episodes=50000,
        opponents="self_play",   # "self_play" | "random" | "mixed" | Algorithm
        seed=None,
        eval_every=500,
        eval_episodes=200,
    ) -> dict:
        """Devuelve historial: rewards, win_rates, epsilon, length."""
```

### Detalles importantes

**Dual perspective**: `Q[state, action]` se interpreta siempre desde el punto de vista del jugador que está a punto de mover. Antes de consultar/actualizar, normalizar el tablero para que ese jugador sea `+1`. Eso significa que cuando le toca a O, multiplicamos el tablero por `-1` antes de indexar Q.

**Reward shaping potential-based**: `Φ(s) = w_lines_two_in_a_row(s) - w_lines_opponent_two_in_a_row(s)`. Añadir `γΦ(s') - Φ(s)` a cada paso. Justificar con teorema de Ng et al. (1999): no altera la política óptima.

**Self-play correcto**: las jugadas del oponente se actualizan en `Q` con perspectiva del oponente. Equivale a "ambos jugadores aprenden simultáneamente con una Q compartida pero correctamente interpretada".

**Mixed opponents**: por episodio, muestrear de un pool: `[random, self, algorithm]` con probabilidades configurables. Por defecto: `[0.4, 0.5, 0.1]`.

### Wrapper torch-compatible

Mismo `TabularToTorchAdapter` que `BaseQAgent`, pero con `default_value = optimistic_init`.

---

## 6. `diagnostics.py` — análisis del agente base

Genera evidencia empírica de las tres patologías. Devuelve gráficas y tablas listas para el notebook.

```python
def show_self_play_degeneracy(agent: BaseQAgent, n_episodes=1000) -> Figure:
    """Mostrar que las jugadas de O durante el self-play del agente base
    son sistemáticamente peores que random."""

def show_credit_assignment_bug(agent: BaseQAgent) -> Figure:
    """Mostrar estados donde Q(s, a) tiene signo erróneo: jugadas buenas
    para O quedaron con valor negativo porque la actualización terminal
    se hizo con la recompensa de X."""

def show_no_convergence(seeds: list[int]) -> Figure:
    """Entrenar el agente base con 10+ semillas, graficar win rate vs
    Algorithm a lo largo de los episodios. Mostrar que NO converge:
    alta varianza entre semillas + estancamiento."""

def show_slow_learning(seeds: list[int]) -> Figure:
    """Comparar curva de aprendizaje base vs improved en la misma escala."""
```

---

## 7. `multi_seed.py` — paralelización de semillas

```python
def run_parallel_seeds(
    agent_class,
    agent_kwargs: dict,
    train_kwargs: dict,
    seeds: list[int],
    n_workers: int = None,   # default: min(len(seeds), cpu_count() - 2)
) -> list[dict]:
    """Lanza un proceso por semilla con multiprocessing.Pool.
    Devuelve lista de historiales (uno por semilla)."""
```

Importante: cada worker debe llamar `set_seed(seed)` al inicio, no se hereda.

---

## 8. `bootstrap.py` — intervalos de confianza y tests

```python
def bootstrap_ci(
    values: np.ndarray,
    statistic: callable = np.mean,
    n_resamples: int = 10000,
    confidence: float = 0.95,
) -> tuple[float, float, float]:
    """Devuelve (estadístico, low, high)."""

def paired_bootstrap_test(
    a: np.ndarray,
    b: np.ndarray,
    n_resamples: int = 10000,
) -> float:
    """p-value de H0: media(a) == media(b)."""
```

Uso en el notebook: para cada par (agente A, agente B) reportar **media de win rate ± IC95 bootstrap** y **p-value** de la comparación.

---

## 9. `ablations.py` — ablation study del ImprovedQAgent

Tabla obligatoria en el notebook:

| Variante | ε decay | Reward shaping | Symmetries | Dual persp. | Win vs Random | Win vs Algorithm | Loss rate vs Minimax |
|---|---|---|---|---|---|---|---|
| Full | ✓ | ✓ | ✓ | ✓ | ? | ? | ? |
| − decay | ✗ | ✓ | ✓ | ✓ | ? | ? | ? |
| − shaping | ✓ | ✗ | ✓ | ✓ | ? | ? | ? |
| − symmetries | ✓ | ✓ | ✗ | ✓ | ? | ? | ? |
| − dual persp. | ✓ | ✓ | ✓ | ✗ | ? | ? | ? |

Cada celda es **media ± IC95 sobre 10 semillas**, con `paired_bootstrap_test` contra la variante Full para detectar mejoras significativas.

```python
def run_ablation_study(seeds: list[int]) -> pd.DataFrame: ...
```

---

## 10. Notebook final `taller_rl_triqui.ipynb`

### Estructura sugerida (secciones de markdown + celdas)

1. **Portada y resumen ejecutivo** — 1 párrafo con la respuesta a la pregunta del taller.
2. **Marco teórico** — MDP, valor de estado/acción, Bellman, Q-Learning vs SARSA, reward shaping potential-based, minimax para juegos resueltos.
3. **Entorno y oponentes** — descripción del `Game`, del `Algorithm` heurístico, de `MinimaxAgent`, de `RandomAgent`.
4. **Agente base** — código, justificación de los hiperparámetros del notebook del profesor, entrenamiento, primera evaluación.
5. **Diagnóstico** — las 3 patologías con gráficas de `diagnostics.py`. Esta sección es **central**.
6. **Agente mejorado (Q-Learning)** — diseño, ecuaciones modificadas, código, entrenamiento multi-semilla.
7. **SARSA** — usando `MasterSARSA` del compañero. Mismo protocolo.
8. **DQN** — usando `MasterDQN` del compañero. Mismo protocolo.
9. **Comparación final** — tablas con IC bootstrap, gráficas de barras, heatmaps del torneo, distancia a Minimax.
10. **Análisis de robustez** — agente inicia / oponente inicia / tableros parcialmente llenos.
11. **Ablation study** — tabla del módulo 9 + discusión.
12. **Discusión y conclusiones** — respuesta a la pregunta del taller con evidencia.
13. **Limitaciones y trabajo futuro**.
14. **Referencias**.

### Reglas para el notebook

- Las celdas pesadas (entrenamientos) deben **cargar resultados desde disco si existen** (`results/logs/*.csv`, `results/models/*.pkl`). Si no existen, entrenar. Así el notebook se puede re-ejecutar para el video sin re-entrenar todo.
- Cada figura se guarda en `results/figures/<seccion>_<nombre>.png` con `dpi=150`.
- Las tablas relevantes se exportan también a `.csv` en `results/`.

---

## 11. Scripts CLI

```bash
python scripts/train_base.py --seeds 0,1,2,...,9 --episodes 10000
python scripts/train_improved.py --seeds 0,1,2,...,9 --episodes 50000
python scripts/run_diagnostics.py
python scripts/run_ablations.py --seeds 0,1,2,...,9
python scripts/run_tournament.py
```

Cada script guarda en `results/` y es idempotente (skip si los archivos ya existen, salvo `--force`).

---

## 12. Tests mínimos

- `test_base_q_agent.py`: la `Q` después de N episodios con seed fijo es determinista; el wrapper `TabularToTorchAdapter` devuelve tensor de la forma correcta.
- `test_minimax.py`: lo descrito en sección 2.
- `test_symmetry.py`: lo descrito en sección 4.

Comando: `pytest tests/ -v`. Debe pasar antes de generar resultados para el notebook.

---

## 13. Estimación de tiempo

| Tarea | Tiempo |
|---|---|
| Setup repo + integración del código del compañero | 30 min |
| `seeds.py`, `minimax.py` + tests | 1 h |
| `base_q_agent.py` + adapter + tests | 1 h |
| `diagnostics.py` (las 3 patologías) | 1.5 h |
| `symmetry.py` + tests | 45 min |
| `improved_q_agent.py` | 2 h |
| `multi_seed.py` + `bootstrap.py` | 1 h |
| `ablations.py` | 1 h |
| Entrenamientos reales (en paralelo, 10 semillas) | 30 min – 1 h de máquina |
| Notebook final | 3 h |
| Total estimado | **~12 h de trabajo + 1 h de máquina** |

Con el hardware disponible (20 cores), los entrenamientos paralelos no son cuello de botella.
