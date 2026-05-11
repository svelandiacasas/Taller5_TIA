# Referencia: agente Q-Learning del notebook del profesor

> **Fuente de verdad para `BaseQAgent`.** Replicar bit a bit.
>
> Origen: notebook `TIA_20260423__Aprendizaje_Automatico_3_PorRefuerzo.ipynb`
> del Prof. Flavio Prieto (Ing. Mecatrónica — UNAL Bogotá, abril 2026), entregado
> por Santiago en el chat de planificación del Taller 5. El archivo `.ipynb`
> original no está versionado en este repo; este Markdown es la única copia
> autoritativa dentro del proyecto.

## Código literal del profesor

```python
import numpy as np
import random

EMPTY = 0
X = 1   # Agente
O = -1  # Humano

def initial_state():
    return np.zeros((3,3), dtype=int)

def available_actions(state):
    return [(i,j) for i in range(3) for j in range(3) if state[i,j] == EMPTY]

def check_winner(state):
    for i in range(3):
        if abs(sum(state[i,:])) == 3:
            return np.sign(sum(state[i,:]))
        if abs(sum(state[:,i])) == 3:
            return np.sign(sum(state[:,i]))
    diag1 = state[0,0] + state[1,1] + state[2,2]
    diag2 = state[0,2] + state[1,1] + state[2,0]
    if abs(diag1) == 3: return np.sign(diag1)
    if abs(diag2) == 3: return np.sign(diag2)
    if not available_actions(state): return 0
    return None

def state_to_tuple(state):
    return tuple(state.flatten())

# Q-Learning
Q = {}
alpha = 0.1
gamma = 0.9
epsilon = 0.2

def get_Q(state, action):
    return Q.get((state, action), 0.0)

def choose_action(state):
    actions = available_actions(np.array(state).reshape(3,3))
    if random.random() < epsilon:
        return random.choice(actions)
    qs = [get_Q(state, a) for a in actions]
    max_q = max(qs)
    return actions[qs.index(max_q)]

def update_Q(state, action, reward, next_state):
    actions = available_actions(np.array(next_state).reshape(3,3))
    max_q_next = 0 if not actions else max([get_Q(next_state, a) for a in actions])
    old_q = get_Q(state, action)
    Q[(state, action)] = old_q + alpha * (reward + gamma * max_q_next - old_q)

def train(episodes=10000):
    for _ in range(episodes):
        state = initial_state()
        current_player = X
        while True:
            s = state_to_tuple(state)
            action = choose_action(s)
            state[action] = current_player
            result = check_winner(state)
            s_next = state_to_tuple(state)
            if result is not None:
                reward = 1 if result == X else -1 if result == O else 0
                update_Q(s, action, reward, s_next)
                break
            else:
                update_Q(s, action, 0, s_next)
            current_player *= -1
```

## Convenciones (no se desvían en `BaseQAgent`)

- Tablero `3×3` de `int`. `X = 1`, `O = -1`, `EMPTY = 0`.
- Acción: tupla `(i, j)`.
- Estado para clave de `Q`: `tuple(state.flatten())`.
- `current_player *= -1` para alternar entre `+1` y `-1`.
- Hiperparámetros: `alpha = 0.1`, `gamma = 0.9`, `epsilon = 0.2` (constante).
- Recompensa: `+1` si gana X, `-1` si gana O, `0` si empata. Solo terminales.
- 10 000 episodios por defecto.

## Patologías intencionales que `BaseQAgent` debe replicar

1. **Q única compartida** entre X y O. La misma `Q(s, a)` se usa para decidir la
   jugada de ambos jugadores y para actualizarla. Inconsistente porque el valor
   de un estado-acción depende de quién mueva.
2. **Asignación de crédito errónea**. En el bloque terminal, `update_Q(s, action, reward, s_next)`
   aplica la recompensa al ÚLTIMO movimiento, que pudo ser de X o de O. Si O ganó
   (recompensa `-1` desde la perspectiva de X), esa recompensa se asigna al
   estado-acción de O — para O fue una jugada **ganadora** y debería tener valor
   positivo desde su perspectiva.
3. **Exploración mal calibrada**: `ε = 0.2` constante, sin decay. Nunca explota
   suficientemente y mantiene ruido permanente.
4. **Self-play degenerado**: ambos jugadores comparten política, así que el
   "rival" durante el entrenamiento es el propio agente. No hay adversario real.

## Sesgo del `argmax` con Q vacía

`actions[qs.index(max_q)]` con todos los `qs == 0.0` (estado nunca visto)
devuelve siempre la primera acción de la lista — i.e., `(0, 0)`. Este sesgo
"primer índice válido" es parte del comportamiento del notebook y se preserva.
El `TabularToTorchAdapter` lo refleja al devolver `0.0` en todas las posiciones
para estados no vistos, dejando que el `argmax` del torneo del compañero
seleccione la primera posición legal.

## Lo que NO está en este snippet (y por tanto NO va en `BaseQAgent`)

- `select_action_eval`, traducción de convenciones (1/2 vs ±1), wrapper torch.
  Esas piezas viven en la capa de boundary (`BaseQAgent.select_action_eval` y
  `TabularToTorchAdapter`), nunca dentro del núcleo Q-Learning.
- ε decay, dual perspective, reward shaping, simetrías. Eso es `ImprovedQAgent`.

## Criterios del enunciado del PDF (resumen operativo)

- 50 000 episodios de entrenamiento (`ImprovedQAgent`); 10 000 para `BaseQAgent`
  (lo que el profesor usa).
- ≥ 10 000 partidas de evaluación, sin aprendizaje (`epsilon = 0`).
- Semillas controladas (≥ 10 semillas para análisis multi-semilla).
- Métricas: Win Rate vs random y vs base; distribución W/D/L; curva vs episodios
  + episodios a estabilización; calidad de política; estabilidad entre semillas;
  eficiencia (tiempo + memoria).
- Variaciones obligatorias: agente inicia / oponente inicia / tableros parcialmente llenos.
