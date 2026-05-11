# Prompt inicial para Claude Code

Copia y pega esto en la primera interacción con Claude Code, después de haber colocado `CLAUDE.md` y `PLAN_DE_TRABAJO.md` en la raíz del proyecto.

---

```
Antes de escribir cualquier código, lee con atención:
1. CLAUDE.md (contexto del proyecto y reglas)
2. PLAN_DE_TRABAJO.md (especificación técnica detallada)
3. Los archivos heredados de mi compañero en src/: triqui.py, triqui_algorithm.py, master_RL.py, triqui_train.py, triqui_championship.py. Esos NO se tocan, se importan.
4. El notebook original del profesor que está en el directorio raíz (TIA_20260423__Aprendizaje_Automatico_3_PorRefuerzo.ipynb) y el enunciado del taller (Taller_AprendizajePorRefuerzo-RL.pdf).

Después de leerlos, NO empieces a implementar todavía. Hazme tres cosas:

(a) Un resumen de tu entendimiento del proyecto en 5-7 bullets, destacando especialmente: qué se reutiliza del compañero, qué se construye nuevo, y la pregunta central del taller.

(b) Un plan detallado de implementación: lista ordenada de archivos a crear con su contenido conceptual, en el orden que recomiendes ejecutar. Indica dependencias entre módulos.

(c) Preguntas concretas sobre cualquier ambigüedad antes de empezar. No asumas; pregunta.

Una vez yo confirme tu plan, procederemos por fases. En cada fase: implementas, corres los tests asociados, y me muestras los resultados antes de pasar a la siguiente. No avances de fase sin mi visto bueno.

Empezamos.
```

---

## Cómo armar el proyecto en tu máquina

```bash
# 1. Crear el directorio del proyecto
mkdir -p ~/taller-rl-triqui
cd ~/taller-rl-triqui

# 2. Copiar los archivos del compañero (asumiendo que ya clonaste el repo)
mkdir -p src tests
cp ~/ruta/al/repo/Programacion/Python/TIA/L5v2/triqui.py src/
cp ~/ruta/al/repo/Programacion/Python/TIA/L5v2/triqui_algorithm.py src/
cp ~/ruta/al/repo/Programacion/Python/TIA/L5v2/master_RL.py src/
cp ~/ruta/al/repo/Programacion/Python/TIA/L5v2/triqui_train.py src/
cp ~/ruta/al/repo/Programacion/Python/TIA/L5v2/triqui_championship.py src/
cp ~/ruta/al/repo/Programacion/Python/TIA/L5v2/test_triqui.py tests/
cp ~/ruta/al/repo/Programacion/Python/TIA/L5v2/requirements.txt .

# 3. Copiar los archivos de planificación que generamos en este chat
cp ~/ruta/de/descarga/CLAUDE.md .
cp ~/ruta/de/descarga/PLAN_DE_TRABAJO.md .

# 4. Copiar también el material del profesor (para referencia de Claude Code)
cp ~/ruta/al/PDF/Taller_AprendizajePorRefuerzo-RL.pdf .
cp ~/ruta/al/notebook/TIA_20260423__Aprendizaje_Automatico_3_PorRefuerzo.ipynb .

# 5. Inicializar git (opcional pero recomendable)
git init
echo "results/models/*.pkl
results/models/*.pt
results/logs/*.csv
__pycache__/
*.pyc
.ipynb_checkpoints/" > .gitignore
git add . && git commit -m "Initial setup: heritage from group partner + planning docs"

# 6. Crear el entorno conda
conda create -n triqui-rl python=3.10 -y
conda activate triqui-rl
pip install -r requirements.txt
pip install pytest  # no está en requirements

# 7. Arrancar Claude Code
claude
```

Cuando Claude Code arranque, pégale el prompt de arriba.

---

## Flujo de trabajo recomendado durante la sesión con Claude Code

1. **Fase de planificación** (sin código aún): Claude Code lee todo, te entrega su entendimiento, su plan y sus preguntas. **Tú validas o corriges.**
2. **Fase 1** — utilidades sin dependencias: `seeds.py`, `minimax.py` + tests. Verificar que `pytest tests/test_minimax.py` pasa.
3. **Fase 2** — agente base: `base_q_agent.py` + adapter + tests. Verificar que `BaseQAgent` reproduce el comportamiento del notebook del profesor.
4. **Fase 3** — diagnóstico: `diagnostics.py`. Generar las tres figuras de patologías. **Punto crítico**: estas figuras justifican todo lo que sigue.
5. **Fase 4** — agente mejorado: `symmetry.py` + `improved_q_agent.py`. Entrenamiento de prueba con 1 semilla.
6. **Fase 5** — análisis estadístico: `multi_seed.py` + `bootstrap.py`.
7. **Fase 6** — ablations: `ablations.py`. Correr el estudio completo.
8. **Fase 7** — notebook final: integrar todo con narrativa.
9. **Fase 8** — pulir, verificar reproducibilidad, exportar figuras finales.

Entre fases, **commitea**. Si algo sale mal, retrocedes fácil.

---

## Trucos para Claude Code en este proyecto

- **Mantén `CLAUDE.md` actualizado** si cambias decisiones de diseño durante el camino.
- **Si Claude Code se pierde**: dile "consulta `CLAUDE.md` sección X" — relee y se reorienta.
- **Si Claude Code intenta modificar código del compañero**: detenlo. "Eso es del compañero, no se toca, usa importación."
- **Si propone añadir dependencias** (Ray, Wandb, PyTorch Lightning): rechaza salvo justificación fuerte.
- **Al final de cada fase**: pídele que corra `pytest -v` y reporte el resultado antes de avanzar.
- **Para el notebook**: pídele que genere primero un esqueleto con todas las secciones de markdown vacías, lo valides, y después rellene las celdas de código.

---

## Si algo se complica

- **Si los entrenamientos paralelos son lentos**: revisa si las semillas no están serializando correctamente algún objeto (los `nn.Module` no son `pickleable` directamente; usar `state_dict`).
- **Si los heatmaps del torneo del compañero rompen con `BaseQAgent`**: el wrapper `TabularToTorchAdapter` no está devolviendo el shape correcto. Es el punto más frágil de la integración.
- **Si el reward shaping cambia la política óptima**: estás violando el teorema de Ng et al. — debe ser potential-based puro (`Φ(s') γ - Φ(s)`), no shaping arbitrario.
