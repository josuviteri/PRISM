# Bitácora de uso de IA

Obligatoria según la guía docente. Vale **4 puntos de E2**.

**No se puntúa usar poco.** Se puntúa el criterio: una entrada que dice
«se lo pedimos, nos dio esto, lo rechazamos porque…» vale más nota que no
haber usado ninguna herramienta.

**La línea roja:** la IA se puede usar para comprender y generar código,
citándola. **No** para el análisis automático de los resultados ni para la
discusión reflexiva sobre el trabajo. Eso es lo que evalúa la asignatura, y se
comprueba en la defensa.

---

| Tarea | Herramienta | Qué devolvió | Qué hicimos y por qué |
|---|---|---|---|
| Papers and Research | AlphaXiv |  |  |
| Planning and Research | Claude web CLI (Sonnet 5.5) |  |  |
| Debugging | Gemini Flash 3.6 web CLI |  |   |
| Development | GitHub Copilot (GPT-5.6 Luna) | |   |
| Autocompletion | GitHub Copilot |  |  |
| Bug fixing and doc alignment (`guidance_scale: 0.0` sampling, `--solo-evaluar` with the MNIST evaluator checkpoint, README/AGENTS.md/W&B consistency) | Claude Code (Sonnet 5.5) | Patches in `diffusion_utils.py`, `diffusion_loop.py`, `ddpm_utils.py` and `run.py`, new regression tests and rewritten AGENTS.md/README sections. | Reviewed the diff and ran `pytest tests/ -q`; checked that the new `--solo-evaluar` test fails without the `run.py` fix. Kept the guidance convention (`1` = conditional only) and made unconditional models ignore the scale instead of changing the maths. |

---