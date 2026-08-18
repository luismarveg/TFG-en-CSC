# Análisis de un modelo estocástico de células madre cancerígenas

Repositorio de códigos para mi Trabajo de Fin de Grado (TFG) en Física, *Análisis de un modelo estocástico de células madre cancerígenas*.

Este repositorio contiene los scripts en Python desarrollados para realizar las simulaciones y el análisis de la dinámica poblacional tumoral presentados en el trabajo. El modelado se fundamenta en el uso del algoritmo de Gillespie para el régimen microscópico y la aproximación continua mediante ecuaciones de Langevin.

## Setup

Para poder ejecutar los códigos, se recomienda crear un entorno virtual e instalar las dependencias necesarias.

```bash
# Crear entorno virtual
python -m venv .venv

# Activar entorno virtual (Windows)
.venv\Scripts\activate

# Instalar dependencias 
pip install -r requirements.txt
```

## Códigos y Experimentos

A continuación se detalla la función de cada uno de los programas incluidos en el repositorio:

| Archivo | Descripción |
|---|---|
| `Histograma.py` | Comparativa de la distribución de probabilidad estacionaria obtenida mediante el algoritmo de Gillespie frente a las ecuaciones de Langevin. |
| `Simulacion_CSC_Gillespie.py` | Simulación estocástica microscópica de las células madre cancerígenas utilizando el algoritmo de Gillespie. |
| `Simulacion_CSC_Langevin.py` | Aproximación continua mesoscópica del modelo mediante la integración numérica de ecuaciones de Langevin. |
| `Fases_interactivo.py` | Script para la visualización del diagrama de fases y las nulclinas del sistema determinista. |
| `Diagrama_parametros.py` | Representación del mapa de regímenes dinámicos y análisis del espacio de parámetros de las tasas de mortalidad ($d_S$ vs $d_D$). |
| `Extincion_estocastica_continuo.py` | Simulación y evaluación de los tiempos de extinción asumiendo un tratamiento continuo convencional. |
| `Extincion_estocastica_ciclo.py` | Simulación y evaluación de los tiempos de extinción bajo distintos regímenes de tratamiento cíclico (pulsos). |
| `Comparacion_tiempos.py` | Comparación de los tiempos medios de extinción tumoral entre tratamientos continuos y tratamientos cíclicos equivalentes. |
| `trayectorias_estocasticas.py` | Generación de series temporales de la dinámica poblacional (células madre y diferenciadas) convergiendo al equilibrio. |
