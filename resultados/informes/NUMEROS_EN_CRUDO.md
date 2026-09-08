# Todos los números en crudo

> **Huecos conocidos, para que no se lean como ceros.** `nan` en el ajuste de parámetros
> significa que el ajuste no convergió en la mediana de ese escenario (el trazador pierde la
> pelota en demasiados cuadros; la columna `det` da la tasa de detección). Los checkpoints 250 y
> 875 de la corrida en velocidad no tienen ajuste de parámetros porque el análisis falló sobre
> ese directorio. La corrida con aceleración se cortó en el paso 807, así que no tiene 875 ni
> 1000 ni cadena final. `ood_t2v` aparece sólo en algunos brazos: son los prompts fuera de
> dominio sin ground truth, y sus valores no son comparables contra el simulador.

Generado desde los JSON de evaluación y los logs de entrenamiento por `scripts/gen_reporte.py`. Sin filtrar: toda métrica, todo checkpoint, todo escenario, con su n y su p. Las métricas `mae_ay`, `jerk_rms`, `l_rot` y `l_rot_norm` **las gana un video estático**: no son interpretables sin comparar contra la cota `quieto`.

## 1. Corrida con la pérdida sobre VELOCIDAD (λ = 4,7e-03, 1000 pasos)

### Checkpoint 125, en distribución (10 clips/escenario)
`e4vel/results/e4vel/paso_125/eval_results.json` · brazos: base, equiv_vel, ctrl_l40s · frames generados: 33

**vel_mae**
| brazo | free_fall | pendulum | bouncing |
|---|---|---|---|
| base | 12.212 | 6.141 | 9.956 |
| equiv_vel | 5.314 | 3.894 | 6.290 |
| ctrl_l40s | 4.535 | 3.959 | 6.251 |

**mov_ratio**
| brazo | free_fall | pendulum | bouncing |
|---|---|---|---|
| base | 0.524 | 0.479 | 0.093 |
| equiv_vel | 0.597 | 0.697 | 0.501 |
| ctrl_l40s | 0.679 | 0.629 | 0.504 |

**mae_ay**
| brazo | free_fall | pendulum | bouncing |
|---|---|---|---|
| base | 1.093 | 1.589 | 3.228 |
| equiv_vel | 0.754 | 0.810 | 3.748 |
| ctrl_l40s | 0.798 | 0.674 | 3.889 |

**jerk_rms**
| brazo | free_fall | pendulum | bouncing | ood_t2v |
|---|---|---|---|---|
| base | 4.401 | 3.180 | 0.604 | 1.017 |
| equiv_vel | 1.118 | 0.987 | 1.891 | 2.072 |
| ctrl_l40s | 1.358 | 0.786 | 2.620 | 3.251 |

**l_rot**
| brazo | free_fall | pendulum | bouncing |
|---|---|---|---|
| base | 15.794 | 11.088 | 0.208 |
| equiv_vel | 2.005 | 1.030 | 3.639 |
| ctrl_l40s | 2.543 | 2.067 | 5.731 |

**l_rot_norm**
| brazo | free_fall | pendulum | bouncing |
|---|---|---|---|
| base | 0.014 | 0.000 | 0.000 |
| equiv_vel | 0.169 | 0.095 | 0.351 |
| ctrl_l40s | 0.009 | 0.067 | 0.160 |

**Apareado por clip contra `ctrl_l40s`** (Wilcoxon, todos los escenarios juntos)
| brazo | métrica | n | ref | brazo | Δ | mejor | p |
|---|---|---|---|---|---|---|---|
| base | vel_mae | 30 | 4.915 | 9.436 | +4.521 | 2/30 | 0.000 |
| base | mov_ratio | 30 | 0.604 | 0.366 | -0.238 | 26/30 | 0.000 |
| base | mae_ay | 30 | 1.787 | 1.970 | +0.183 | 19/30 | 0.289 |
| base | jerk_rms | 30 | 1.588 | 2.729 | +1.141 | 20/30 | 0.440 |
| base | l_rot | 30 | 3.447 | 9.030 | +5.582 | 22/30 | 0.031 |
| base | l_rot_norm | 30 | 0.078 | 0.005 | -0.074 | 9/30 | 0.022 |
| equiv_vel | vel_mae | 30 | 4.915 | 5.166 | +0.251 | 13/30 | 0.328 |
| equiv_vel | mov_ratio | 30 | 0.604 | 0.598 | -0.006 | 14/30 | 1.000 |
| equiv_vel | mae_ay | 30 | 1.787 | 1.771 | -0.016 | 13/30 | 0.984 |
| equiv_vel | jerk_rms | 30 | 1.588 | 1.332 | -0.256 | 21/30 | 0.031 |
| equiv_vel | l_rot | 30 | 3.447 | 2.225 | -1.223 | 22/30 | 0.001 |
| equiv_vel | l_rot_norm | 30 | 0.078 | 0.205 | +0.126 | 4/30 | 0.035 |

### Checkpoint 250, en distribución (10 clips/escenario)
`e4vel/results/e4vel/paso_250/eval_results.json` · brazos: base, equiv_vel, ctrl_l40s · frames generados: 33

**vel_mae**
| brazo | free_fall | pendulum | bouncing |
|---|---|---|---|
| base | 11.737 | 5.675 | 9.960 |
| equiv_vel | 4.766 | 4.068 | 6.735 |
| ctrl_l40s | 4.613 | 3.736 | 6.163 |

**mov_ratio**
| brazo | free_fall | pendulum | bouncing |
|---|---|---|---|
| base | 0.446 | 0.457 | 0.093 |
| equiv_vel | 0.649 | 0.629 | 0.424 |
| ctrl_l40s | 0.664 | 0.649 | 0.511 |

**mae_ay**
| brazo | free_fall | pendulum | bouncing |
|---|---|---|---|
| base | 0.910 | 1.363 | 3.232 |
| equiv_vel | 0.756 | 0.755 | 3.669 |
| ctrl_l40s | 0.786 | 0.631 | 3.820 |

**jerk_rms**
| brazo | free_fall | pendulum | bouncing | ood_t2v |
|---|---|---|---|---|
| base | 3.841 | 3.366 | 0.616 | 0.934 |
| equiv_vel | 1.206 | 0.868 | 1.739 | 1.808 |
| ctrl_l40s | 1.423 | 0.769 | 2.584 | 2.982 |

**l_rot**
| brazo | free_fall | pendulum | bouncing |
|---|---|---|---|
| base | 15.191 | 7.484 | 0.212 |
| equiv_vel | 1.103 | 1.191 | 4.537 |
| ctrl_l40s | 2.170 | 3.128 | 5.648 |

**l_rot_norm**
| brazo | free_fall | pendulum | bouncing |
|---|---|---|---|
| base | 0.010 | 0.000 | 0.000 |
| equiv_vel | 0.000 | 0.173 | 0.486 |
| ctrl_l40s | 0.081 | 0.477 | 0.233 |

**Apareado por clip contra `ctrl_l40s`** (Wilcoxon, todos los escenarios juntos)
| brazo | métrica | n | ref | brazo | Δ | mejor | p |
|---|---|---|---|---|---|---|---|
| base | vel_mae | 30 | 4.838 | 9.124 | +4.287 | 1/30 | 0.000 |
| base | mov_ratio | 30 | 0.608 | 0.332 | -0.276 | 26/30 | 0.000 |
| base | mae_ay | 30 | 1.746 | 1.835 | +0.089 | 20/30 | 0.299 |
| base | jerk_rms | 30 | 1.592 | 2.608 | +1.015 | 18/30 | 0.570 |
| base | l_rot | 30 | 3.649 | 7.629 | +3.981 | 24/30 | 0.005 |
| base | l_rot_norm | 30 | 0.264 | 0.003 | -0.261 | 14/30 | 0.001 |
| equiv_vel | vel_mae | 30 | 4.838 | 5.190 | +0.352 | 13/30 | 0.096 |
| equiv_vel | mov_ratio | 30 | 0.608 | 0.568 | -0.040 | 16/30 | 0.213 |
| equiv_vel | mae_ay | 30 | 1.746 | 1.727 | -0.019 | 13/30 | 1.000 |
| equiv_vel | jerk_rms | 30 | 1.592 | 1.271 | -0.321 | 18/30 | 0.164 |
| equiv_vel | l_rot | 30 | 3.649 | 2.277 | -1.372 | 25/30 | 0.000 |
| equiv_vel | l_rot_norm | 30 | 0.264 | 0.220 | -0.044 | 12/30 | 0.523 |

### Checkpoint 375, en distribución (10 clips/escenario)
`e4vel/results/e4vel/paso_375/eval_results.json` · brazos: base, equiv_vel, ctrl_l40s · frames generados: 33

**vel_mae**
| brazo | free_fall | pendulum | bouncing |
|---|---|---|---|
| base | 12.057 | 6.036 | 9.954 |
| equiv_vel | 5.176 | 4.101 | 6.660 |
| ctrl_l40s | 4.805 | 3.815 | 6.119 |

**mov_ratio**
| brazo | free_fall | pendulum | bouncing |
|---|---|---|---|
| base | 0.493 | 0.472 | 0.094 |
| equiv_vel | 0.598 | 0.694 | 0.536 |
| ctrl_l40s | 0.627 | 0.627 | 0.545 |

**mae_ay**
| brazo | free_fall | pendulum | bouncing |
|---|---|---|---|
| base | 1.127 | 1.412 | 3.230 |
| equiv_vel | 0.475 | 0.763 | 3.557 |
| ctrl_l40s | 0.755 | 0.603 | 4.123 |

**jerk_rms**
| brazo | free_fall | pendulum | bouncing | ood_t2v |
|---|---|---|---|---|
| base | 4.824 | 3.315 | 0.612 | 1.995 |
| equiv_vel | 1.318 | 0.762 | 1.471 | 2.208 |
| ctrl_l40s | 1.352 | 0.736 | 3.096 | 2.300 |

**l_rot**
| brazo | free_fall | pendulum | bouncing |
|---|---|---|---|
| base | 19.324 | 11.983 | 0.300 |
| equiv_vel | 1.025 | 0.998 | 2.732 |
| ctrl_l40s | 1.616 | 1.844 | 7.986 |

**l_rot_norm**
| brazo | free_fall | pendulum | bouncing |
|---|---|---|---|
| base | 0.005 | 0.000 | 0.054 |
| equiv_vel | 0.000 | 0.148 | 0.190 |
| ctrl_l40s | 0.035 | 0.298 | 0.389 |

**Apareado por clip contra `ctrl_l40s`** (Wilcoxon, todos los escenarios juntos)
| brazo | métrica | n | ref | brazo | Δ | mejor | p |
|---|---|---|---|---|---|---|---|
| base | vel_mae | 30 | 4.913 | 9.349 | +4.436 | 0/30 | 0.000 |
| base | mov_ratio | 30 | 0.599 | 0.353 | -0.246 | 27/30 | 0.000 |
| base | mae_ay | 30 | 1.827 | 1.923 | +0.096 | 19/30 | 0.289 |
| base | jerk_rms | 30 | 1.728 | 2.917 | +1.189 | 19/30 | 0.516 |
| base | l_rot | 30 | 3.815 | 10.535 | +6.720 | 22/30 | 0.119 |
| base | l_rot_norm | 30 | 0.241 | 0.020 | -0.221 | 13/30 | 0.004 |
| equiv_vel | vel_mae | 30 | 4.913 | 5.312 | +0.399 | 11/30 | 0.050 |
| equiv_vel | mov_ratio | 30 | 0.599 | 0.609 | +0.010 | 17/30 | 0.598 |
| equiv_vel | mae_ay | 30 | 1.827 | 1.598 | -0.229 | 16/30 | 0.262 |
| equiv_vel | jerk_rms | 30 | 1.728 | 1.184 | -0.544 | 24/30 | 0.001 |
| equiv_vel | l_rot | 30 | 3.815 | 1.585 | -2.230 | 23/30 | 0.001 |
| equiv_vel | l_rot_norm | 30 | 0.241 | 0.113 | -0.128 | 11/30 | 0.088 |

### Checkpoint 500, en distribución (10 clips/escenario)
`e4vel/results/e4vel/paso_500/eval_results.json` · brazos: base, equiv_vel, ctrl_l40s · frames generados: 33

**vel_mae**
| brazo | free_fall | pendulum | bouncing |
|---|---|---|---|
| base | 12.538 | 5.758 | 9.959 |
| equiv_vel | 3.743 | 3.046 | 6.304 |
| ctrl_l40s | 4.414 | 3.752 | 5.454 |

**mov_ratio**
| brazo | free_fall | pendulum | bouncing |
|---|---|---|---|
| base | 0.518 | 0.475 | 0.093 |
| equiv_vel | 0.729 | 1.085 | 0.584 |
| ctrl_l40s | 0.681 | 0.790 | 0.575 |

**mae_ay**
| brazo | free_fall | pendulum | bouncing |
|---|---|---|---|
| base | 1.258 | 1.451 | 3.234 |
| equiv_vel | 0.580 | 0.683 | 3.773 |
| ctrl_l40s | 0.798 | 0.752 | 3.999 |

**jerk_rms**
| brazo | free_fall | pendulum | bouncing | ood_t2v |
|---|---|---|---|---|
| base | 4.695 | 3.606 | 0.601 | 0.602 |
| equiv_vel | 1.462 | 1.239 | 1.718 | 2.196 |
| ctrl_l40s | 1.626 | 1.075 | 2.869 | 2.553 |

**l_rot**
| brazo | free_fall | pendulum | bouncing |
|---|---|---|---|
| base | 20.229 | 8.308 | 0.249 |
| equiv_vel | 1.477 | 1.817 | 3.008 |
| ctrl_l40s | 1.874 | 3.587 | 8.260 |

**l_rot_norm**
| brazo | free_fall | pendulum | bouncing |
|---|---|---|---|
| base | 0.000 | 0.000 | 0.000 |
| equiv_vel | 0.000 | 0.000 | 0.042 |
| ctrl_l40s | 0.000 | 0.336 | 0.260 |

**Apareado por clip contra `ctrl_l40s`** (Wilcoxon, todos los escenarios juntos)
| brazo | métrica | n | ref | brazo | Δ | mejor | p |
|---|---|---|---|---|---|---|---|
| base | vel_mae | 30 | 4.540 | 9.418 | +4.878 | 1/30 | 0.000 |
| base | mov_ratio | 30 | 0.682 | 0.362 | -0.320 | 25/30 | 0.000 |
| base | mae_ay | 30 | 1.850 | 1.981 | +0.131 | 21/30 | 0.088 |
| base | jerk_rms | 30 | 1.857 | 2.967 | +1.110 | 19/30 | 0.440 |
| base | l_rot | 30 | 4.573 | 9.595 | +5.022 | 23/30 | 0.073 |
| base | l_rot_norm | 30 | 0.198 | 0.000 | -0.198 | 10/30 | 0.005 |
| equiv_vel | vel_mae | 30 | 4.540 | 4.365 | -0.176 | 15/30 | 0.792 |
| equiv_vel | mov_ratio | 30 | 0.682 | 0.799 | +0.117 | 10/30 | 0.017 |
| equiv_vel | mae_ay | 30 | 1.850 | 1.679 | -0.171 | 20/30 | 0.015 |
| equiv_vel | jerk_rms | 30 | 1.857 | 1.473 | -0.384 | 18/30 | 0.047 |
| equiv_vel | l_rot | 30 | 4.573 | 2.101 | -2.472 | 23/30 | 0.000 |
| equiv_vel | l_rot_norm | 30 | 0.198 | 0.014 | -0.184 | 10/30 | 0.010 |

### Checkpoint 625, en distribución (10 clips/escenario)
`e4vel/results/e4vel/paso_625/eval_results.json` · brazos: base, equiv_vel, ctrl_l40s · frames generados: 33

**vel_mae**
| brazo | free_fall | pendulum | bouncing |
|---|---|---|---|
| base | 11.580 | 5.954 | 9.961 |
| equiv_vel | 2.833 | 3.739 | 6.404 |
| ctrl_l40s | 5.767 | 3.895 | 6.236 |

**mov_ratio**
| brazo | free_fall | pendulum | bouncing |
|---|---|---|---|
| base | 0.476 | 0.488 | 0.093 |
| equiv_vel | 0.810 | 0.870 | 0.499 |
| ctrl_l40s | 0.565 | 0.652 | 0.492 |

**mae_ay**
| brazo | free_fall | pendulum | bouncing |
|---|---|---|---|
| base | 1.048 | 1.404 | 3.231 |
| equiv_vel | 0.728 | 0.714 | 3.918 |
| ctrl_l40s | 0.887 | 0.689 | 3.889 |

**jerk_rms**
| brazo | free_fall | pendulum | bouncing | ood_t2v |
|---|---|---|---|---|
| base | 4.299 | 3.578 | 0.615 | 1.176 |
| equiv_vel | 1.458 | 0.871 | 2.267 | 2.760 |
| ctrl_l40s | 1.317 | 1.147 | 2.232 | 2.597 |

**l_rot**
| brazo | free_fall | pendulum | bouncing |
|---|---|---|---|
| base | 14.637 | 8.921 | 0.235 |
| equiv_vel | 1.829 | 1.160 | 4.665 |
| ctrl_l40s | 3.192 | 5.268 | 6.302 |

**l_rot_norm**
| brazo | free_fall | pendulum | bouncing |
|---|---|---|---|
| base | 0.023 | 0.000 | 0.000 |
| equiv_vel | 0.000 | 0.000 | 0.065 |
| ctrl_l40s | 0.149 | 0.523 | 0.476 |

**Apareado por clip contra `ctrl_l40s`** (Wilcoxon, todos los escenarios juntos)
| brazo | métrica | n | ref | brazo | Δ | mejor | p |
|---|---|---|---|---|---|---|---|
| base | vel_mae | 30 | 5.299 | 9.165 | +3.866 | 3/30 | 0.000 |
| base | mov_ratio | 30 | 0.570 | 0.353 | -0.217 | 26/30 | 0.001 |
| base | mae_ay | 30 | 1.822 | 1.894 | +0.073 | 17/30 | 0.221 |
| base | jerk_rms | 30 | 1.565 | 2.830 | +1.265 | 19/30 | 0.700 |
| base | l_rot | 30 | 4.921 | 7.931 | +3.010 | 24/30 | 0.012 |
| base | l_rot_norm | 30 | 0.383 | 0.008 | -0.375 | 17/30 | 0.000 |
| equiv_vel | vel_mae | 30 | 5.299 | 4.326 | -0.974 | 17/30 | 0.080 |
| equiv_vel | mov_ratio | 30 | 0.570 | 0.726 | +0.157 | 9/30 | 0.000 |
| equiv_vel | mae_ay | 30 | 1.822 | 1.787 | -0.035 | 15/30 | 0.968 |
| equiv_vel | jerk_rms | 30 | 1.565 | 1.532 | -0.034 | 10/30 | 0.339 |
| equiv_vel | l_rot | 30 | 4.921 | 2.551 | -2.369 | 19/30 | 0.009 |
| equiv_vel | l_rot_norm | 30 | 0.383 | 0.022 | -0.361 | 17/30 | 0.000 |

### Checkpoint 750, en distribución (10 clips/escenario)
`e4vel/results/e4vel/paso_750/eval_results.json` · brazos: base, equiv_vel, ctrl_l40s · frames generados: 33

**vel_mae**
| brazo | free_fall | pendulum | bouncing |
|---|---|---|---|
| base | 11.909 | 5.915 | 9.950 |
| equiv_vel | 5.577 | 3.480 | 6.620 |
| ctrl_l40s | 4.157 | 3.395 | 5.084 |

**mov_ratio**
| brazo | free_fall | pendulum | bouncing |
|---|---|---|---|
| base | 0.471 | 0.504 | 0.094 |
| equiv_vel | 0.574 | 0.878 | 0.484 |
| ctrl_l40s | 0.705 | 0.783 | 0.657 |

**mae_ay**
| brazo | free_fall | pendulum | bouncing |
|---|---|---|---|
| base | 1.136 | 1.425 | 3.232 |
| equiv_vel | 0.833 | 0.661 | 3.771 |
| ctrl_l40s | 0.804 | 0.671 | 4.052 |

**jerk_rms**
| brazo | free_fall | pendulum | bouncing | ood_t2v |
|---|---|---|---|---|
| base | 4.010 | 3.512 | 0.607 | 0.891 |
| equiv_vel | 1.137 | 0.940 | 1.226 | 1.547 |
| ctrl_l40s | 1.489 | 1.025 | 2.945 | 2.632 |

**l_rot**
| brazo | free_fall | pendulum | bouncing |
|---|---|---|---|
| base | 13.094 | 10.399 | 0.214 |
| equiv_vel | 1.053 | 0.929 | 2.340 |
| ctrl_l40s | 2.306 | 4.605 | 7.872 |

**l_rot_norm**
| brazo | free_fall | pendulum | bouncing |
|---|---|---|---|
| base | 0.000 | 0.000 | 0.000 |
| equiv_vel | 0.000 | 0.000 | 0.431 |
| ctrl_l40s | 0.084 | 0.288 | 0.128 |

**Apareado por clip contra `ctrl_l40s`** (Wilcoxon, todos los escenarios juntos)
| brazo | métrica | n | ref | brazo | Δ | mejor | p |
|---|---|---|---|---|---|---|---|
| base | vel_mae | 30 | 4.212 | 9.258 | +5.046 | 1/30 | 0.000 |
| base | mov_ratio | 30 | 0.715 | 0.356 | -0.359 | 26/30 | 0.000 |
| base | mae_ay | 30 | 1.842 | 1.931 | +0.089 | 19/30 | 0.237 |
| base | jerk_rms | 30 | 1.820 | 2.710 | +0.890 | 17/30 | 0.626 |
| base | l_rot | 30 | 4.928 | 7.902 | +2.975 | 23/30 | 0.021 |
| base | l_rot_norm | 30 | 0.167 | 0.000 | -0.167 | 7/30 | 0.018 |
| equiv_vel | vel_mae | 30 | 4.212 | 5.225 | +1.013 | 11/30 | 0.003 |
| equiv_vel | mov_ratio | 30 | 0.715 | 0.645 | -0.070 | 20/30 | 0.105 |
| equiv_vel | mae_ay | 30 | 1.842 | 1.755 | -0.087 | 14/30 | 0.730 |
| equiv_vel | jerk_rms | 30 | 1.820 | 1.101 | -0.718 | 21/30 | 0.001 |
| equiv_vel | l_rot | 30 | 4.928 | 1.441 | -3.487 | 26/30 | 0.000 |
| equiv_vel | l_rot_norm | 30 | 0.167 | 0.144 | -0.023 | 6/30 | 0.959 |

### Checkpoint 875, en distribución (10 clips/escenario)
`e4vel/results/e4vel/paso_875/eval_results.json` · brazos: base, equiv_vel, ctrl_l40s · frames generados: 33

**vel_mae**
| brazo | free_fall | pendulum | bouncing |
|---|---|---|---|
| base | 11.709 | 5.620 | 9.958 |
| equiv_vel | 3.763 | 3.741 | 6.796 |
| ctrl_l40s | 5.154 | 4.263 | 5.399 |

**mov_ratio**
| brazo | free_fall | pendulum | bouncing |
|---|---|---|---|
| base | 0.429 | 0.449 | 0.093 |
| equiv_vel | 0.729 | 0.669 | 0.454 |
| ctrl_l40s | 0.615 | 0.573 | 0.588 |

**mae_ay**
| brazo | free_fall | pendulum | bouncing |
|---|---|---|---|
| base | 1.283 | 1.502 | 3.232 |
| equiv_vel | 0.822 | 0.659 | 3.888 |
| ctrl_l40s | 0.646 | 0.692 | 3.911 |

**jerk_rms**
| brazo | free_fall | pendulum | bouncing | ood_t2v |
|---|---|---|---|---|
| base | 4.887 | 3.296 | 0.607 | 1.147 |
| equiv_vel | 1.343 | 0.732 | 1.693 | 1.768 |
| ctrl_l40s | 1.236 | 0.765 | 2.636 | 2.102 |

**l_rot**
| brazo | free_fall | pendulum | bouncing |
|---|---|---|---|
| base | 21.590 | 9.545 | 0.257 |
| equiv_vel | 1.592 | 1.055 | 2.450 |
| ctrl_l40s | 1.019 | 0.902 | 5.696 |

**l_rot_norm**
| brazo | free_fall | pendulum | bouncing |
|---|---|---|---|
| base | 0.000 | 0.000 | 0.006 |
| equiv_vel | 0.000 | 0.076 | 0.058 |
| ctrl_l40s | 0.000 | 0.132 | 0.328 |

**Apareado por clip contra `ctrl_l40s`** (Wilcoxon, todos los escenarios juntos)
| brazo | métrica | n | ref | brazo | Δ | mejor | p |
|---|---|---|---|---|---|---|---|
| base | vel_mae | 30 | 4.939 | 9.096 | +4.157 | 4/30 | 0.000 |
| base | mov_ratio | 30 | 0.592 | 0.324 | -0.268 | 26/30 | 0.003 |
| base | mae_ay | 30 | 1.750 | 2.006 | +0.256 | 18/30 | 0.371 |
| base | jerk_rms | 30 | 1.546 | 2.930 | +1.384 | 19/30 | 0.670 |
| base | l_rot | 30 | 2.539 | 10.464 | +7.925 | 20/30 | 0.299 |
| base | l_rot_norm | 30 | 0.153 | 0.002 | -0.151 | 6/30 | 0.028 |
| equiv_vel | vel_mae | 30 | 4.939 | 4.767 | -0.172 | 16/30 | 0.465 |
| equiv_vel | mov_ratio | 30 | 0.592 | 0.617 | +0.025 | 13/30 | 0.393 |
| equiv_vel | mae_ay | 30 | 1.750 | 1.790 | +0.040 | 10/30 | 0.124 |
| equiv_vel | jerk_rms | 30 | 1.546 | 1.256 | -0.290 | 17/30 | 0.280 |
| equiv_vel | l_rot | 30 | 2.539 | 1.699 | -0.840 | 12/30 | 0.700 |
| equiv_vel | l_rot_norm | 30 | 0.153 | 0.045 | -0.109 | 6/30 | 0.161 |

### Checkpoint 1000, en distribución (10 clips/escenario)
`e4vel/results/e4vel/paso_1000/eval_results.json` · brazos: base, equiv_vel, ctrl_l40s · frames generados: 33

**vel_mae**
| brazo | free_fall | pendulum | bouncing |
|---|---|---|---|
| base | 12.476 | 5.712 | 9.960 |
| equiv_vel | 3.506 | 3.835 | 7.359 |
| ctrl_l40s | 4.768 | 3.440 | 5.647 |

**mov_ratio**
| brazo | free_fall | pendulum | bouncing |
|---|---|---|---|
| base | 0.519 | 0.464 | 0.093 |
| equiv_vel | 0.778 | 0.680 | 0.380 |
| ctrl_l40s | 0.659 | 0.723 | 0.569 |

**mae_ay**
| brazo | free_fall | pendulum | bouncing |
|---|---|---|---|
| base | 1.155 | 1.519 | 3.234 |
| equiv_vel | 1.007 | 0.688 | 4.106 |
| ctrl_l40s | 0.861 | 0.714 | 3.935 |

**jerk_rms**
| brazo | free_fall | pendulum | bouncing | ood_t2v |
|---|---|---|---|---|
| base | 3.891 | 3.255 | 0.616 | 1.462 |
| equiv_vel | 1.596 | 0.740 | 2.172 | 2.169 |
| ctrl_l40s | 1.519 | 1.614 | 2.588 | 2.821 |

**l_rot**
| brazo | free_fall | pendulum | bouncing |
|---|---|---|---|
| base | 14.442 | 9.336 | 0.216 |
| equiv_vel | 2.423 | 1.001 | 4.222 |
| ctrl_l40s | 2.358 | 7.427 | 7.104 |

**l_rot_norm**
| brazo | free_fall | pendulum | bouncing |
|---|---|---|---|
| base | 0.025 | 0.000 | 0.000 |
| equiv_vel | 0.051 | 0.059 | 0.141 |
| ctrl_l40s | 0.040 | 0.436 | 0.215 |

**Apareado por clip contra `ctrl_l40s`** (Wilcoxon, todos los escenarios juntos)
| brazo | métrica | n | ref | brazo | Δ | mejor | p |
|---|---|---|---|---|---|---|---|
| base | vel_mae | 30 | 4.619 | 9.383 | +4.764 | 2/30 | 0.000 |
| base | mov_ratio | 30 | 0.651 | 0.359 | -0.292 | 26/30 | 0.000 |
| base | mae_ay | 30 | 1.837 | 1.969 | +0.133 | 19/30 | 0.253 |
| base | jerk_rms | 30 | 1.907 | 2.588 | +0.681 | 19/30 | 0.328 |
| base | l_rot | 30 | 5.630 | 7.998 | +2.368 | 24/30 | 0.009 |
| base | l_rot_norm | 30 | 0.231 | 0.008 | -0.222 | 13/30 | 0.002 |
| equiv_vel | vel_mae | 30 | 4.619 | 4.900 | +0.281 | 11/30 | 0.213 |
| equiv_vel | mov_ratio | 30 | 0.651 | 0.613 | -0.038 | 16/30 | 0.452 |
| equiv_vel | mae_ay | 30 | 1.837 | 1.934 | +0.097 | 9/30 | 0.050 |
| equiv_vel | jerk_rms | 30 | 1.907 | 1.503 | -0.404 | 19/30 | 0.253 |
| equiv_vel | l_rot | 30 | 5.630 | 2.549 | -3.081 | 21/30 | 0.021 |
| equiv_vel | l_rot_norm | 30 | 0.231 | 0.084 | -0.147 | 10/30 | 0.035 |

### Tabla final, en distribución (20 clips/escenario)
`e4vel/results/e4vel/indist/eval_results.json` · brazos: base, ctrl_s42, equiv_norm, equiv_vel, ctrl_l40s, quieto · frames generados: 33

**vel_mae**
| brazo | free_fall | pendulum | bouncing |
|---|---|---|---|
| base | 15.435 | 9.529 | 10.161 |
| ctrl_s42 | 4.430 | 4.379 | 6.328 |
| equiv_norm | 4.562 | 4.341 | 6.186 |
| equiv_vel | 3.234 | 4.826 | 6.967 |
| ctrl_l40s | 4.274 | 4.325 | 6.084 |
| quieto | 11.498 | 7.519 | 9.953 |

**mov_ratio**
| brazo | free_fall | pendulum | bouncing |
|---|---|---|---|
| base | 0.892 | 1.192 | 0.093 |
| ctrl_s42 | 0.662 | 0.656 | 0.517 |
| equiv_norm | 0.685 | 0.615 | 0.514 |
| equiv_vel | 0.797 | 0.588 | 0.410 |
| ctrl_l40s | 0.695 | 0.621 | 0.533 |
| quieto | 0.140 | 0.067 | 0.142 |

**mae_ay**
| brazo | free_fall | pendulum | bouncing |
|---|---|---|---|
| base | 1.402 | 1.691 | 3.129 |
| ctrl_s42 | 0.680 | 0.766 | 3.768 |
| equiv_norm | 0.719 | 0.740 | 3.799 |
| equiv_vel | 0.877 | 0.812 | 3.951 |
| ctrl_l40s | 0.935 | 0.748 | 3.802 |
| quieto | 0.000 | 0.826 | 3.170 |

**jerk_rms**
| brazo | free_fall | pendulum | bouncing | ood_t2v |
|---|---|---|---|---|
| base | 8.049 | 8.350 | 0.661 | 0.998 |
| ctrl_s42 | 1.244 | 0.912 | 2.576 | 2.341 |
| equiv_norm | 1.237 | 0.766 | 2.354 | 2.991 |
| equiv_vel | 1.404 | 0.661 | 2.419 | 2.105 |
| ctrl_l40s | 1.481 | 0.935 | 2.505 | 2.615 |
| quieto | 0.000 | 0.000 | 0.000 | — |

**l_rot**
| brazo | free_fall | pendulum | bouncing |
|---|---|---|---|
| base | 80.270 | 69.796 | 0.504 |
| ctrl_s42 | 2.594 | 2.168 | 6.995 |
| equiv_norm | 2.518 | 1.169 | 5.844 |
| equiv_vel | 2.484 | 0.908 | 5.244 |
| ctrl_l40s | 3.053 | 3.645 | 7.819 |
| quieto | 0.000 | 0.000 | 0.000 |

**l_rot_norm**
| brazo | free_fall | pendulum | bouncing |
|---|---|---|---|
| base | 0.000 | 0.000 | 0.042 |
| ctrl_s42 | 0.159 | 0.276 | 0.312 |
| equiv_norm | 0.125 | 0.162 | 0.292 |
| equiv_vel | 0.073 | 0.059 | 0.151 |
| ctrl_l40s | 0.159 | 0.296 | 0.365 |
| quieto | 0.000 | 0.000 | 0.000 |

**Apareado por clip contra `ctrl_l40s`** (Wilcoxon, todos los escenarios juntos)
| brazo | métrica | n | ref | brazo | Δ | mejor | p |
|---|---|---|---|---|---|---|---|
| base | vel_mae | 60 | 4.895 | 11.708 | +6.814 | 2/60 | 0.000 |
| base | mov_ratio | 60 | 0.616 | 0.726 | +0.109 | 49/60 | 0.003 |
| base | mae_ay | 60 | 1.828 | 2.074 | +0.246 | 35/60 | 0.566 |
| base | jerk_rms | 60 | 1.640 | 5.686 | +4.046 | 35/60 | 0.971 |
| base | l_rot | 60 | 4.839 | 50.190 | +45.351 | 46/60 | 0.075 |
| base | l_rot_norm | 60 | 0.273 | 0.014 | -0.259 | 23/60 | 0.000 |
| ctrl_s42 | vel_mae | 60 | 4.895 | 5.046 | +0.151 | 29/60 | 0.480 |
| ctrl_s42 | mov_ratio | 60 | 0.616 | 0.612 | -0.005 | 27/60 | 0.808 |
| ctrl_s42 | mae_ay | 60 | 1.828 | 1.738 | -0.090 | 30/60 | 0.361 |
| ctrl_s42 | jerk_rms | 60 | 1.640 | 1.577 | -0.063 | 33/60 | 0.397 |
| ctrl_s42 | l_rot | 60 | 4.839 | 3.919 | -0.920 | 34/60 | 0.257 |
| ctrl_s42 | l_rot_norm | 60 | 0.273 | 0.249 | -0.024 | 16/60 | 0.666 |
| equiv_norm | vel_mae | 60 | 4.895 | 5.030 | +0.135 | 33/60 | 0.611 |
| equiv_norm | mov_ratio | 60 | 0.616 | 0.605 | -0.012 | 28/60 | 0.906 |
| equiv_norm | mae_ay | 60 | 1.828 | 1.753 | -0.076 | 34/60 | 0.139 |
| equiv_norm | jerk_rms | 60 | 1.640 | 1.453 | -0.188 | 40/60 | 0.005 |
| equiv_norm | l_rot | 60 | 4.839 | 3.177 | -1.662 | 40/60 | 0.004 |
| equiv_norm | l_rot_norm | 60 | 0.273 | 0.193 | -0.080 | 18/60 | 0.254 |
| equiv_vel | vel_mae | 60 | 4.895 | 5.009 | +0.114 | 24/60 | 0.299 |
| equiv_vel | mov_ratio | 60 | 0.616 | 0.598 | -0.018 | 29/60 | 0.802 |
| equiv_vel | mae_ay | 60 | 1.828 | 1.880 | +0.052 | 24/60 | 0.126 |
| equiv_vel | jerk_rms | 60 | 1.640 | 1.495 | -0.145 | 35/60 | 0.067 |
| equiv_vel | l_rot | 60 | 4.839 | 2.879 | -1.960 | 39/60 | 0.002 |
| equiv_vel | l_rot_norm | 60 | 0.273 | 0.094 | -0.179 | 21/60 | 0.005 |
| quieto | vel_mae | 60 | 4.895 | 9.657 | +4.762 | 5/60 | 0.000 |
| quieto | mov_ratio | 60 | 0.616 | 0.116 | -0.500 | 55/60 | 0.000 |
| quieto | mae_ay | 60 | 1.828 | 1.332 | -0.496 | 47/60 | 0.000 |
| quieto | jerk_rms | 60 | 1.640 | 0.000 | -1.640 | 60/60 | 0.000 |
| quieto | l_rot | 60 | 4.839 | 0.000 | -4.839 | 60/60 | 0.000 |
| quieto | l_rot_norm | 60 | 0.273 | 0.000 | -0.273 | 23/60 | 0.000 |

### Tabla final, fuera de distribución: tiro vertical (15 clips)
`e4vel/results/e4vel/ood/eval_results.json` · brazos: base, ctrl_s42, equiv_norm, equiv_vel, ctrl_l40s, quieto · frames generados: 33

**vel_mae**
| brazo | vertical_throw |
|---|---|
| base | 10.616 |
| ctrl_s42 | 7.867 |
| equiv_norm | 8.098 |
| equiv_vel | 8.799 |
| ctrl_l40s | 7.634 |
| quieto | 8.899 |

**mov_ratio**
| brazo | vertical_throw |
|---|---|
| base | 0.794 |
| ctrl_s42 | 0.299 |
| equiv_norm | 0.230 |
| equiv_vel | 0.119 |
| ctrl_l40s | 0.427 |
| quieto | 0.216 |

**mae_ay**
| brazo | vertical_throw |
|---|---|
| base | 1.861 |
| ctrl_s42 | 1.477 |
| equiv_norm | 1.287 |
| equiv_vel | 1.212 |
| ctrl_l40s | 1.646 |
| quieto | 1.125 |

**jerk_rms**
| brazo | vertical_throw | ood_t2v |
|---|---|---|
| base | 4.429 | 0.870 |
| ctrl_s42 | 1.423 | 2.224 |
| equiv_norm | 1.345 | 2.607 |
| equiv_vel | 0.817 | 2.255 |
| ctrl_l40s | 1.651 | 2.634 |
| quieto | 0.000 | — |

**l_rot**
| brazo | vertical_throw |
|---|---|
| base | 13.691 |
| ctrl_s42 | 2.902 |
| equiv_norm | 2.936 |
| equiv_vel | 1.007 |
| ctrl_l40s | 5.380 |
| quieto | 0.000 |

**l_rot_norm**
| brazo | vertical_throw |
|---|---|
| base | 0.000 |
| ctrl_s42 | 0.344 |
| equiv_norm | 0.270 |
| equiv_vel | 0.178 |
| ctrl_l40s | 0.442 |
| quieto | 0.000 |

**Apareado por clip contra `ctrl_l40s`** (Wilcoxon, todos los escenarios juntos)
| brazo | métrica | n | ref | brazo | Δ | mejor | p |
|---|---|---|---|---|---|---|---|
| base | vel_mae | 15 | 7.634 | 10.616 | +2.982 | 6/15 | 0.277 |
| base | mov_ratio | 15 | 0.427 | 0.794 | +0.366 | 9/15 | 0.804 |
| base | mae_ay | 15 | 1.646 | 1.861 | +0.215 | 6/15 | 0.804 |
| base | jerk_rms | 15 | 1.651 | 4.429 | +2.778 | 4/15 | 0.083 |
| base | l_rot | 15 | 5.380 | 13.691 | +8.311 | 10/15 | 0.561 |
| base | l_rot_norm | 15 | 0.442 | 0.000 | -0.442 | 10/15 | 0.005 |
| ctrl_s42 | vel_mae | 15 | 7.634 | 7.867 | +0.233 | 7/15 | 0.890 |
| ctrl_s42 | mov_ratio | 15 | 0.427 | 0.299 | -0.128 | 12/15 | 0.026 |
| ctrl_s42 | mae_ay | 15 | 1.646 | 1.477 | -0.168 | 10/15 | 0.188 |
| ctrl_s42 | jerk_rms | 15 | 1.651 | 1.423 | -0.228 | 9/15 | 0.330 |
| ctrl_s42 | l_rot | 15 | 5.380 | 2.902 | -2.478 | 11/15 | 0.135 |
| ctrl_s42 | l_rot_norm | 15 | 0.442 | 0.344 | -0.097 | 7/15 | 0.530 |
| equiv_norm | vel_mae | 15 | 7.634 | 8.098 | +0.464 | 5/15 | 0.151 |
| equiv_norm | mov_ratio | 15 | 0.427 | 0.230 | -0.197 | 11/15 | 0.030 |
| equiv_norm | mae_ay | 15 | 1.646 | 1.287 | -0.359 | 10/15 | 0.055 |
| equiv_norm | jerk_rms | 15 | 1.651 | 1.345 | -0.306 | 9/15 | 0.303 |
| equiv_norm | l_rot | 15 | 5.380 | 2.936 | -2.444 | 11/15 | 0.121 |
| equiv_norm | l_rot_norm | 15 | 0.442 | 0.270 | -0.172 | 9/15 | 0.028 |
| equiv_vel | vel_mae | 15 | 7.634 | 8.799 | +1.164 | 2/15 | 0.005 |
| equiv_vel | mov_ratio | 15 | 0.427 | 0.119 | -0.308 | 13/15 | 0.001 |
| equiv_vel | mae_ay | 15 | 1.646 | 1.212 | -0.433 | 11/15 | 0.030 |
| equiv_vel | jerk_rms | 15 | 1.651 | 0.817 | -0.834 | 13/15 | 0.005 |
| equiv_vel | l_rot | 15 | 5.380 | 1.007 | -4.373 | 12/15 | 0.005 |
| equiv_vel | l_rot_norm | 15 | 0.442 | 0.178 | -0.264 | 10/15 | 0.041 |
| quieto | vel_mae | 15 | 7.634 | 8.899 | +1.265 | 6/15 | 0.073 |
| quieto | mov_ratio | 15 | 0.427 | 0.216 | -0.211 | 11/15 | 0.026 |
| quieto | mae_ay | 15 | 1.646 | 1.125 | -0.521 | 10/15 | 0.008 |
| quieto | jerk_rms | 15 | 1.651 | 0.000 | -1.651 | 15/15 | 0.000 |
| quieto | l_rot | 15 | 5.380 | 0.000 | -5.380 | 15/15 | 0.000 |
| quieto | l_rot_norm | 15 | 0.442 | 0.000 | -0.442 | 10/15 | 0.005 |

### Ajuste de parámetros físicos por checkpoint (T2V, generación libre)
Mediana por escenario, n=10 clips. `param1`/`param2`: caída libre aceleración y CV de velocidad; péndulo ω y amplitud fin/inicio; rebote g y dispersión entre tramos.

| paso | escenario | residuo GT | residuo brazo | param1 GT | param1 brazo | param2 GT | param2 brazo | det |
|---|---|---|---|---|---|---|---|---|
| 125 | free_fall | 0.020 | 0.047 | 0.306 | 0.006 | 0.326 | 0.645 | 85% |
| 125 | pendulum | 0.030 | 0.215 | 0.168 | 0.150 | 0.710 | 1.245 | 84% |
| 125 | bouncing | 0.016 | 0.020 | 3.281 | 0.639 | 0.229 | 2.367 | 73% |
| 375 | free_fall | 0.020 | 0.084 | 0.306 | 0.045 | 0.326 | 2.572 | 68% |
| 375 | pendulum | 0.030 | nan | 0.168 | 0.257 | 0.710 | nan | 69% |
| 375 | bouncing | 0.016 | 0.027 | 3.281 | 0.523 | 0.229 | nan | 63% |
| 500 | free_fall | 0.020 | 0.031 | 0.306 | 0.089 | 0.326 | 0.402 | 78% |
| 500 | pendulum | 0.030 | 0.249 | 0.168 | 0.194 | 0.710 | 1.065 | 72% |
| 500 | bouncing | 0.016 | 0.036 | 3.281 | 0.641 | 0.229 | nan | 58% |
| 625 | free_fall | 0.020 | 0.047 | 0.306 | 0.063 | 0.326 | 0.472 | 58% |
| 625 | pendulum | 0.030 | 0.141 | 0.168 | 0.275 | 0.710 | 1.473 | 70% |
| 625 | bouncing | 0.016 | nan | 3.281 | nan | 0.229 | nan | 37% |
| 750 | free_fall | 0.020 | 0.096 | 0.306 | -0.001 | 0.326 | 1.242 | 75% |
| 750 | pendulum | 0.030 | 0.165 | 0.168 | 0.162 | 0.710 | 0.909 | 90% |
| 750 | bouncing | 0.016 | 0.047 | 3.281 | 0.216 | 0.229 | nan | 73% |
| 1000 | free_fall | 0.020 | 0.015 | 0.306 | 0.217 | 0.326 | 0.285 | 68% |
| 1000 | pendulum | 0.030 | 0.231 | 0.168 | 0.262 | 0.710 | 1.660 | 77% |
| 1000 | bouncing | 0.016 | 0.022 | 3.281 | 0.749 | 0.229 | nan | 52% |

### Equivarianza directa, checkpoint 250
Diferencia media de píxeles entre la generación con condicionamiento original y la rotada-des-rotada (menor = más equivariante). Calibración: 0,3 perfecto, 15 ninguna.

| brazo | media | n | bouncing | free_fall | pendulum |
|---|---|---|---|---|---|
| equiv_vel250 | 5.07 | 9 | 6.10 | 4.27 | 4.83 |
| ctrl_rep250 | 4.78 | 9 | 5.70 | 3.87 | 4.77 |

### Equivarianza directa, checkpoint 1000
Diferencia media de píxeles entre la generación con condicionamiento original y la rotada-des-rotada (menor = más equivariante). Calibración: 0,3 perfecto, 15 ninguna.

| brazo | media | n | bouncing | free_fall | pendulum |
|---|---|---|---|---|---|
| equiv_vel1000 | 4.99 | 9 | 5.63 | 4.10 | 5.23 |
| ctrl_rep1000 | 5.38 | 9 | 6.63 | 4.20 | 5.30 |

### Curvas de entrenamiento
`vel_run/checkpoints_sana/stage2_equiv_vel_s42_20260907-140554_5de5e17a-fixpiso-vel/training_log.jsonl` · 1000 pasos · λ_rot = 0.0047

| ventana | L_dif | L_rot/rama | cos_ramas | escala_ramas | razón grad | cos grad fís/dif | ‖q‖ | salteados | s/paso |
|---|---|---|---|---|---|---|---|---|---|
| 1-100 | 0.265 | 0.183 | 0.806 | 1.080 | 0.785 | 0.014 | 5.472 | 11/100 | 69.939 |
| 101-200 | 0.296 | 0.147 | 0.827 | 1.027 | 1.590 | -0.002 | 6.096 | 17/100 | 62.076 |
| 201-300 | 0.268 | 0.176 | 0.827 | 1.015 | 0.670 | -0.012 | 5.799 | 14/100 | 57.955 |
| 301-400 | 0.241 | 0.120 | 0.860 | 1.012 | 0.554 | 0.015 | 6.797 | 16/100 | 49.841 |
| 401-500 | 0.273 | 0.125 | 0.871 | 0.984 | 0.590 | -0.003 | 7.530 | 6/100 | 54.513 |
| 501-600 | 0.290 | 0.132 | 0.859 | 1.022 | 0.702 | -0.011 | 6.735 | 15/100 | 58.979 |
| 601-700 | 0.289 | 0.158 | 0.842 | 0.965 | 1.360 | 0.003 | 7.094 | 7/100 | 64.133 |
| 701-800 | 0.279 | 0.142 | 0.844 | 0.993 | 0.715 | -0.001 | 6.245 | 18/100 | 54.242 |
| 801-900 | 0.261 | 0.141 | 0.848 | 1.011 | 0.684 | -0.006 | 5.905 | 19/100 | 48.908 |
| 901-1000 | 0.240 | 0.127 | 0.861 | 1.043 | 0.561 | -0.008 | 6.833 | 8/100 | 56.516 |

**Prueba de tendencia preregistrada** (primeros 100 pasos vs últimos 100, Mann-Whitney)
| cantidad | primeros | últimos | p |
|---|---|---|---|
| loss_rotation | 0.1834 | 0.1266 | 0.0426 |
| cos_ramas | 0.8064 | 0.8614 | 0.0622 |
| escala_ramas | 0.1857 | 0.1184 | 0.0012 |

**Validación de difusión**: 50: 0.0903 · 100: 0.0795 · 150: 0.0782 · 200: 0.0725 · 250: 0.0698 · 300: 0.0727 · 350: 0.0720 · 400: 0.0722 · 450: 0.0780 · 500: 0.0808 · 550: 0.0732 · 600: 0.0808 · 650: 0.0690 · 700: 0.0936 · 750: 0.0846 · 800: 0.0770 · 850: 0.0733 · 900: 0.0818 · 950: 0.0693 · 1000: 0.0724

## 2. Corrida con la pérdida sobre ACELERACIÓN (λ = 1,54e-03, cortada en el 807)

### Checkpoint 125, en distribución (10 clips/escenario)
`e4lambdafix/results/e4lambdafix/paso_125/eval_results.json` · brazos: base, equiv_lambdafix, ctrl_l40s · frames generados: 33

**vel_mae**
| brazo | free_fall | pendulum | bouncing |
|---|---|---|---|
| base | 11.681 | 6.133 | 9.950 |
| equiv_lambdafix | 3.518 | 3.552 | 5.761 |
| ctrl_l40s | 4.471 | 3.946 | 6.308 |

**mov_ratio**
| brazo | free_fall | pendulum | bouncing |
|---|---|---|---|
| base | 0.430 | 0.517 | 0.094 |
| equiv_lambdafix | 0.801 | 1.141 | 0.614 |
| ctrl_l40s | 0.683 | 0.633 | 0.497 |

**mae_ay**
| brazo | free_fall | pendulum | bouncing |
|---|---|---|---|
| base | 1.080 | 1.517 | 3.240 |
| equiv_lambdafix | 1.826 | 0.972 | 4.227 |
| ctrl_l40s | 0.768 | 0.680 | 3.816 |

**jerk_rms**
| brazo | free_fall | pendulum | bouncing | ood_t2v |
|---|---|---|---|---|
| base | 3.713 | 3.516 | 0.615 | 0.976 |
| equiv_lambdafix | 2.225 | 1.443 | 3.467 | 2.297 |
| ctrl_l40s | 1.354 | 0.812 | 2.514 | 3.054 |

**l_rot**
| brazo | free_fall | pendulum | bouncing |
|---|---|---|---|
| base | 11.430 | 12.560 | 0.277 |
| equiv_lambdafix | 5.637 | 2.750 | 11.248 |
| ctrl_l40s | 2.629 | 2.176 | 5.408 |

**l_rot_norm**
| brazo | free_fall | pendulum | bouncing |
|---|---|---|---|
| base | 0.000 | 0.000 | 0.000 |
| equiv_lambdafix | 0.284 | 0.045 | 0.306 |
| ctrl_l40s | 0.012 | 0.050 | 0.205 |

**Apareado por clip contra `ctrl_l40s`** (Wilcoxon, todos los escenarios juntos)
| brazo | métrica | n | ref | brazo | Δ | mejor | p |
|---|---|---|---|---|---|---|---|
| base | vel_mae | 30 | 4.908 | 9.255 | +4.346 | 1/30 | 0.000 |
| base | mov_ratio | 30 | 0.604 | 0.347 | -0.257 | 27/30 | 0.000 |
| base | mae_ay | 30 | 1.754 | 1.945 | +0.191 | 20/30 | 0.339 |
| base | jerk_rms | 30 | 1.560 | 2.615 | +1.055 | 20/30 | 0.516 |
| base | l_rot | 30 | 3.404 | 8.089 | +4.685 | 22/30 | 0.036 |
| base | l_rot_norm | 30 | 0.089 | 0.000 | -0.089 | 8/30 | 0.012 |
| equiv_lambdafix | vel_mae | 30 | 4.908 | 4.277 | -0.631 | 21/30 | 0.050 |
| equiv_lambdafix | mov_ratio | 30 | 0.604 | 0.852 | +0.248 | 3/30 | 0.000 |
| equiv_lambdafix | mae_ay | 30 | 1.754 | 2.342 | +0.587 | 7/30 | 0.000 |
| equiv_lambdafix | jerk_rms | 30 | 1.560 | 2.378 | +0.818 | 5/30 | 0.000 |
| equiv_lambdafix | l_rot | 30 | 3.404 | 6.545 | +3.141 | 4/30 | 0.000 |
| equiv_lambdafix | l_rot_norm | 30 | 0.089 | 0.212 | +0.123 | 5/30 | 0.124 |

### Checkpoint 250, en distribución (10 clips/escenario)
`e4lambdafix/results/e4lambdafix/paso_250/eval_results.json` · brazos: base, equiv_lambdafix, ctrl_l40s · frames generados: 33

**vel_mae**
| brazo | free_fall | pendulum | bouncing |
|---|---|---|---|
| base | 11.250 | 6.070 | 9.951 |
| equiv_lambdafix | 4.506 | 4.188 | 5.683 |
| ctrl_l40s | 4.606 | 3.715 | 6.127 |

**mov_ratio**
| brazo | free_fall | pendulum | bouncing |
|---|---|---|---|
| base | 0.415 | 0.509 | 0.094 |
| equiv_lambdafix | 0.737 | 1.239 | 0.863 |
| ctrl_l40s | 0.664 | 0.652 | 0.515 |

**mae_ay**
| brazo | free_fall | pendulum | bouncing |
|---|---|---|---|
| base | 1.132 | 1.451 | 3.227 |
| equiv_lambdafix | 1.793 | 0.920 | 3.920 |
| ctrl_l40s | 0.786 | 0.640 | 3.803 |

**jerk_rms**
| brazo | free_fall | pendulum | bouncing | ood_t2v |
|---|---|---|---|---|
| base | 4.116 | 3.293 | 0.596 | 0.997 |
| equiv_lambdafix | 2.041 | 1.588 | 3.876 | 1.573 |
| ctrl_l40s | 1.489 | 0.802 | 2.471 | 2.473 |

**l_rot**
| brazo | free_fall | pendulum | bouncing |
|---|---|---|---|
| base | 18.409 | 10.167 | 0.199 |
| equiv_lambdafix | 5.752 | 3.488 | 15.135 |
| ctrl_l40s | 2.267 | 3.200 | 5.527 |

**l_rot_norm**
| brazo | free_fall | pendulum | bouncing |
|---|---|---|---|
| base | 0.000 | 0.000 | 0.000 |
| equiv_lambdafix | 0.000 | 0.005 | 0.272 |
| ctrl_l40s | 0.090 | 0.456 | 0.237 |

**Apareado por clip contra `ctrl_l40s`** (Wilcoxon, todos los escenarios juntos)
| brazo | métrica | n | ref | brazo | Δ | mejor | p |
|---|---|---|---|---|---|---|---|
| base | vel_mae | 30 | 4.816 | 9.091 | +4.275 | 0/30 | 0.000 |
| base | mov_ratio | 30 | 0.610 | 0.339 | -0.271 | 27/30 | 0.000 |
| base | mae_ay | 30 | 1.743 | 1.937 | +0.194 | 20/30 | 0.309 |
| base | jerk_rms | 30 | 1.587 | 2.668 | +1.081 | 18/30 | 0.612 |
| base | l_rot | 30 | 3.665 | 9.592 | +5.927 | 24/30 | 0.040 |
| base | l_rot_norm | 30 | 0.261 | 0.000 | -0.261 | 14/30 | 0.001 |
| equiv_lambdafix | vel_mae | 30 | 4.816 | 4.792 | -0.024 | 13/30 | 0.871 |
| equiv_lambdafix | mov_ratio | 30 | 0.610 | 0.946 | +0.336 | 4/30 | 0.000 |
| equiv_lambdafix | mae_ay | 30 | 1.743 | 2.211 | +0.468 | 7/30 | 0.002 |
| equiv_lambdafix | jerk_rms | 30 | 1.587 | 2.502 | +0.914 | 3/30 | 0.000 |
| equiv_lambdafix | l_rot | 30 | 3.665 | 8.125 | +4.460 | 8/30 | 0.001 |
| equiv_lambdafix | l_rot_norm | 30 | 0.261 | 0.092 | -0.169 | 11/30 | 0.047 |

### Checkpoint 375, en distribución (10 clips/escenario)
`e4lambdafix/results/e4lambdafix/paso_375/eval_results.json` · brazos: base, equiv_lambdafix, ctrl_l40s · frames generados: 33

**vel_mae**
| brazo | free_fall | pendulum | bouncing |
|---|---|---|---|
| base | 11.831 | 5.752 | 9.959 |
| equiv_lambdafix | 5.335 | 3.707 | 5.534 |
| ctrl_l40s | 4.862 | 3.801 | 6.017 |

**mov_ratio**
| brazo | free_fall | pendulum | bouncing |
|---|---|---|---|
| base | 0.422 | 0.455 | 0.093 |
| equiv_lambdafix | 0.655 | 0.825 | 0.726 |
| ctrl_l40s | 0.622 | 0.631 | 0.550 |

**mae_ay**
| brazo | free_fall | pendulum | bouncing |
|---|---|---|---|
| base | 1.077 | 1.435 | 3.232 |
| equiv_lambdafix | 1.640 | 0.719 | 3.986 |
| ctrl_l40s | 0.762 | 0.599 | 4.118 |

**jerk_rms**
| brazo | free_fall | pendulum | bouncing | ood_t2v |
|---|---|---|---|---|
| base | 3.768 | 2.964 | 0.595 | 0.650 |
| equiv_lambdafix | 2.148 | 1.102 | 3.296 | 2.092 |
| ctrl_l40s | 1.359 | 0.732 | 3.317 | 2.952 |

**l_rot**
| brazo | free_fall | pendulum | bouncing |
|---|---|---|---|
| base | 19.288 | 6.742 | 0.240 |
| equiv_lambdafix | 4.381 | 1.903 | 11.122 |
| ctrl_l40s | 1.541 | 1.995 | 8.427 |

**l_rot_norm**
| brazo | free_fall | pendulum | bouncing |
|---|---|---|---|
| base | 0.027 | 0.000 | 0.000 |
| equiv_lambdafix | 0.053 | 0.103 | 0.312 |
| ctrl_l40s | 0.053 | 0.349 | 0.240 |

**Apareado por clip contra `ctrl_l40s`** (Wilcoxon, todos los escenarios juntos)
| brazo | métrica | n | ref | brazo | Δ | mejor | p |
|---|---|---|---|---|---|---|---|
| base | vel_mae | 30 | 4.893 | 9.181 | +4.288 | 1/30 | 0.000 |
| base | mov_ratio | 30 | 0.601 | 0.323 | -0.278 | 27/30 | 0.000 |
| base | mae_ay | 30 | 1.826 | 1.915 | +0.089 | 19/30 | 0.309 |
| base | jerk_rms | 30 | 1.803 | 2.442 | +0.640 | 20/30 | 0.280 |
| base | l_rot | 30 | 3.987 | 8.756 | +4.769 | 23/30 | 0.029 |
| base | l_rot_norm | 30 | 0.214 | 0.009 | -0.205 | 12/30 | 0.002 |
| equiv_lambdafix | vel_mae | 30 | 4.893 | 4.859 | -0.034 | 15/30 | 0.984 |
| equiv_lambdafix | mov_ratio | 30 | 0.601 | 0.735 | +0.134 | 8/30 | 0.001 |
| equiv_lambdafix | mae_ay | 30 | 1.826 | 2.115 | +0.289 | 8/30 | 0.026 |
| equiv_lambdafix | jerk_rms | 30 | 1.803 | 2.182 | +0.379 | 9/30 | 0.096 |
| equiv_lambdafix | l_rot | 30 | 3.987 | 5.802 | +1.814 | 11/30 | 0.061 |
| equiv_lambdafix | l_rot_norm | 30 | 0.214 | 0.156 | -0.058 | 11/30 | 0.314 |

### Checkpoint 500, en distribución (10 clips/escenario)
`e4lambdafix/results/e4lambdafix/paso_500/eval_results.json` · brazos: base, equiv_lambdafix, ctrl_l40s · frames generados: 33

**vel_mae**
| brazo | free_fall | pendulum | bouncing |
|---|---|---|---|
| base | 12.031 | 5.961 | 9.961 |
| equiv_lambdafix | 6.157 | 4.347 | 6.009 |
| ctrl_l40s | 4.390 | 3.750 | 5.417 |

**mov_ratio**
| brazo | free_fall | pendulum | bouncing |
|---|---|---|---|
| base | 0.511 | 0.474 | 0.093 |
| equiv_lambdafix | 0.530 | 0.605 | 0.767 |
| ctrl_l40s | 0.683 | 0.791 | 0.578 |

**mae_ay**
| brazo | free_fall | pendulum | bouncing |
|---|---|---|---|
| base | 1.147 | 1.380 | 3.231 |
| equiv_lambdafix | 1.258 | 0.759 | 3.794 |
| ctrl_l40s | 0.813 | 0.742 | 4.005 |

**jerk_rms**
| brazo | free_fall | pendulum | bouncing | ood_t2v |
|---|---|---|---|---|
| base | 3.771 | 3.022 | 0.605 | 1.022 |
| equiv_lambdafix | 1.572 | 0.924 | 3.158 | 2.701 |
| ctrl_l40s | 1.704 | 1.104 | 2.882 | 2.119 |

**l_rot**
| brazo | free_fall | pendulum | bouncing |
|---|---|---|---|
| base | 18.653 | 9.231 | 0.259 |
| equiv_lambdafix | 2.608 | 1.987 | 10.081 |
| ctrl_l40s | 1.952 | 3.045 | 8.498 |

**l_rot_norm**
| brazo | free_fall | pendulum | bouncing |
|---|---|---|---|
| base | 0.000 | 0.000 | 0.013 |
| equiv_lambdafix | 0.156 | 0.301 | 0.293 |
| ctrl_l40s | 0.000 | 0.168 | 0.280 |

**Apareado por clip contra `ctrl_l40s`** (Wilcoxon, todos los escenarios juntos)
| brazo | métrica | n | ref | brazo | Δ | mejor | p |
|---|---|---|---|---|---|---|---|
| base | vel_mae | 30 | 4.519 | 9.317 | +4.799 | 1/30 | 0.000 |
| base | mov_ratio | 30 | 0.684 | 0.359 | -0.325 | 27/30 | 0.000 |
| base | mae_ay | 30 | 1.854 | 1.920 | +0.066 | 21/30 | 0.129 |
| base | jerk_rms | 30 | 1.897 | 2.466 | +0.569 | 20/30 | 0.198 |
| base | l_rot | 30 | 4.498 | 9.381 | +4.883 | 23/30 | 0.009 |
| base | l_rot_norm | 30 | 0.149 | 0.004 | -0.145 | 8/30 | 0.012 |
| equiv_lambdafix | vel_mae | 30 | 4.519 | 5.504 | +0.985 | 6/30 | 0.001 |
| equiv_lambdafix | mov_ratio | 30 | 0.684 | 0.634 | -0.050 | 19/30 | 0.198 |
| equiv_lambdafix | mae_ay | 30 | 1.854 | 1.937 | +0.084 | 13/30 | 0.655 |
| equiv_lambdafix | jerk_rms | 30 | 1.897 | 1.885 | -0.012 | 15/30 | 0.808 |
| equiv_lambdafix | l_rot | 30 | 4.498 | 4.892 | +0.394 | 15/30 | 0.984 |
| equiv_lambdafix | l_rot_norm | 30 | 0.149 | 0.250 | +0.101 | 6/30 | 0.438 |

### Checkpoint 625, en distribución (10 clips/escenario)
`e4lambdafix/results/e4lambdafix/paso_625/eval_results.json` · brazos: base, equiv_lambdafix, ctrl_l40s · frames generados: 33

**vel_mae**
| brazo | free_fall | pendulum | bouncing |
|---|---|---|---|
| base | 12.103 | 5.828 | 9.969 |
| equiv_lambdafix | 5.384 | 4.343 | 6.169 |
| ctrl_l40s | 5.637 | 3.904 | 6.287 |

**mov_ratio**
| brazo | free_fall | pendulum | bouncing |
|---|---|---|---|
| base | 0.467 | 0.477 | 0.092 |
| equiv_lambdafix | 0.616 | 0.616 | 0.699 |
| ctrl_l40s | 0.575 | 0.649 | 0.487 |

**mae_ay**
| brazo | free_fall | pendulum | bouncing |
|---|---|---|---|
| base | 1.143 | 1.431 | 3.233 |
| equiv_lambdafix | 0.871 | 0.818 | 4.057 |
| ctrl_l40s | 0.934 | 0.690 | 3.875 |

**jerk_rms**
| brazo | free_fall | pendulum | bouncing | ood_t2v |
|---|---|---|---|---|
| base | 4.972 | 3.455 | 0.599 | 1.165 |
| equiv_lambdafix | 1.455 | 0.680 | 3.966 | 1.932 |
| ctrl_l40s | 1.332 | 1.121 | 2.169 | 2.589 |

**l_rot**
| brazo | free_fall | pendulum | bouncing |
|---|---|---|---|
| base | 24.196 | 9.768 | 0.225 |
| equiv_lambdafix | 2.058 | 3.569 | 10.774 |
| ctrl_l40s | 3.216 | 4.773 | 6.186 |

**l_rot_norm**
| brazo | free_fall | pendulum | bouncing |
|---|---|---|---|
| base | 0.008 | 0.000 | 0.000 |
| equiv_lambdafix | 0.032 | 0.489 | 0.270 |
| ctrl_l40s | 0.170 | 0.478 | 0.538 |

**Apareado por clip contra `ctrl_l40s`** (Wilcoxon, todos los escenarios juntos)
| brazo | métrica | n | ref | brazo | Δ | mejor | p |
|---|---|---|---|---|---|---|---|
| base | vel_mae | 30 | 5.276 | 9.300 | +4.024 | 2/30 | 0.000 |
| base | mov_ratio | 30 | 0.571 | 0.346 | -0.225 | 26/30 | 0.000 |
| base | mae_ay | 30 | 1.833 | 1.936 | +0.103 | 20/30 | 0.070 |
| base | jerk_rms | 30 | 1.541 | 3.009 | +1.468 | 18/30 | 0.670 |
| base | l_rot | 30 | 4.725 | 11.396 | +6.671 | 24/30 | 0.004 |
| base | l_rot_norm | 30 | 0.395 | 0.003 | -0.393 | 18/30 | 0.000 |
| equiv_lambdafix | vel_mae | 30 | 5.276 | 5.299 | +0.023 | 16/30 | 0.598 |
| equiv_lambdafix | mov_ratio | 30 | 0.571 | 0.643 | +0.073 | 10/30 | 0.031 |
| equiv_lambdafix | mae_ay | 30 | 1.833 | 1.915 | +0.083 | 15/30 | 0.404 |
| equiv_lambdafix | jerk_rms | 30 | 1.541 | 2.034 | +0.493 | 12/30 | 0.213 |
| equiv_lambdafix | l_rot | 30 | 4.725 | 5.467 | +0.742 | 17/30 | 0.824 |
| equiv_lambdafix | l_rot_norm | 30 | 0.395 | 0.264 | -0.132 | 14/30 | 0.099 |

### Checkpoint 750, en distribución (10 clips/escenario)
`e4lambdafix/results/e4lambdafix/paso_750/eval_results.json` · brazos: base, equiv_lambdafix, ctrl_l40s · frames generados: 33

**vel_mae**
| brazo | free_fall | pendulum | bouncing |
|---|---|---|---|
| base | 12.097 | 5.746 | 9.963 |
| equiv_lambdafix | 9.014 | 4.925 | 7.698 |
| ctrl_l40s | 4.209 | 3.381 | 5.095 |

**mov_ratio**
| brazo | free_fall | pendulum | bouncing |
|---|---|---|---|
| base | 0.476 | 0.462 | 0.093 |
| equiv_lambdafix | 0.300 | 0.429 | 0.394 |
| ctrl_l40s | 0.702 | 0.781 | 0.657 |

**mae_ay**
| brazo | free_fall | pendulum | bouncing |
|---|---|---|---|
| base | 1.246 | 1.487 | 3.231 |
| equiv_lambdafix | 0.733 | 0.763 | 3.520 |
| ctrl_l40s | 0.743 | 0.664 | 4.041 |

**jerk_rms**
| brazo | free_fall | pendulum | bouncing | ood_t2v |
|---|---|---|---|---|
| base | 4.567 | 3.108 | 0.603 | 0.813 |
| equiv_lambdafix | 0.820 | 0.552 | 1.563 | 1.944 |
| ctrl_l40s | 1.377 | 1.024 | 2.922 | 2.873 |

**l_rot**
| brazo | free_fall | pendulum | bouncing |
|---|---|---|---|
| base | 18.991 | 7.539 | 0.260 |
| equiv_lambdafix | 1.708 | 0.696 | 4.850 |
| ctrl_l40s | 2.094 | 4.498 | 7.882 |

**l_rot_norm**
| brazo | free_fall | pendulum | bouncing |
|---|---|---|---|
| base | 0.003 | 0.000 | 0.000 |
| equiv_lambdafix | 0.297 | 0.280 | 0.525 |
| ctrl_l40s | 0.062 | 0.303 | 0.148 |

**Apareado por clip contra `ctrl_l40s`** (Wilcoxon, todos los escenarios juntos)
| brazo | métrica | n | ref | brazo | Δ | mejor | p |
|---|---|---|---|---|---|---|---|
| base | vel_mae | 30 | 4.229 | 9.268 | +5.040 | 1/30 | 0.000 |
| base | mov_ratio | 30 | 0.713 | 0.344 | -0.369 | 28/30 | 0.000 |
| base | mae_ay | 30 | 1.816 | 1.988 | +0.172 | 19/30 | 0.262 |
| base | jerk_rms | 30 | 1.775 | 2.759 | +0.985 | 19/30 | 0.477 |
| base | l_rot | 30 | 4.825 | 8.930 | +4.105 | 23/30 | 0.026 |
| base | l_rot_norm | 30 | 0.171 | 0.001 | -0.170 | 8/30 | 0.011 |
| equiv_lambdafix | vel_mae | 30 | 4.229 | 7.212 | +2.984 | 1/30 | 0.000 |
| equiv_lambdafix | mov_ratio | 30 | 0.713 | 0.374 | -0.339 | 30/30 | 0.000 |
| equiv_lambdafix | mae_ay | 30 | 1.816 | 1.672 | -0.144 | 17/30 | 0.280 |
| equiv_lambdafix | jerk_rms | 30 | 1.775 | 0.978 | -0.796 | 29/30 | 0.000 |
| equiv_lambdafix | l_rot | 30 | 4.825 | 2.418 | -2.407 | 24/30 | 0.000 |
| equiv_lambdafix | l_rot_norm | 30 | 0.171 | 0.367 | +0.197 | 4/30 | 0.023 |

### Ajuste de parámetros físicos por checkpoint
Mediana por escenario, n=10 clips. `param1`/`param2`: caída libre aceleración y CV de velocidad; péndulo ω y amplitud fin/inicio; rebote g y dispersión entre tramos.

| paso | escenario | residuo GT | residuo brazo | param1 GT | param1 brazo | param2 GT | param2 brazo | det |
|---|---|---|---|---|---|---|---|---|
| 125 | free_fall | 0.020 | 0.034 | 0.306 | -0.596 | 0.326 | 0.600 | 51% |
| 125 | pendulum | 0.030 | 0.136 | 0.168 | 0.150 | 0.710 | 1.248 | 87% |
| 125 | bouncing | 0.016 | 0.063 | 3.281 | nan | 0.229 | nan | 64% |
| 250 | free_fall | 0.020 | 0.159 | 0.306 | 0.248 | 0.326 | 3.134 | 86% |
| 250 | pendulum | 0.030 | 0.258 | 0.168 | 0.150 | 0.710 | 0.843 | 80% |
| 250 | bouncing | 0.016 | nan | 3.281 | 1.811 | 0.229 | nan | 67% |
| 375 | free_fall | 0.020 | 0.166 | 0.306 | 0.191 | 0.326 | 2.323 | 52% |
| 375 | pendulum | 0.030 | 0.127 | 0.168 | nan | 0.710 | 1.501 | 71% |
| 375 | bouncing | 0.016 | nan | 3.281 | 0.849 | 0.229 | nan | 48% |
| 500 | free_fall | 0.020 | 0.053 | 0.306 | 0.127 | 0.326 | 0.631 | 71% |
| 500 | pendulum | 0.030 | nan | 0.168 | 0.187 | 0.710 | 0.589 | 71% |
| 500 | bouncing | 0.016 | 0.034 | 3.281 | 0.883 | 0.229 | nan | 66% |
| 625 | free_fall | 0.020 | 0.021 | 0.306 | -0.007 | 0.326 | 0.407 | 75% |
| 625 | pendulum | 0.030 | nan | 0.168 | nan | 0.710 | nan | 68% |
| 625 | bouncing | 0.016 | nan | 3.281 | 0.790 | 0.229 | nan | 48% |

### Equivarianza directa, checkpoint 250
Diferencia media de píxeles entre la generación con condicionamiento original y la rotada-des-rotada (menor = más equivariante). Calibración: 0,3 perfecto, 15 ninguna.

| brazo | media | n | bouncing | free_fall | pendulum |
|---|---|---|---|---|---|
| equiv_fix250 | 6.22 | 9 | 8.17 | 5.13 | 5.37 |
| ctrl_rep250 | 4.78 | 9 | 5.70 | 3.87 | 4.77 |

### Curvas de entrenamiento
`lambdafix_run/checkpoints_sana/stage2_equiv_lambdafix_s42_20260906-183237_5de5e17a-fixpiso/training_log.jsonl` · 257 pasos · λ_rot = 0.00154

| ventana | L_dif | L_rot/rama | cos_ramas | escala_ramas | razón grad | cos grad fís/dif | ‖q‖ | salteados | s/paso |
|---|---|---|---|---|---|---|---|---|---|
| 1-100 | 0.290 | 0.615 | — | — | 1.103 | — | 3.530 | 9/100 | 87.723 |
| 101-200 | 0.338 | 0.546 | — | — | 1.648 | — | 4.279 | 7/100 | 66.809 |
| 201-257 | 0.339 | 0.593 | — | — | 2.070 | — | 4.076 | 5/57 | 45.089 |

**Prueba de tendencia preregistrada** (primeros 100 pasos vs últimos 100, Mann-Whitney)
| cantidad | primeros | últimos | p |
|---|---|---|---|
| loss_rotation | 0.6151 | 0.5581 | 0.0896 |
| escala_ramas | 0.0000 | 0.0000 | 1.0000 |

**Validación de difusión**: 50: 0.0763 · 100: 0.0710 · 150: 0.1027 · 200: 0.0873 · 250: 0.1231

## 3. Ablación de aumentaciones (etapa 1, 3000 pasos, A con vs B sin)

### En distribución (20 clips/escenario)
`ablacion_aug_v2/results/ablacion_aug_v2/indist/eval_results.json` · brazos: base, A_con_aug, B_sin_aug, quieto · frames generados: 33

**vel_mae**
| brazo | free_fall | pendulum | bouncing |
|---|---|---|---|
| base | 16.062 | 9.024 | 10.167 |
| A_con_aug | 4.579 | 4.583 | 6.750 |
| B_sin_aug | 3.775 | 4.688 | 6.458 |
| quieto | 11.498 | 7.519 | 9.953 |

**mov_ratio**
| brazo | free_fall | pendulum | bouncing |
|---|---|---|---|
| base | 0.921 | 1.094 | 0.093 |
| A_con_aug | 0.670 | 0.656 | 0.509 |
| B_sin_aug | 0.749 | 0.550 | 0.478 |
| quieto | 0.140 | 0.067 | 0.142 |

**mae_ay**
| brazo | free_fall | pendulum | bouncing |
|---|---|---|---|
| base | 1.576 | 1.774 | 3.117 |
| A_con_aug | 0.866 | 0.840 | 3.713 |
| B_sin_aug | 0.858 | 0.744 | 3.752 |
| quieto | 0.000 | 0.826 | 3.170 |

**jerk_rms**
| brazo | free_fall | pendulum | bouncing | ood_t2v |
|---|---|---|---|---|
| base | 8.457 | 7.544 | 0.662 | 1.166 |
| A_con_aug | 1.571 | 0.930 | 2.177 | 2.418 |
| B_sin_aug | 1.361 | 0.846 | 2.200 | 1.828 |
| quieto | 0.000 | 0.000 | 0.000 | — |

**l_rot**
| brazo | free_fall | pendulum | bouncing |
|---|---|---|---|
| base | 101.324 | 57.038 | 0.443 |
| A_con_aug | 2.325 | 2.635 | 6.108 |
| B_sin_aug | 1.958 | 2.101 | 5.389 |
| quieto | 0.000 | 0.000 | 0.000 |

**l_rot_norm**
| brazo | free_fall | pendulum | bouncing |
|---|---|---|---|
| base | 0.010 | 0.000 | 0.068 |
| A_con_aug | 0.163 | 0.209 | 0.363 |
| B_sin_aug | 0.034 | 0.244 | 0.270 |
| quieto | 0.000 | 0.000 | 0.000 |

**Apareado por clip contra `A_con_aug`** (Wilcoxon, todos los escenarios juntos)
| brazo | métrica | n | ref | brazo | Δ | mejor | p |
|---|---|---|---|---|---|---|---|
| base | vel_mae | 60 | 5.304 | 11.751 | +6.447 | 3/60 | 0.000 |
| base | mov_ratio | 60 | 0.612 | 0.703 | +0.091 | 48/60 | 0.002 |
| base | mae_ay | 60 | 1.806 | 2.156 | +0.350 | 32/60 | 1.000 |
| base | jerk_rms | 60 | 1.559 | 5.554 | +3.995 | 31/60 | 0.735 |
| base | l_rot | 60 | 3.689 | 52.935 | +49.246 | 46/60 | 0.044 |
| base | l_rot_norm | 60 | 0.245 | 0.026 | -0.219 | 19/60 | 0.001 |
| B_sin_aug | vel_mae | 60 | 5.304 | 4.974 | -0.330 | 37/60 | 0.059 |
| B_sin_aug | mov_ratio | 60 | 0.612 | 0.592 | -0.019 | 32/60 | 0.757 |
| B_sin_aug | mae_ay | 60 | 1.806 | 1.785 | -0.021 | 28/60 | 0.757 |
| B_sin_aug | jerk_rms | 60 | 1.559 | 1.469 | -0.091 | 36/60 | 0.135 |
| B_sin_aug | l_rot | 60 | 3.689 | 3.150 | -0.540 | 32/60 | 0.233 |
| B_sin_aug | l_rot_norm | 60 | 0.245 | 0.182 | -0.063 | 14/60 | 0.370 |
| quieto | vel_mae | 60 | 5.304 | 9.657 | +4.353 | 6/60 | 0.000 |
| quieto | mov_ratio | 60 | 0.612 | 0.116 | -0.495 | 53/60 | 0.000 |
| quieto | mae_ay | 60 | 1.806 | 1.332 | -0.474 | 48/60 | 0.000 |
| quieto | jerk_rms | 60 | 1.559 | 0.000 | -1.559 | 60/60 | 0.000 |
| quieto | l_rot | 60 | 3.689 | 0.000 | -3.689 | 60/60 | 0.000 |
| quieto | l_rot_norm | 60 | 0.245 | 0.000 | -0.245 | 20/60 | 0.000 |

### Fuera de distribución: tiro vertical
`ablacion_aug_v2/results/ablacion_aug_v2/ood/eval_results.json` · brazos: base, A_con_aug, B_sin_aug, quieto · frames generados: 33

**vel_mae**
| brazo | vertical_throw |
|---|---|
| base | 10.058 |
| A_con_aug | 6.826 |
| B_sin_aug | 7.590 |
| quieto | 8.899 |

**mov_ratio**
| brazo | vertical_throw |
|---|---|
| base | 0.724 |
| A_con_aug | 0.540 |
| B_sin_aug | 0.496 |
| quieto | 0.216 |

**mae_ay**
| brazo | vertical_throw |
|---|---|
| base | 1.826 |
| A_con_aug | 1.587 |
| B_sin_aug | 1.427 |
| quieto | 1.125 |

**jerk_rms**
| brazo | vertical_throw | ood_t2v |
|---|---|---|
| base | 4.325 | 0.851 |
| A_con_aug | 1.950 | 2.419 |
| B_sin_aug | 1.565 | 1.641 |
| quieto | 0.000 | — |

**l_rot**
| brazo | vertical_throw |
|---|---|
| base | 13.131 |
| A_con_aug | 5.526 |
| B_sin_aug | 4.176 |
| quieto | 0.000 |

**l_rot_norm**
| brazo | vertical_throw |
|---|---|
| base | 0.054 |
| A_con_aug | 0.271 |
| B_sin_aug | 0.170 |
| quieto | 0.000 |

**Apareado por clip contra `A_con_aug`** (Wilcoxon, todos los escenarios juntos)
| brazo | métrica | n | ref | brazo | Δ | mejor | p |
|---|---|---|---|---|---|---|---|
| base | vel_mae | 15 | 6.826 | 10.058 | +3.232 | 5/15 | 0.055 |
| base | mov_ratio | 15 | 0.540 | 0.724 | +0.184 | 9/15 | 0.389 |
| base | mae_ay | 15 | 1.587 | 1.826 | +0.239 | 5/15 | 0.489 |
| base | jerk_rms | 15 | 1.950 | 4.325 | +2.375 | 6/15 | 0.169 |
| base | l_rot | 15 | 5.526 | 13.131 | +7.605 | 9/15 | 0.524 |
| base | l_rot_norm | 15 | 0.271 | 0.054 | -0.217 | 5/15 | 0.075 |
| B_sin_aug | vel_mae | 15 | 6.826 | 7.590 | +0.764 | 5/15 | 0.169 |
| B_sin_aug | mov_ratio | 15 | 0.540 | 0.496 | -0.044 | 8/15 | 0.762 |
| B_sin_aug | mae_ay | 15 | 1.587 | 1.427 | -0.160 | 9/15 | 0.121 |
| B_sin_aug | jerk_rms | 15 | 1.950 | 1.565 | -0.385 | 7/15 | 0.847 |
| B_sin_aug | l_rot | 15 | 5.526 | 4.176 | -1.350 | 8/15 | 0.890 |
| B_sin_aug | l_rot_norm | 15 | 0.271 | 0.170 | -0.101 | 4/15 | 0.499 |
| quieto | vel_mae | 15 | 6.826 | 8.899 | +2.073 | 5/15 | 0.008 |
| quieto | mov_ratio | 15 | 0.540 | 0.216 | -0.324 | 11/15 | 0.035 |
| quieto | mae_ay | 15 | 1.587 | 1.125 | -0.462 | 11/15 | 0.018 |
| quieto | jerk_rms | 15 | 1.950 | 0.000 | -1.950 | 15/15 | 0.000 |
| quieto | l_rot | 15 | 5.526 | 0.000 | -5.526 | 15/15 | 0.000 |
| quieto | l_rot_norm | 15 | 0.271 | 0.000 | -0.271 | 5/15 | 0.043 |

## 4. Corrida E4-norm2 original (λ = 6,694e-05)

### Checkpoint 250
`results/e4norm/paso_250/eval_results.json` · brazos: base, ctrl, equiv_norm, quieto · frames generados: 33

**vel_mae**
| brazo | free_fall | pendulum | bouncing |
|---|---|---|---|
| base | 15.310 | 14.979 | 9.553 |
| ctrl | 6.468 | 4.277 | 6.730 |
| equiv_norm | 5.800 | 4.053 | 6.106 |
| quieto | 11.103 | 7.019 | 10.172 |

**mov_ratio**
| brazo | free_fall | pendulum | bouncing |
|---|---|---|---|
| base | 0.848 | 1.798 | 0.151 |
| ctrl | 0.502 | 0.535 | 0.535 |
| equiv_norm | 0.577 | 0.549 | 0.504 |
| quieto | 0.202 | 0.039 | 0.070 |

**mae_ay**
| brazo | free_fall | pendulum | bouncing |
|---|---|---|---|
| base | 1.069 | 1.609 | 3.165 |
| ctrl | 0.789 | 0.793 | 3.603 |
| equiv_norm | 1.070 | 0.650 | 3.953 |
| quieto | 0.000 | 0.717 | 3.273 |

**jerk_rms**
| brazo | free_fall | pendulum | bouncing | ood_t2v |
|---|---|---|---|---|
| base | 9.198 | 14.297 | 0.913 | 1.132 |
| ctrl | 1.351 | 0.767 | 1.918 | 2.239 |
| equiv_norm | 1.627 | 0.798 | 2.646 | 2.138 |
| quieto | 0.000 | 0.000 | 0.000 | — |

**l_rot**
| brazo | free_fall | pendulum | bouncing |
|---|---|---|---|
| base | 67.715 | 146.543 | 1.701 |
| ctrl | 2.305 | 1.575 | 4.025 |
| equiv_norm | 3.683 | 1.547 | 6.646 |
| quieto | 0.000 | 0.000 | 0.000 |

**l_rot_norm**
| brazo | free_fall | pendulum | bouncing |
|---|---|---|---|
| base | -0.836 | -0.871 | -0.773 |
| ctrl | -0.167 | -0.097 | -0.319 |
| equiv_norm | -0.298 | -0.245 | -0.149 |
| quieto | 0.000 | 0.000 | 0.000 |

**Apareado por clip contra `ctrl`** (Wilcoxon, todos los escenarios juntos)
| brazo | métrica | n | ref | brazo | Δ | mejor | p |
|---|---|---|---|---|---|---|---|
| base | vel_mae | 30 | 5.825 | 13.281 | +7.456 | 3/30 | 0.000 |
| base | mov_ratio | 30 | 0.524 | 0.933 | +0.409 | 21/30 | 0.280 |
| base | mae_ay | 30 | 1.728 | 1.948 | +0.219 | 15/30 | 0.655 |
| base | jerk_rms | 30 | 1.345 | 8.136 | +6.791 | 12/30 | 0.038 |
| base | l_rot | 30 | 2.635 | 71.987 | +69.352 | 16/30 | 0.452 |
| base | l_rot_norm | 30 | -0.195 | -0.827 | -0.632 | 28/30 | 0.000 |
| equiv_norm | vel_mae | 30 | 5.825 | 5.320 | -0.505 | 21/30 | 0.025 |
| equiv_norm | mov_ratio | 30 | 0.524 | 0.543 | +0.019 | 11/30 | 0.245 |
| equiv_norm | mae_ay | 30 | 1.728 | 1.891 | +0.163 | 11/30 | 0.146 |
| equiv_norm | jerk_rms | 30 | 1.345 | 1.691 | +0.345 | 10/30 | 0.035 |
| equiv_norm | l_rot | 30 | 2.635 | 3.959 | +1.324 | 12/30 | 0.073 |
| equiv_norm | l_rot_norm | 30 | -0.195 | -0.231 | -0.036 | 14/30 | 0.871 |
| quieto | vel_mae | 30 | 5.825 | 9.431 | +3.607 | 2/30 | 0.000 |
| quieto | mov_ratio | 30 | 0.524 | 0.104 | -0.420 | 28/30 | 0.000 |
| quieto | mae_ay | 30 | 1.728 | 1.330 | -0.398 | 24/30 | 0.000 |
| quieto | jerk_rms | 30 | 1.345 | 0.000 | -1.345 | 30/30 | 0.000 |
| quieto | l_rot | 30 | 2.635 | 0.000 | -2.635 | 30/30 | 0.000 |
| quieto | l_rot_norm | 30 | -0.195 | 0.000 | +0.195 | 10/30 | 0.073 |

### Checkpoint 500
`results/e4norm/paso_500/eval_results.json` · brazos: base, ctrl, equiv_norm, quieto · frames generados: 33

**vel_mae**
| brazo | free_fall | pendulum | bouncing |
|---|---|---|---|
| base | 14.803 | 16.022 | 9.506 |
| ctrl | 5.729 | 3.968 | 6.410 |
| equiv_norm | 5.773 | 3.386 | 6.538 |
| quieto | 11.103 | 7.019 | 10.172 |

**mov_ratio**
| brazo | free_fall | pendulum | bouncing |
|---|---|---|---|
| base | 0.867 | 1.909 | 0.155 |
| ctrl | 0.563 | 0.639 | 0.541 |
| equiv_norm | 0.552 | 0.753 | 0.539 |
| quieto | 0.202 | 0.039 | 0.070 |

**mae_ay**
| brazo | free_fall | pendulum | bouncing |
|---|---|---|---|
| base | 1.123 | 1.567 | 3.160 |
| ctrl | 0.698 | 0.763 | 3.767 |
| equiv_norm | 0.864 | 0.805 | 4.010 |
| quieto | 0.000 | 0.717 | 3.273 |

**jerk_rms**
| brazo | free_fall | pendulum | bouncing | ood_t2v |
|---|---|---|---|---|
| base | 9.949 | 16.761 | 0.947 | 0.886 |
| ctrl | 1.190 | 0.802 | 2.109 | 2.331 |
| equiv_norm | 1.352 | 1.082 | 3.165 | 2.742 |
| quieto | 0.000 | 0.000 | 0.000 | — |

**l_rot**
| brazo | free_fall | pendulum | bouncing |
|---|---|---|---|
| base | 58.999 | 218.204 | 1.950 |
| ctrl | 2.117 | 1.641 | 4.463 |
| equiv_norm | 3.673 | 2.782 | 9.884 |
| quieto | 0.000 | 0.000 | 0.000 |

**l_rot_norm**
| brazo | free_fall | pendulum | bouncing |
|---|---|---|---|
| base | -0.861 | -0.896 | -0.739 |
| ctrl | -0.525 | -0.048 | -0.238 |
| equiv_norm | -0.097 | -0.097 | 0.238 |
| quieto | 0.000 | 0.000 | 0.000 |

**Apareado por clip contra `ctrl`** (Wilcoxon, todos los escenarios juntos)
| brazo | métrica | n | ref | brazo | Δ | mejor | p |
|---|---|---|---|---|---|---|---|
| base | vel_mae | 30 | 5.369 | 13.444 | +8.075 | 1/30 | 0.000 |
| base | mov_ratio | 30 | 0.581 | 0.977 | +0.396 | 21/30 | 0.271 |
| base | mae_ay | 30 | 1.743 | 1.950 | +0.207 | 17/30 | 0.792 |
| base | jerk_rms | 30 | 1.367 | 9.219 | +7.852 | 13/30 | 0.038 |
| base | l_rot | 30 | 2.740 | 93.051 | +90.311 | 15/30 | 0.129 |
| base | l_rot_norm | 30 | -0.271 | -0.832 | -0.561 | 24/30 | 0.000 |
| equiv_norm | vel_mae | 30 | 5.369 | 5.232 | -0.137 | 16/30 | 0.428 |
| equiv_norm | mov_ratio | 30 | 0.581 | 0.615 | +0.034 | 15/30 | 0.428 |
| equiv_norm | mae_ay | 30 | 1.743 | 1.893 | +0.150 | 12/30 | 0.262 |
| equiv_norm | jerk_rms | 30 | 1.367 | 1.866 | +0.499 | 8/30 | 0.003 |
| equiv_norm | l_rot | 30 | 2.740 | 5.446 | +2.706 | 8/30 | 0.001 |
| equiv_norm | l_rot_norm | 30 | -0.271 | 0.015 | +0.285 | 10/30 | 0.052 |
| quieto | vel_mae | 30 | 5.369 | 9.431 | +4.062 | 2/30 | 0.000 |
| quieto | mov_ratio | 30 | 0.581 | 0.104 | -0.477 | 28/30 | 0.000 |
| quieto | mae_ay | 30 | 1.743 | 1.330 | -0.412 | 26/30 | 0.000 |
| quieto | jerk_rms | 30 | 1.367 | 0.000 | -1.367 | 30/30 | 0.000 |
| quieto | l_rot | 30 | 2.740 | 0.000 | -2.740 | 30/30 | 0.000 |
| quieto | l_rot_norm | 30 | -0.271 | 0.000 | +0.271 | 8/30 | 0.070 |

### Checkpoint 750
`results/e4norm/paso_750/eval_results.json` · brazos: base, ctrl, equiv_norm, quieto · frames generados: 33

**vel_mae**
| brazo | free_fall | pendulum | bouncing |
|---|---|---|---|
| base | 15.459 | 15.673 | 9.496 |
| ctrl | 5.182 | 3.534 | 6.577 |
| equiv_norm | 5.610 | 3.701 | 5.756 |
| quieto | 11.103 | 7.019 | 10.172 |

**mov_ratio**
| brazo | free_fall | pendulum | bouncing |
|---|---|---|---|
| base | 0.887 | 1.870 | 0.157 |
| ctrl | 0.605 | 0.822 | 0.523 |
| equiv_norm | 0.581 | 0.659 | 0.657 |
| quieto | 0.202 | 0.039 | 0.070 |

**mae_ay**
| brazo | free_fall | pendulum | bouncing |
|---|---|---|---|
| base | 1.314 | 1.631 | 3.147 |
| ctrl | 0.799 | 0.809 | 3.891 |
| equiv_norm | 1.188 | 0.781 | 4.390 |
| quieto | 0.000 | 0.717 | 3.273 |

**jerk_rms**
| brazo | free_fall | pendulum | bouncing | ood_t2v |
|---|---|---|---|---|
| base | 10.400 | 15.238 | 0.943 | 0.962 |
| ctrl | 1.461 | 1.041 | 2.188 | 2.676 |
| equiv_norm | 1.840 | 1.000 | 3.058 | 2.681 |
| quieto | 0.000 | 0.000 | 0.000 | — |

**l_rot**
| brazo | free_fall | pendulum | bouncing |
|---|---|---|---|
| base | 74.745 | 172.752 | 1.846 |
| ctrl | 2.361 | 2.932 | 5.463 |
| equiv_norm | 3.532 | 2.393 | 10.920 |
| quieto | 0.000 | 0.000 | 0.000 |

**l_rot_norm**
| brazo | free_fall | pendulum | bouncing |
|---|---|---|---|
| base | -0.797 | -0.871 | -0.763 |
| ctrl | -0.356 | -0.111 | -0.024 |
| equiv_norm | -0.032 | -0.060 | 0.056 |
| quieto | 0.000 | 0.000 | 0.000 |

**Apareado por clip contra `ctrl`** (Wilcoxon, todos los escenarios juntos)
| brazo | métrica | n | ref | brazo | Δ | mejor | p |
|---|---|---|---|---|---|---|---|
| base | vel_mae | 30 | 5.098 | 13.542 | +8.445 | 1/30 | 0.000 |
| base | mov_ratio | 30 | 0.650 | 0.971 | +0.321 | 22/30 | 0.114 |
| base | mae_ay | 30 | 1.833 | 2.031 | +0.198 | 16/30 | 0.612 |
| base | jerk_rms | 30 | 1.563 | 8.860 | +7.297 | 17/30 | 0.140 |
| base | l_rot | 30 | 3.585 | 83.114 | +79.529 | 17/30 | 0.318 |
| base | l_rot_norm | 30 | -0.164 | -0.810 | -0.647 | 27/30 | 0.000 |
| equiv_norm | vel_mae | 30 | 5.098 | 5.022 | -0.076 | 15/30 | 0.871 |
| equiv_norm | mov_ratio | 30 | 0.650 | 0.632 | -0.018 | 17/30 | 0.465 |
| equiv_norm | mae_ay | 30 | 1.833 | 2.120 | +0.287 | 11/30 | 0.040 |
| equiv_norm | jerk_rms | 30 | 1.563 | 1.966 | +0.403 | 16/30 | 0.280 |
| equiv_norm | l_rot | 30 | 3.585 | 5.615 | +2.030 | 13/30 | 0.043 |
| equiv_norm | l_rot_norm | 30 | -0.164 | -0.012 | +0.151 | 12/30 | 0.280 |
| quieto | vel_mae | 30 | 5.098 | 9.431 | +4.334 | 2/30 | 0.000 |
| quieto | mov_ratio | 30 | 0.650 | 0.104 | -0.546 | 28/30 | 0.000 |
| quieto | mae_ay | 30 | 1.833 | 1.330 | -0.503 | 25/30 | 0.000 |
| quieto | jerk_rms | 30 | 1.563 | 0.000 | -1.563 | 30/30 | 0.000 |
| quieto | l_rot | 30 | 3.585 | 0.000 | -3.585 | 30/30 | 0.000 |
| quieto | l_rot_norm | 30 | -0.164 | 0.000 | +0.164 | 10/30 | 0.245 |

### Checkpoint 1000
`results/e4norm/paso_1000/eval_results.json` · brazos: base, ctrl, equiv_norm, quieto · frames generados: 33

**vel_mae**
| brazo | free_fall | pendulum | bouncing |
|---|---|---|---|
| base | 15.164 | 15.271 | 9.493 |
| ctrl | 5.089 | 3.671 | 6.065 |
| equiv_norm | 5.280 | 3.608 | 5.997 |
| quieto | 11.103 | 7.019 | 10.172 |

**mov_ratio**
| brazo | free_fall | pendulum | bouncing |
|---|---|---|---|
| base | 0.900 | 1.860 | 0.158 |
| ctrl | 0.607 | 0.747 | 0.623 |
| equiv_norm | 0.602 | 0.779 | 0.564 |
| quieto | 0.202 | 0.039 | 0.070 |

**mae_ay**
| brazo | free_fall | pendulum | bouncing |
|---|---|---|---|
| base | 1.086 | 1.625 | 3.159 |
| ctrl | 0.730 | 0.721 | 3.899 |
| equiv_norm | 0.992 | 0.724 | 4.059 |
| quieto | 0.000 | 0.717 | 3.273 |

**jerk_rms**
| brazo | free_fall | pendulum | bouncing | ood_t2v |
|---|---|---|---|---|
| base | 10.059 | 19.362 | 0.918 | 1.265 |
| ctrl | 1.478 | 0.830 | 2.617 | 2.689 |
| equiv_norm | 1.515 | 0.764 | 2.405 | 2.448 |
| quieto | 0.000 | 0.000 | 0.000 | — |

**l_rot**
| brazo | free_fall | pendulum | bouncing |
|---|---|---|---|
| base | 59.500 | 268.902 | 1.968 |
| ctrl | 3.260 | 2.065 | 6.535 |
| equiv_norm | 3.083 | 1.512 | 6.739 |
| quieto | 0.000 | 0.000 | 0.000 |

**l_rot_norm**
| brazo | free_fall | pendulum | bouncing |
|---|---|---|---|
| base | -0.880 | -0.886 | -0.750 |
| ctrl | -0.335 | -0.012 | -0.322 |
| equiv_norm | -0.218 | -0.271 | -0.038 |
| quieto | 0.000 | 0.000 | 0.000 |

**Apareado por clip contra `ctrl`** (Wilcoxon, todos los escenarios juntos)
| brazo | métrica | n | ref | brazo | Δ | mejor | p |
|---|---|---|---|---|---|---|---|
| base | vel_mae | 30 | 4.942 | 13.310 | +8.368 | 2/30 | 0.000 |
| base | mov_ratio | 30 | 0.659 | 0.973 | +0.314 | 22/30 | 0.119 |
| base | mae_ay | 30 | 1.784 | 1.957 | +0.173 | 17/30 | 0.655 |
| base | jerk_rms | 30 | 1.641 | 10.113 | +8.472 | 16/30 | 0.135 |
| base | l_rot | 30 | 3.953 | 110.123 | +106.170 | 17/30 | 0.404 |
| base | l_rot_norm | 30 | -0.223 | -0.838 | -0.615 | 26/30 | 0.000 |
| equiv_norm | vel_mae | 30 | 4.942 | 4.961 | +0.020 | 15/30 | 1.000 |
| equiv_norm | mov_ratio | 30 | 0.659 | 0.648 | -0.011 | 17/30 | 0.503 |
| equiv_norm | mae_ay | 30 | 1.784 | 1.925 | +0.142 | 14/30 | 0.100 |
| equiv_norm | jerk_rms | 30 | 1.641 | 1.561 | -0.080 | 17/30 | 0.477 |
| equiv_norm | l_rot | 30 | 3.953 | 3.778 | -0.175 | 16/30 | 0.984 |
| equiv_norm | l_rot_norm | 30 | -0.223 | -0.176 | +0.047 | 15/30 | 0.761 |
| quieto | vel_mae | 30 | 4.942 | 9.431 | +4.490 | 2/30 | 0.000 |
| quieto | mov_ratio | 30 | 0.659 | 0.104 | -0.555 | 28/30 | 0.000 |
| quieto | mae_ay | 30 | 1.784 | 1.330 | -0.454 | 25/30 | 0.000 |
| quieto | jerk_rms | 30 | 1.641 | 0.000 | -1.641 | 30/30 | 0.000 |
| quieto | l_rot | 30 | 3.953 | 0.000 | -3.953 | 30/30 | 0.000 |
| quieto | l_rot_norm | 30 | -0.223 | 0.000 | +0.223 | 9/30 | 0.064 |

### Tabla final, en distribución
`results/e4norm/indist/eval_results.json` · brazos: base, stage1, ctrl_s42, equiv_norm, quieto · frames generados: 33

**vel_mae**
| brazo | free_fall | pendulum | bouncing |
|---|---|---|---|
| base | 12.070 | 14.097 | 10.203 |
| stage1 | 4.845 | 4.733 | 6.342 |
| ctrl_s42 | 4.791 | 4.257 | 5.845 |
| equiv_norm | 4.831 | 4.385 | 6.162 |
| quieto | 11.504 | 7.515 | 9.953 |

**mov_ratio**
| brazo | free_fall | pendulum | bouncing |
|---|---|---|---|
| base | 0.719 | 1.831 | 0.099 |
| stage1 | 0.628 | 0.599 | 0.590 |
| ctrl_s42 | 0.629 | 0.695 | 0.586 |
| equiv_norm | 0.630 | 0.662 | 0.523 |
| quieto | 0.139 | 0.068 | 0.142 |

**mae_ay**
| brazo | free_fall | pendulum | bouncing |
|---|---|---|---|
| base | 1.752 | 2.259 | 3.106 |
| stage1 | 1.005 | 0.883 | 3.623 |
| ctrl_s42 | 0.642 | 0.812 | 3.728 |
| equiv_norm | 0.818 | 0.805 | 3.871 |
| quieto | 0.000 | 0.826 | 3.170 |

**jerk_rms**
| brazo | free_fall | pendulum | bouncing | ood_t2v |
|---|---|---|---|---|
| base | 8.110 | 12.770 | 0.681 | 0.840 |
| stage1 | 1.312 | 0.755 | 2.078 | 2.507 |
| ctrl_s42 | 1.283 | 0.913 | 2.571 | 2.812 |
| equiv_norm | 1.383 | 0.765 | 2.282 | — |
| quieto | 0.000 | 0.000 | 0.000 | — |

**l_rot**
| brazo | free_fall | pendulum | bouncing |
|---|---|---|---|
| base | 61.538 | 114.096 | 0.957 |
| stage1 | 2.707 | 1.332 | 4.912 |
| ctrl_s42 | 2.760 | 1.948 | 5.593 |
| equiv_norm | 2.576 | 1.440 | 6.137 |
| quieto | 0.000 | 0.000 | 0.000 |

**l_rot_norm**
| brazo | free_fall | pendulum | bouncing |
|---|---|---|---|
| base | -0.857 | -0.874 | -0.806 |
| stage1 | -0.180 | -0.240 | 0.022 |
| ctrl_s42 | -0.340 | -0.174 | -0.303 |
| equiv_norm | -0.338 | -0.288 | 0.172 |
| quieto | 0.000 | 0.000 | 0.000 |

### Tabla final, fuera de distribución
`results/e4norm/ood/eval_results.json` · brazos: base, ctrl_s42, equiv_norm, quieto · frames generados: 33

**vel_mae**
| brazo | vertical_throw |
|---|---|
| base | 8.524 |
| ctrl_s42 | 7.198 |
| equiv_norm | 7.891 |
| quieto | 8.892 |

**mov_ratio**
| brazo | vertical_throw |
|---|---|
| base | 0.547 |
| ctrl_s42 | 0.494 |
| equiv_norm | 0.334 |
| quieto | 0.216 |

**mae_ay**
| brazo | vertical_throw |
|---|---|
| base | 1.882 |
| ctrl_s42 | 1.454 |
| equiv_norm | 1.338 |
| quieto | 1.125 |

**jerk_rms**
| brazo | vertical_throw | ood_t2v |
|---|---|---|
| base | 5.345 | 1.738 |
| ctrl_s42 | 2.149 | 2.331 |
| equiv_norm | 1.327 | 2.719 |
| quieto | 0.000 | — |

**l_rot**
| brazo | vertical_throw |
|---|---|
| base | 13.840 |
| ctrl_s42 | 5.454 |
| equiv_norm | 1.818 |
| quieto | 0.000 |

**l_rot_norm**
| brazo | vertical_throw |
|---|---|
| base | -0.793 |
| ctrl_s42 | -0.229 |
| equiv_norm | -0.353 |
| quieto | 0.000 |

## 5. Corrida con la pérdida vieja a λ efectivo (colapsada, cortada en el 125)

### Checkpoint 125
`lambda_vieja_colapsada/results/e4lambda/paso_125/eval_results.json` · brazos: base, ctrl_l40s, equiv_lambda, quieto · frames generados: 33

**vel_mae**
| brazo | free_fall | pendulum | bouncing |
|---|---|---|---|
| base | 12.534 | 6.132 | 9.966 |
| ctrl_l40s | 4.511 | 3.934 | 6.360 |
| equiv_lambda | 11.902 | 6.464 | 8.901 |
| quieto | 11.091 | 7.018 | 10.175 |

**mov_ratio**
| brazo | free_fall | pendulum | bouncing |
|---|---|---|---|
| base | 0.528 | 0.498 | 0.093 |
| ctrl_l40s | 0.680 | 0.632 | 0.494 |
| equiv_lambda | 0.076 | 0.126 | 0.203 |
| quieto | 0.203 | 0.040 | 0.070 |

**mae_ay**
| brazo | free_fall | pendulum | bouncing |
|---|---|---|---|
| base | 1.290 | 1.384 | 3.228 |
| ctrl_l40s | 0.758 | 0.682 | 3.833 |
| equiv_lambda | 0.152 | 0.718 | 3.232 |
| quieto | 0.000 | 0.717 | 3.273 |

**jerk_rms**
| brazo | free_fall | pendulum | bouncing | ood_t2v |
|---|---|---|---|---|
| base | 4.371 | 3.375 | 0.597 | 1.186 |
| ctrl_l40s | 1.324 | 0.808 | 2.504 | 3.252 |
| equiv_lambda | 0.366 | 0.223 | 0.607 | — |
| quieto | 0.000 | 0.000 | 0.000 | — |

**l_rot**
| brazo | free_fall | pendulum | bouncing |
|---|---|---|---|
| base | 18.376 | 9.989 | 0.226 |
| ctrl_l40s | 2.529 | 2.353 | 5.532 |
| equiv_lambda | 0.094 | 0.118 | 0.737 |
| quieto | 0.000 | 0.000 | 0.000 |

**l_rot_norm**
| brazo | free_fall | pendulum | bouncing |
|---|---|---|---|
| base | 0.000 | 0.000 | 0.000 |
| ctrl_l40s | 0.009 | 0.053 | 0.206 |
| equiv_lambda | 0.000 | 0.110 | 0.135 |
| quieto | 0.000 | 0.000 | 0.000 |

**Apareado por clip contra `ctrl_l40s`** (Wilcoxon, todos los escenarios juntos)
| brazo | métrica | n | ref | brazo | Δ | mejor | p |
|---|---|---|---|---|---|---|---|
| base | vel_mae | 30 | 4.935 | 9.544 | +4.609 | 2/30 | 0.000 |
| base | mov_ratio | 30 | 0.602 | 0.373 | -0.229 | 26/30 | 0.000 |
| base | mae_ay | 30 | 1.758 | 1.968 | +0.210 | 20/30 | 0.289 |
| base | jerk_rms | 30 | 1.545 | 2.781 | +1.236 | 20/30 | 0.598 |
| base | l_rot | 30 | 3.471 | 9.530 | +6.059 | 23/30 | 0.055 |
| base | l_rot_norm | 30 | 0.089 | 0.000 | -0.089 | 8/30 | 0.012 |
| equiv_lambda | vel_mae | 30 | 4.935 | 9.089 | +4.154 | 0/30 | 0.000 |
| equiv_lambda | mov_ratio | 30 | 0.602 | 0.135 | -0.467 | 30/30 | 0.000 |
| equiv_lambda | mae_ay | 30 | 1.758 | 1.367 | -0.390 | 24/30 | 0.000 |
| equiv_lambda | jerk_rms | 30 | 1.545 | 0.399 | -1.147 | 30/30 | 0.000 |
| equiv_lambda | l_rot | 30 | 3.471 | 0.316 | -3.155 | 30/30 | 0.000 |
| equiv_lambda | l_rot_norm | 30 | 0.089 | 0.082 | -0.008 | 8/30 | 0.594 |
| quieto | vel_mae | 30 | 4.935 | 9.428 | +4.493 | 2/30 | 0.000 |
| quieto | mov_ratio | 30 | 0.602 | 0.104 | -0.498 | 28/30 | 0.000 |
| quieto | mae_ay | 30 | 1.758 | 1.330 | -0.427 | 24/30 | 0.000 |
| quieto | jerk_rms | 30 | 1.545 | 0.000 | -1.545 | 30/30 | 0.000 |
| quieto | l_rot | 30 | 3.471 | 0.000 | -3.471 | 30/30 | 0.000 |
| quieto | l_rot_norm | 30 | 0.089 | 0.000 | -0.089 | 8/30 | 0.012 |

