# Ficha de propuestas de proyecto · Hito H1

**Entrega: domingo 20 de septiembre, 23:59, en la Tarea «H1 · Propuestas de proyecto».**
**Decisión: domingo 27 de septiembre (hito H2).** Sin visto bueno no se sigue adelante.

Hasta **tres propuestas, en orden de preferencia**. La primera tiene que estar completa; la segunda y
la tercera son vuestras alternativas si la primera no pasa la puerta. Las tres tienen las mismas cuatro
comprobaciones.

---

## Cómo se entrega

1. Copiad esta página entera a un fichero `docs/propuestas.md` de vuestro repositorio y rellenadla ahí.
   Es Markdown: las tablas se editan en cualquier editor, y así queda en la historia de _commits_.
2. Marcad las casillas cambiando `[ ]` por `[x]` **solo cuando sea verdad**. Una casilla marcada sin
   evidencia en la tabla de debajo cuenta como no marcada.
3. Haced _commit_ y _push_, y en la Tarea de ALUD pegad **la URL del fichero en GitHub**. Entrega uno,
   cuenta por los dos.
4. El 27 de septiembre tendréis la decisión de cada propuesta en la retroalimentación de la Tarea. La
   aprobada se convierte en vuestro `docs/viabilidad.md`, que es el que se corrige con E1.

---

## Las dos vías

**Vía A · lista curada.** Uno de los doce papers ya comprobados, que están en el capítulo _Lista de
papers_ de este Libro: ResNet, U-Net, YOLO (v1), Vision Transformer, CycleGAN, Grad-CAM, SimCLR,
ejemplos adversarios (FGSM), Word2Vec, Transformer, BERT (solo preentrenamiento) y Neural
Collaborative Filtering. Sabemos que caben en la regla 5/20 y que tienen un aplicativo natural encima.
La ficha se rellena igual, pero es rápido. **Máximo dos parejas por paper**, por orden de entrega.

**Vía B · propuesta propia.** Cualquier paper o caso industrial que os interese, si la ficha pasa las
cuatro comprobaciones. Da más trabajo al principio, y ese trabajo de acotar es justo lo que venís a
aprender aquí.

---

## Pareja

|                                     |                                                                       |
| ----------------------------------- | --------------------------------------------------------------------- |
| **Nombres**                         |  Josu Viteri y Gotzon Viteri                                                                      |
| **Repositorio**                     | https://github.com/josuviteri/CLEAR                           |
| **Rol profesional al que apuntáis** | _Machine Learning Engineering, Data Science y MLOps en Research o Producción_ |

---

## Propuesta 1

### Identificación

|                            |                                                                     |
| -------------------------- | ------------------------------------------------------------------- |
| **Paper o referencia**     | _Denoising Diffusion Probabilistic Models (baseline), Classifier-Free Guidance (CFG)_                                                   |
| **Autores y año**          |  Jonathan Ho, Ajay Jain y Pieter Abbeel   (DDPM), Jonathan Ho & Tim Salimans (CFG)                                                              |
| **Enlace**                 | https://arxiv.org/abs/2006.11239 (DDPMs), https://arxiv.org/pdf/2207.12598 (CFG)                              |
| **Vía**                    | B                             |
| **Por qué este y no otro** | _Nos interesa aprender sobre modelos de visión-lenguaje (VLMs) e IA generativa aplicada a la visión por computador, especialmente sobre modelos de difusión, como los Denoising Diffusion Probabilistic Models (DDPM), y su evolución hacia la generación guiada. También nos interesa estudiar cómo la incorporación de embeddings textuales permite condicionar la generación de imágenes. Queremos darle un enfoque investigador y hacer un trabajo interesante._ |

### Comprobación 1 · Datos

- [x] Son **públicos y descargables hoy**. Enlace que funciona, no una promesa.
- [x] La licencia permite el uso académico.

|                                       |                                                                |
| ------------------------------------- | -------------------------------------------------------------- |
| **Dataset**                           |        CIFAR-10 y MNIST                                                        |
| **Enlace de descarga**                |      https://cave.cs.toronto.edu/kriz/cifar.html + https://www.kaggle.com/datasets/hojjatk/mnist-dataset                                                          |
| **Tamaño**                            | 60k imágenes - 752.88 MB (Cifar-10) y/o  60k imágenes 54.95MB (MNIST)                             |
| **Licencia**                          |                                              CC BY 4.0 (Cifar-10), CC BY-SA 3.0 (MNIST)                  |
| **¿Hace falta registro o solicitud?** | _Si, Kaggle_ |

### Comprobación 2 · Especificación

- [x] Hay implementación de referencia, **o bien** el paper especifica la arquitectura lo bastante como
      para implementarla sin adivinar.

|                                                          |                                                                                                               |
| -------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------- |
| **¿Hay código de referencia?**                           | _Sí (https://github.com/hojonathanho/diffusion) DDPM_                                                                                            |
| **Si no lo hay, ¿el paper da la arquitectura completa?** | _NA_                                                         |
| **Qué NO especifica el paper (DDPM)**                           | 
- Manejo exacto de los bordes/índices temporales
- Tratamiento del EMA (Exponential Moving Average)
- Padding y resolución de autoatención
- Discretización de log-verosimilitud vs. Muestreo continuo

### Comprobación 3 · Cómputo · la regla 5/20

- [en principio si, sería interesante contar con una segunda opinión] Cabe en el presupuesto, **con el plan de recorte escrito**.

|                                                |                                                                                                                   |
| ---------------------------------------------- | ----------------------------------------------------------------------------------------------------------------- |
| **Escala del paper original (DDPM)**                  | _CIFAR-10 completo (50.000 imágenes), 800.000 pasos de optimizador (batch size 128, ~2.048 épocas) en TPU v3-8 durante varios días; U-Net con ~35.7M parámetros._                                                          |
| **Escala que vais a hacer vosotros**           | _Tenemos pensado usar ambos datasets MNIST y CIFAR-10, con el objetivo de comprobar que todo va bien con un dataset más simple (MNIST) antes de ir con CIFAR-10, tanto para el modelo de difusión sin guiado, como para el de extensión con CFG. Por lo que tenemos varios milestones en los que tendríamos un modelo listo para entregar por si nos estancamos en algún punto del desarrollo o no da más tiempo y es necesario desarrollar la GUI o el documento tex. 1. Fase de validación (MNIST): 60.000 imágenes $1\times 28\times 28$ (padded a $32\times 32$), U-Net reducida (~1.5M parámetros: canales base 32 en vez de 128, sin self-attention pesada), ~20 epochs. 2. Fase de extensión (CIFAR-10): 50.000 imágenes $3\times 32\times 32$, U-Net compacta (~4M a 7M parámetros: canales base 64, atención solo a resolución $16\times 16$), reduciendo a 60.000 steps (~150 épocas con batch 128) usando el encoder CLIP con weights congelados (clip-vit-base-patch32) para la extensión de guiar la generación con texto sin entrenar pesos del lenguaje._                                                                       |
| **Tiempo estimado del entrenamiento completo** | _Esperamos completar el entrenamiento de demostración en colab para MNIST en 10 minutos (~10 epochs) y el extendido de forma local en 10 minutos (~30 epochs). Respecto a CIFAR-10, el sanity check en colab en 30/45 minutos (~65 epochs), y el extendido de forma local en 3.5 a 5 hora (~380 epochs). Para ejecutar en inferencia no esperamos tener problemas (varios steps por segundo), ya que es viable tanto en colab mediante scripting en forma de demo al terminar el sanity check, como usando nuestro hardware como servidor para la GUI._                                            |
| **Qué se pierde al recortar**                  | _Se espera perder calidad y que los fondos no sean muy detallados._ |
| **Hardware que vais a usar**                   | _Usaremos Google Colab (GPU T4) para la demo en forma de sanity check, con el objetivo de demostrar que los scripts de entrenamiento funcionan y GPUs locales para desarrollo, con el objetivo de extender el tiempo de entrenamiento, sin llegar a replicar el tiempo de cómputo original de los papers (NVIDIA RTX 3060 y 5070 12GB)._                                                                                |

### Comprobación 4 · Aplicativo

- [ ] Hay una capa de servicio natural encima de la replicación.

|                                |                                                                                   |
| ------------------------------ | --------------------------------------------------------------------------------- |
| **Qué construís encima**       | _Hemos elegido los papers DDPM y CFG con el objetivo de empezar replicando los modelos de difusión sin generación guiada, y ampliaremos a la generación guiada si todo va bien (primero DDPM, seguido de conditional routing guiado por texto con CFG. Pensamos desplegar los modelos en una aplicación web en la que usarlos en inferencia, con o sin generación guiada por texto (Classifier-Free Guidance - CFG)._                           |
| **Quién lo usaría y para qué** | _Investigadores o estudiantes que quieran experimentar visualmente con modelos de difusión, el parámetro de guidance (CFG) durante el denoising._ |
| **Qué necesita del modelo**    | _Dependiendo de si el denoising se lleva a cabo con o sin guía (dependiendo de nuestro progreso), hacemos uso de una seed aleatoria o es necesario un prompt que encodear con CLIP. Input: Texto del prompt (class labels del dataset correspondiente), seed aleatoria, escala CFG ($w \in [1.0, 10.0]$) y número de pasos de muestreo ($K \in [20, 50]$). Output: Imagen sintética $32\times 32$ (escalada para visualización a $128\times 128$._                                             |

### Riesgo principal

|                                                    |                                                                                              |
| -------------------------------------------------- | -------------------------------------------------------------------------------------------- |
| **Qué es lo que más probablemente va a salir mal** |  _1. Riesgo de infraestructura: Que Colab desconecte la máquina por tiempo límite o inactividad tras 3–4 horas de entrenamiento en CIFAR-10. 2. Riesgo algorítmico: Que la U-Net aprenda a generar imágenes realistas de CIFAR-10 ignorando por completo el embedding de texto de CLIP, o que aparezca guidance blowout (imágenes sobresaturadas y con ruido de alta frecuencia al subir el peso de CFG)._                                                                                            |
| **Qué haríais si pasa**                            | _1. Frente a desconexiones: El bucle guardará checkpoints cada X steps en Google Drive (model_state, optimizer_state, step_idx), aunque tenemos pensado usar nuestro hardware en la medida de lo posible, excepto para las demostraciones necesarias. Si la sesión muere, reanudamos en una máquina nueva sin perder cómputo previo. 2. Frente al ignorado del texto (falta de alineación semántica): Si en 20.000 steps el mapa de atención cruzada sigue plano o la imagen no responde al prompt, pasaremos a usar MNIST con dígitos como texto ("seven", etc.). Al tener mucha menos entropía visual que CIFAR-10, esperamos que converja antes._ |

### Visto bueno del profesor (H2) · no rellenar

|                 |                                            |
| --------------- | ------------------------------------------ |
| **Decisión**    | Aprobado |
| **Condiciones** |                                            | 

---

## Propuesta 2 (alternativa)

### Identificación

|                            |                                                                     |
| -------------------------- | ------------------------------------------------------------------- |
| **Paper o referencia**     | _Modeling Polypharmacy Side Effects with Graph Convolutional Networks (Decagon)_ |
| **Autores y año**          |  Marinka Zitnik, Monica Agrawal y Jure Leskovec, 2018 (Bioinformatics 34(13), ISMB 2018) |
| **Enlace**                 | https://arxiv.org/abs/1802.00543                              |
| **Vía**                    | B (propuesta propia)                                                |
| **Por qué este y no otro** | _Predicción de enlaces en un grafo con 964 tipos, evaluación por clase muy desbalanceada y un modelo cuya inferencia se puede levantar con pocos recursos. Encaja con el rol de ML Engineering y Data Science. Trata un área diferente respecto a la propuesta 1 (GNN sobre datos biomédicos en lugar de generación de imagen), con un coste de cómputo menor._ |

### Comprobación 1 · Datos

- [x] Son **públicos y descargables hoy**. Enlace que funciona, no una promesa.
- [x] La licencia permite el uso académico.

|                                       |                                                                |
| ------------------------------------- | -------------------------------------------------------------- |
| **Dataset**                           |  Decagon: `bio-decagon-ppi` (proteína-proteína), `bio-decagon-targets` y `-targets-all` (fármaco-proteína),  bio-decagon-combo` (efectos de polifarmacia, tripletas fármaco A – efecto – fármaco B), `bio-decagon-mono` (efectos de fármacos individuales) y `bio-decagon-effectcategories` |
| **Enlace de descarga**                | https://www.kaggle.com/datasets/josuviteri/molecular-network-and-polypharmacy-data |
| **Tamaño**                            | _645 fármacos, 19.085 proteínas, 715.612 aristas proteína-proteína, 18.596 fármaco-proteína y 4.651.131 aristas fármaco-fármaco, con 964 tipos de efecto. `45 MB` de almacenamiento.  |
| **Licencia**                          | MIT  |
| **¿Hace falta registro o solicitud?** | _Si, Kaggle_ |

### Comprobación 2 · Especificación

- [x] Hay implementación de referencia, **o bien** el paper especifica la arquitectura lo bastante como
      para implementarla sin adivinar.

|                                                          |                                                                                                               |
| -------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------- |
| **¿Hay código de referencia?**                           | _https://github.com/marinkaz/decagon_  |
| **Si no lo hay, ¿el paper da la arquitectura completa?** | _El paper menciona un encoder GCN relacional, 2 capas con 64 y 32 unidades, dropout 0.1, decoder DEDICOM para pares fármaco-fármaco y bilineal para el resto, pérdida de crossed-entropy con muestreo negativo, Adam con learning rate 0.001, batch de 512, máximo 100 epochs y early stopping de 2._ |
| **Qué NO especifica el paper**                           | _Features de entrada: cómo se codifican las proteínas ¿one-hot encoder? y su normalización. Activación de la última capa, dónde va el dropout y si hay weight decay. Las 4.651.131 aristas fármaco-fármaco coinciden con el total de 1.318 tipos, pero el modelo solo usa 964, hay que comprobarlo._ |

### Comprobación 3 · Cómputo · la regla 5/20

- [ ] Cabe en el presupuesto, **con el plan de recorte escrito**.

|                                                |                                                                                                                   |
| ---------------------------------------------- | ----------------------------------------------------------------------------------------------------------------- |
| **Escala del paper original**                  | _Grafo completo (645 fármacos, 19.085 proteínas), 964 tipos de efecto, hasta 100 epochs con early stopping, batch 512, 2 capas (64 y 32). No se mencionan hardware ni tiempo, el paper solo dice que el modelo cabe en memoria de GPU y que la implementación es lineal en el número de aristas._  |
| **Escala que vais a hacer vosotros**           | _El grafo es pequeño, así que lo que se recorta es el número de relaciones, que es lo que escala los parámetros. Primero todo el grafo de proteínas y fármacos, con un subconjunto de 50 tipos de efecto frecuentes, medios y raros, y el mismo split 80-10-10 por efecto. Después subir a 200, y si caben a los 964. Se puede probar el baseline sin encoder (DEDICOM/RESCAL) para comprobar la tesis del paper, que aporta el encoder. Además de replicar el paper original, buscamos ampliarlo añadiendo los avances que se han hecho en graph ML desde la publicación del paper en 2018, actualizando en encoder, el decoder y probando otras vías como transformaciones del grafo y de integraciones moleculares multimodales usando modelos como ChemBERTa._ |
| **Tiempo estimado del entrenamiento completo** | _La fase inicial de 50 tipos de efectos podría llevar 10 minutos, si lo aumentamos, podría llegar a unas pocas horas._ |
| **Qué se pierde al recortar**                  | Las métricas medias sobre 50 o 200 relaciones no son comparables con la Tabla 2 del paper (AUROC 0.872, AUPRC 0.832, AP@50 0.803), que tiene 964 de media. Con menos relaciones se reduce el reparto de parámetros, según el paper ayuda a los efectos raros. Esperemos quedar algo por debajo de las cifras del paper, pero no podemos garantizarlo._ |
| **Hardware que vais a usar**                   | _Usaremos Google Colab (GPU T4) para la demo en forma de sanity check, con el objetivo de demostrar que los scripts de entrenamiento funcionan y GPUs locales para desarrollo, con el objetivo de extender el tiempo de entrenamiento, sin llegar a replicar el tiempo de cómputo original de los papers (NVIDIA RTX 3060 y 5070 12GB)_ |

### Comprobación 4 · Aplicativo

- [x] Hay una capa de servicio natural encima de la replicación.

|                                |                                                                                   |
| ------------------------------ | --------------------------------------------------------------------------------- |
| **Qué construís encima**       | _Un test farmacéutico (API + interfaz web). Dada una lista de fármacos, puntuar todos los pares y devuelve los efectos secundarios más probables con su probabilidad. Sino, elegir un efecto y ver los pares más probables. Para explicar la predicción, muestra las dianas proteicas compartidas de cada par. Puede ser interesante un mapa de los efectos._ |
| **Quién lo usaría y para qué** | _Estudiantes e investigadores de farmacéutica computacional que quieran priorizar qué combinaciones merecen un estudio._ |
| **Qué necesita del modelo**    | _Entrada: 2 o más fármacos del vocabulario de 645. Salida: top-k efectos por par con probabilidad. Latencia: los embeddings de los 645 fármacos se calculan una vez y guardan. Puntuar un par contra los 964 efectos es un producto matricial pequeño, así que podría llevar menos de un segundo._ |

### Riesgo principal

|                                                    |                                                                                              |
| -------------------------------------------------- | -------------------------------------------------------------------------------------------- |
| **Qué es lo que más probablemente va a salir mal** | _1. Coste: una matriz por cada una de las 964 relaciones puede disparar memoria y tiempo, sobre todo en la primera capa si las features de entrada son de alta dimensión._<br>_2. Métricas no comparables: por los huecos del paper (muestreo de negativos, AP@50), podemos obtener cifras distintas a las suyas sin que haya un problema de implementación. Además, hay riesgo de fuga si las features de fármacos individuales incluyen los efectos que se predicen._ |
| **Qué haríais si pasa**                            | _1. Principalmente escalar por fases de 50, 200 y 964 relaciones, y parar donde quepa. Si aun así no cabe, compartir parámetros entre relaciones (descomposición en bases, estilo R-GCN) y documentarlo como desviación._<br>_2. Mismo número de negativos que positivos, mismo muestreo en validación y test, quitar de las features individuales los efectos a predecir, y comparar sobre todo contra nuestras propias baselines DEDICOM/RESCAL._ |

### Visto bueno del profesor (H2) · no rellenar

|                 |                                            |
| --------------- | ------------------------------------------ |
| **Decisión**    | Alternativa Elegiada |
| **Condiciones** |                                            |

---