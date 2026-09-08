# Introducción al aprendizaje automático

## [Cuestionario antes de la lección](https://ff-quizzes.netlify.app/en/ml/)

---

[![ML for beginners - Introducción al Aprendizaje Automático para Principiantes](https://img.youtube.com/vi/6mSx_KJxcHI/0.jpg)](https://youtu.be/6mSx_KJxcHI "ML for beginners - Introduction to Machine Learning for Beginners")

> 🎥 Haz clic en la imagen de arriba para un video corto que explica esta lección.

¡Bienvenido a este curso de aprendizaje automático clásico para principiantes! Ya seas completamente nuevo en este tema o un practicante experimentado de ML que desea repasar un área, ¡nos alegra que te unas a nosotros! Queremos crear un punto de lanzamiento amigable para tu estudio de ML y estaríamos encantados de evaluar, responder e incorporar tus [comentarios](https://github.com/microsoft/ML-For-Beginners/discussions).

[![Introducción al ML](https://img.youtube.com/vi/h0e2HAPTGF4/0.jpg)](https://youtu.be/h0e2HAPTGF4 "Introducción al ML")

> 🎥 Haz clic en la imagen de arriba para un video: John Guttag del MIT introduce el aprendizaje automático

---
## Comenzando con el aprendizaje automático

Antes de empezar con este currículo, necesitas tener tu computadora configurada y lista para ejecutar cuadernos localmente.

- **Configura tu máquina con estos videos**. Usa los siguientes enlaces para aprender [cómo instalar Python](https://youtu.be/CXZYvNRIAKM) en tu sistema y [configurar un editor de texto](https://youtu.be/EU8eayHWoZg) para el desarrollo.
- **Aprende Python**. También se recomienda tener un conocimiento básico de [Python](https://docs.microsoft.com/learn/paths/python-language/?WT.mc_id=academic-77952-leestott), un lenguaje de programación útil para científicos de datos que usaremos en este curso.
- **Aprende Node.js y JavaScript**. También usamos JavaScript en algunas ocasiones en este curso para construir aplicaciones web, por lo que necesitarás tener instalado [node](https://nodejs.org) y [npm](https://www.npmjs.com/), así como tener disponible [Visual Studio Code](https://code.visualstudio.com/) para el desarrollo en Python y JavaScript.
- **Crea una cuenta de GitHub**. Ya que nos encontraste aquí en [GitHub](https://github.com), probablemente ya tengas una cuenta, pero si no, crea una y luego haz un fork de este currículo para usarlo por tu cuenta. (También si quieres, danos una estrella 😊)
- **Explora Scikit-learn**. Familiarízate con [Scikit-learn](https://scikit-learn.org/stable/user_guide.html), un conjunto de bibliotecas ML que referenciamos en estas lecciones.

---
## ¿Qué es el aprendizaje automático?

El término 'aprendizaje automático' es uno de los términos más populares y frecuentemente usados hoy en día. Existe una posibilidad no trivial de que hayas escuchado este término al menos una vez si tienes cierta familiaridad con la tecnología, sin importar en qué dominio trabajes. Sin embargo, la mecánica del aprendizaje automático es un misterio para la mayoría de las personas. Para un principiante en aprendizaje automático, el tema puede a veces sentirse abrumador. Por lo tanto, es importante entender qué es realmente el aprendizaje automático y aprenderlo paso a paso, a través de ejemplos prácticos.

---
## La curva de hype

![curva de hype de ml](../../../../translated_images/es/hype.07183d711a17aafe.webp)

> Google Trends muestra la reciente 'curva de hype' del término 'aprendizaje automático'

---
## Un universo misterioso

Vivimos en un universo lleno de misterios fascinantes. Grandes científicos como Stephen Hawking, Albert Einstein y muchos más han dedicado sus vidas a buscar información significativa que revele los misterios del mundo que nos rodea. Esta es la condición humana de aprender: un niño humano aprende cosas nuevas y descubre la estructura de su mundo año tras año mientras crece hasta la adultez.

---
## El cerebro del niño

El cerebro y los sentidos de un niño perciben los hechos de su entorno y gradualmente aprenden los patrones ocultos de la vida que ayudan al niño a construir reglas lógicas para identificar los patrones aprendidos. El proceso de aprendizaje del cerebro humano hace a los humanos la criatura más sofisticada de este mundo. Aprender continuamente descubriendo patrones ocultos y luego innovar sobre esos patrones nos permite mejorar cada vez más a lo largo de nuestra vida. Esta capacidad de aprendizaje y la capacidad evolutiva están relacionadas con un concepto llamado [plasticidad cerebral](https://www.simplypsychology.org/brain-plasticity.html). Superficialmente, podemos trazar algunas similitudes motivacionales entre el proceso de aprendizaje del cerebro humano y los conceptos del aprendizaje automático.

---
## El cerebro humano

El [cerebro humano](https://www.livescience.com/29365-human-brain.html) percibe cosas del mundo real, procesa la información percibida, toma decisiones racionales y realiza ciertas acciones basadas en las circunstancias. Esto es lo que llamamos comportamiento inteligente. Cuando programamos una réplica del proceso de comportamiento inteligente a una máquina, se llama inteligencia artificial (IA).

---
## Algunos términos

Aunque los términos pueden confundirse, el aprendizaje automático (ML) es un subconjunto importante de la inteligencia artificial. **ML se ocupa de utilizar algoritmos especializados para descubrir información significativa y encontrar patrones ocultos a partir de datos percibidos para corroborar el proceso de toma de decisiones racional.**

---
## IA, ML, Aprendizaje Profundo

![IA, ML, aprendizaje profundo, ciencia de datos](../../../../translated_images/es/ai-ml-ds.537ea441b124ebf6.webp)

> Un diagrama que muestra las relaciones entre IA, ML, aprendizaje profundo y ciencia de datos. Infografía por [Jen Looper](https://twitter.com/jenlooper) inspirada por [este gráfico](https://softwareengineering.stackexchange.com/questions/366996/distinction-between-ai-ml-neural-networks-deep-learning-and-data-mining)

---
## Conceptos a cubrir

En este currículo, vamos a cubrir solo los conceptos centrales del aprendizaje automático que un principiante debe conocer. Cubrimos lo que llamamos 'aprendizaje automático clásico' principalmente usando Scikit-learn, una excelente biblioteca que muchos estudiantes usan para aprender lo básico. Para entender conceptos más amplios de inteligencia artificial o aprendizaje profundo, un conocimiento fundamental sólido de aprendizaje automático es indispensable, por lo que nos gustaría ofrecerlo aquí.

---
## En este curso aprenderás:

- conceptos centrales del aprendizaje automático
- la historia del ML
- ML y equidad
- técnicas de ML de regresión
- técnicas de ML de clasificación
- técnicas de ML de agrupamiento
- técnicas de ML para procesamiento de lenguaje natural
- técnicas de ML para pronóstico de series temporales
- aprendizaje por refuerzo
- aplicaciones reales para ML

---
## Lo que no cubriremos

- aprendizaje profundo
- redes neuronales
- IA

Para una mejor experiencia de aprendizaje, evitaremos las complejidades de las redes neuronales, el 'aprendizaje profundo' —construcción de modelos con muchas capas usando redes neuronales— y la IA, que discutiremos en otro currículo diferente. También ofreceremos próximamente un currículo de ciencia de datos para enfocarnos en ese aspecto de este campo más amplio.

---
## ¿Por qué estudiar aprendizaje automático?

El aprendizaje automático, desde una perspectiva de sistemas, se define como la creación de sistemas automatizados que pueden aprender patrones ocultos a partir de datos para ayudar en la toma de decisiones inteligentes.

Esta motivación está vagamente inspirada en cómo el cerebro humano aprende ciertas cosas basadas en los datos que percibe del mundo exterior.

✅ Piensa por un minuto por qué un negocio querría intentar usar estrategias de aprendizaje automático frente a crear un motor basado en reglas codificadas rígidamente.

---
## Por qué la calidad de los datos importa

Los datos de alta calidad mejoran el rendimiento del modelo. Datos pobres o ruidosos pueden llevar a predicciones inexactas, incluso usando algoritmos avanzados de aprendizaje automático.

---
## Aplicaciones del aprendizaje automático

Las aplicaciones del aprendizaje automático están casi en todas partes y son tan ubicuas como los datos que fluyen en nuestras sociedades, generados por nuestros teléfonos inteligentes, dispositivos conectados y otros sistemas. Considerando el inmenso potencial de los algoritmos de aprendizaje automático más avanzados, los investigadores han estado explorando su capacidad para resolver problemas reales multidimensionales y multidisciplinarios con grandes resultados positivos.

---
## Ejemplos de ML aplicado

**Puedes usar el aprendizaje automático de muchas maneras**:

- Para predecir la probabilidad de una enfermedad basándose en el historial médico o informes de un paciente.
- Para aprovechar datos meteorológicos y predecir eventos climáticos.
- Para entender el sentimiento de un texto.
- Para detectar noticias falsas y detener la difusión de propaganda.

Las finanzas, economía, ciencias de la Tierra, exploración espacial, ingeniería biomédica, ciencias cognitivas e incluso campos en las humanidades han adaptado el aprendizaje automático para resolver los arduos problemas de procesamiento de datos de su dominio.

---
## Conclusión

El aprendizaje automático automatiza el proceso de descubrimiento de patrones encontrando perspectivas significativas a partir de datos reales o generados. Ha demostrado ser altamente valioso en negocios, salud y aplicaciones financieras, entre otras.

En un futuro cercano, entender los fundamentos del aprendizaje automático será imprescindible para personas de cualquier dominio debido a su adopción generalizada.

---
# 🚀 Desafío

Haz un boceto, en papel o usando una aplicación en línea como [Excalidraw](https://excalidraw.com/), de tu comprensión de las diferencias entre IA, ML, aprendizaje profundo y ciencia de datos. Añade algunas ideas de problemas para los que cada una de estas técnicas es buena para resolver.

# [Cuestionario después de la lección](https://ff-quizzes.netlify.app/en/ml/)

---
# Revisión y Autoestudio

Para aprender más sobre cómo puedes trabajar con algoritmos de ML en la nube, sigue esta [Ruta de Aprendizaje](https://docs.microsoft.com/learn/paths/create-no-code-predictive-models-azure-machine-learning/?WT.mc_id=academic-77952-leestott).

Realiza una [Ruta de Aprendizaje](https://docs.microsoft.com/learn/modules/introduction-to-machine-learning/?WT.mc_id=academic-77952-leestott) sobre los conceptos básicos de ML.

---
# Tarea

[Ponte en marcha](assignment.md)

---

<!-- CO-OP TRANSLATOR DISCLAIMER START -->
**Descargo de responsabilidad**:
Este documento ha sido traducido utilizando el servicio de traducción automática [Co-op Translator](https://github.com/Azure/co-op-translator). Aunque nos esforzamos por la precisión, tenga en cuenta que las traducciones automatizadas pueden contener errores o inexactitudes. El documento original en su idioma nativo debe considerarse la fuente autorizada. Para información crítica, se recomienda una traducción profesional humana. No somos responsables de cualquier malentendido o interpretación errónea que surja del uso de esta traducción.
<!-- CO-OP TRANSLATOR DISCLAIMER END -->