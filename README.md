# BPE Tokenizer from Scratch

Este subproyecto forma parte de la infraestructura modular de Inteligencia Artificial ai-core-infra. Implementa un tokenizador Byte Pair Encoding (BPE) a nivel de bytes desarrollado íntegramente desde cero en Python, sin dependencias externas.

El tokenizador es el componente fundacional en el ciclo de procesamiento de lenguaje natural (PLN), encargado de traducir texto plano en una secuencia de identificadores numéricos (tokens) que los modelos autoregresivos (como Transformers) pueden procesar y proyectar en su espacio de embeddings.

---

## Arquitectura del Tokenizador

A diferencia de los enfoques basados en caracteres Unicode simples o tokens de palabras completas, este tokenizador implementa una aproximación de nivel de bytes (Byte-Level BPE) similar a la utilizada en arquitecturas modernas como GPT-2, GPT-4 y Llama:

1. **Vocabulario Base Inmutable (256):** El vocabulario se inicializa con los 256 bytes posibles (0 a 255). Esto garantiza que cualquier texto plano (incluyendo caracteres especiales, acentos, emojis o caracteres unicode corruptos) pueda representarse sin generar tokens fuera de vocabulario (<unk>).
2. **Pre-tokenización Estricta:** Antes de buscar fusiones de bytes, se aplica un filtro de expresiones regulares (re con soporte Unicode) que segmenta el texto en palabras individuales, números y signos de puntuación. Esto evita que BPE fusione caracteres a través de límites semánticos (por ejemplo, unir el final de una palabra con un espacio o un signo de puntuación), lo cual generaría un vocabulario caótico y redundante.
3. **Fusión Iterativa (Merges):** Durante la fase de entrenamiento, el algoritmo calcula las frecuencias de todos los pares de bytes/tokens consecutivos en el corpus estructurado. Se selecciona iterativamente el par más frecuente y se fusiona en un nuevo ID único de token. Este proceso continúa hasta alcanzar el tamaño objetivo (vocab_size).
4. **Codificación Codiciosa (Greedy Encoding):** Para codificar un texto nuevo, se dividen los fragmentos mediante el patrón de pre-tokenización y se ejecutan las fusiones aprendidas respetando estrictamente el orden cronológico en el que fueron descubiertas durante el entrenamiento.
5. **Decodificación Segura:** La decodificación simplemente concatena la representación en bytes de cada ID de token y realiza una decodificación UTF-8 robusta utilizando un método de reemplazo de errores para evitar fallos si el flujo de bytes es incompleto.

---

## Tecnologías Utilizadas

- **Python 3.10+** (Implementación con tipado estricto typing y sin librerías externas).
- **Biblioteca Estándar:** re (para pre-tokenización Unicode), json (para serialización del modelo), unittest (para validación automatizada) y tempfile (para pruebas de persistencia).

---

## Instalación y Uso

Dado que el proyecto está diseñado sin dependencias externas, únicamente se requiere un entorno de Python 3 activo.

### 1. Clonar e Inicializar

Clona este repositorio en tu máquina local y accede al directorio del proyecto:
```bash
git clone https://github.com/juanmmm21/bpe-tokenizer-from-scratch.git
cd bpe-tokenizer-from-scratch
```

### 2. Ejecutar Ejemplo de Demostración
Se incluye un archivo `example.py` que realiza un flujo completo interactivo:
- Inicializa y muestra el tamaño del vocabulario base.
- Entrena el tokenizador con un corpus rico en español.
- Codifica un texto de prueba que contiene caracteres especiales y emojis.
- Desglosa y explica qué caracteres representa cada token ID.
- Realiza la decodificación y valida que sea idéntica al original.
- Guarda el modelo a disco en formato JSON y lo carga en una nueva instancia para verificar la persistencia.

Puedes ejecutarlo con:
```bash
python3 example.py
```

### 3. Ejecutar Pruebas Unitarias
Para asegurar que no haya regresiones y que la implementación se mantenga robusta y correcta, ejecuta el arnés de pruebas:
```bash
python3 -m unittest test_tokenizer.py
```

---

## Conexión con el Ecosistema ai-core-infra

Dentro del marco de la arquitectura global, bpe-tokenizer-from-scratch es la piedra angular del procesamiento de datos:
- Los tokens numéricos generados por este módulo alimentarán directamente el [contrastive-embedding-trainer](https://github.com/juanmmm21/contrastive-embedding-trainer) para entrenar representaciones vectoriales densas.
- El análisis estructural que realiza sirve de inspiración para el corte semántico en [semantic-chunking-engine](https://github.com/juanmmm21/semantic-chunking-engine).
- La codificación y vocabulario salvados son consumidos posteriormente por los servidores de inferencia en [llm-inference-server](https://github.com/juanmmm21/llm-inference-server) para reconstruir y emitir las predicciones.
