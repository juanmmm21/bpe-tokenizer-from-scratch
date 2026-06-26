# BPE Tokenizer from Scratch

Este subproyecto implementa un tokenizador Byte Pair Encoding (BPE) a nivel de bytes desarrollado integramente desde cero en Python, sin dependencias externas. Es el componente fundacional en el ciclo de procesamiento de lenguaje natural (PLN), encargado de traducir texto plano en una secuencia de identificadores numericos (tokens) que los modelos autorregresivos (como Transformers) pueden procesar y proyectar en su espacio de embeddings.

## Arquitectura Tecnica y Fundamentos de BPE

A diferencia de los enfoques basados en caracteres Unicode simples o tokens de palabras completas, este tokenizador implementa una aproximacion de nivel de bytes (Byte-Level BPE) inspirada en la utilizada en arquitecturas modernas como GPT-2 y Llama.

### 1. Inicializacion del Vocabulario Base (256)
El vocabulario se inicializa de forma inmutable con los 256 bytes posibles (0 a 255). Esto garantiza la representacion completa de cualquier flujo de texto plano, incluyendo caracteres especiales, acentos, emojis o secuencias Unicode no estandarizadas, sin generar nunca tokens fuera de vocabulario (<unk>).
Para evitar problemas de procesamiento en cadenas con caracteres de control de terminal y espacios en blanco, los bytes se mapean de forma reversible a un rango de caracteres Unicode seguros e imprimibles.

### 2. Pre-tokenizacion por Expresion Regular
Antes de buscar fusiones de bytes, se aplica un filtro de expresiones regulares (con soporte Unicode) que segmenta el texto en palabras individuales, numeros y signos de puntuacion:

```python
# Patron de segmentacion estricto
PATTERN = r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"
```

Esto evita que el algoritmo de BPE fusione caracteres a traves de limites semanticos importantes (por ejemplo, unir la ultima letra de una palabra con un espacio o un signo de puntuacion), lo cual generaria un vocabulario caotico y redundante.

### 3. Algoritmo de Aprendizaje de Fusiones (Training)
Durante el entrenamiento, el algoritmo opera iterativamente sobre el corpus de entrada representado como secuencias de bytes:

1. Se calculan las frecuencias de todos los pares de tokens consecutivos dentro de las palabras segmentadas.
2. Se selecciona el par mas frecuente $(t_i, t_{j})$:
   
   $$(t_i, t_j) = \arg\max_{(x,y)} \text{freq}(x, y)$$
   
3. Se añade una nueva regla de fusion: $(t_i, t_j) \rightarrow t_{new}$.
4. Se actualiza la secuencia sustituyendo todas las ocurrencias consecutivas de $t_i, t_j$ por $t_{new}$.
5. El proceso se repite hasta que el tamano del vocabulario alcanza el limite preestablecido `vocab_size` o no quedan pares con frecuencia mayor que 1.

### 4. Algoritmo de Codificacion (Encoding)
Para codificar una cadena de texto de prueba:
1. Se desglosa en palabras mediante el patron de pre-tokenizacion.
2. Cada palabra se convierte en su representacion de bytes mapeados a caracteres seguros.
3. Se aplican las reglas de fusion aprendidas de forma codiciosa (greedy), respetando estrictamente el orden cronologico de su creacion durante el entrenamiento.
4. Se devuelven los IDs numericos correspondientes.

### 5. Decodificacion Segura (Decoding)
La decodificacion recupera los bytes correspondientes a cada ID del vocabulario y los concatena en un arreglo binario. Posteriormente, se decodifica a UTF-8 utilizando la directiva de reemplazo de errores (`errors="replace"`) para mitigar excepciones ante secuencias truncadas.

## Especificacion del Modelo (Esquema JSON)

El modelo entrenado se persiste como un archivo JSON con la siguiente estructura:

```json
{
  "vocab": {
    "h": 104,
    "o": 111,
    "l": 108,
    "a": 97,
    "ho": 256,
    "ol": 257,
    "hola": 258
  },
  "merges": [
    ["h", "o"],
    ["o", "l"],
    ["ho", "la"]
  ]
}
```

## Requisitos de Instalacion

*   Python 3.10 o superior.
*   Sin dependencias externas.

## Guia de Ejecucion y Verificacion

### 1. Iniciar Entorno y Clonar
```bash
git clone https://github.com/juanmmm21/bpe-tokenizer-from-scratch.git
cd bpe-tokenizer-from-scratch
```

### 2. Ejecutar Pruebas Unitarias
Para verificar la consistencia matematica (ej. propiedad de identidad: `decode(encode(text)) == text`), ejecute:
```bash
python3 -m unittest test_tokenizer.py
```

### 3. Ejecutar Demostracion
Inicie el script de demostracion interactivo para ver la inicializacion, entrenamiento, codificacion de emojis e importacion/exportacion JSON del vocabulario:
```bash
python3 example.py
```

## Conectividad en el Ecosistema ai-core-infra

El tokenizador `bpe-tokenizer-from-scratch` es la base del procesamiento de datos en la infraestructura:
*   Sus IDs de tokens alimentan directamente a [contrastive-embedding-trainer](https://github.com/juanmmm21/contrastive-embedding-trainer) para el entrenamiento de embeddings siameses.
*   Sirve de limite fisico max_tokens en [semantic-chunking-engine](https://github.com/juanmmm21/semantic-chunking-engine).
*   El vocabulario y merges guardados son cargados por [llm-inference-server](https://github.com/juanmmm21/llm-inference-server) para la decodificacion de tokens en streaming.
