import os
from tokenizer import BPETokenizer

def run_example() -> None:
    # 1. Corpus de entrenamiento ilustrativo y rico en español.
    # Incluye redundancia de palabras comunes para que BPE tenga patrones claros que fusionar.
    corpus = (
        "El procesamiento del lenguaje natural (PLN) es una rama de la inteligencia artificial. "
        "La inteligencia artificial busca que las máquinas entiendan el lenguaje humano. "
        "Para procesar el lenguaje, primero necesitamos dividir el texto en tokens. "
        "El tokenizador BPE (Byte Pair Encoding) es un método muy popular y eficiente para esto. "
        "BPE trabaja a nivel de bytes, fusionando de forma iterativa los pares de bytes más comunes. "
        "¿Por qué a nivel de bytes? Porque nos permite manejar cualquier carácter sin tener tokens desconocidos. "
        "Por ejemplo, palabras con acentos como 'canción', 'acción', 'tecnología' o eñes como 'español' y 'año'. "
        "Incluso funciona perfectamente con emojis modernos como 🚀, 🤖 o 🧠."
    )

    print("=== PASO 1: Inicialización del BPETokenizer ===")
    tokenizer = BPETokenizer()
    print(f"Tamaño inicial del vocabulario base: {len(tokenizer.vocab)} (bytes 0 a 255)")
    print("-" * 60)

    print("\n=== PASO 2: Entrenamiento del Tokenizador (BPE) ===")
    # Definimos un vocabulario final de 300 (44 fusiones sobre los 256 bytes base)
    target_vocab_size = 300
    tokenizer.train(corpus, vocab_size=target_vocab_size, verbose=True)
    print(f"Entrenamiento completado. Nuevo tamaño de vocabulario: {len(tokenizer.vocab)}")
    print("-" * 60)

    print("\n=== PASO 3: Codificación de Texto Nuevo (Generalización) ===")
    # Una frase que comparte vocabulario pero no estaba idéntica en el corpus.
    test_text = "La tecnología del lenguaje artificial es 🚀 y 🧠."
    print(f"Texto a codificar: {repr(test_text)}")
    
    encoded_ids = tokenizer.encode(test_text)
    print(f"IDs resultantes: {encoded_ids}")
    print(f"Cantidad de tokens: {len(encoded_ids)}")
    
    # Desglose de tokens para entender qué representa cada ID
    print("\nDetalle de los tokens generados:")
    for idx in encoded_ids:
        token_bytes = tokenizer.vocab[idx]
        # Intentamos decodificar a string para visualización amigable; si no es UTF-8 válido
        # (debido a que representa un byte parcial de un carácter multi-byte), mostramos la representación de bytes raw.
        try:
            token_str = token_bytes.decode("utf-8")
            display = repr(token_str)
        except UnicodeDecodeError:
            display = f"{token_bytes!r} (byte parcial)"
            
        print(f"  ID: {idx:3d} -> Representación: {display}")
    print("-" * 60)

    print("\n=== PASO 4: Decodificación y Verificación ===")
    decoded_text = tokenizer.decode(encoded_ids)
    print(f"Texto decodificado: {repr(decoded_text)}")
    
    # Verificación de integridad.
    assert decoded_text == test_text, "¡Error! El texto decodificado no coincide con el original."
    print("✓ ¡Verificación de identidad exitosa!")
    print("-" * 60)

    print("\n=== PASO 5: Persistencia del Modelo (Guardar y Cargar) ===")
    model_filename = "bpe_model.json"
    print(f"Guardando el tokenizador en '{model_filename}'...")
    tokenizer.save(model_filename)
    
    print("Instanciando un nuevo tokenizador independiente y cargando el modelo...")
    new_tokenizer = BPETokenizer()
    new_tokenizer.load(model_filename)
    
    new_encoded = new_tokenizer.encode(test_text)
    print(f"IDs codificados con el nuevo tokenizador cargado: {new_encoded}")
    
    assert new_encoded == encoded_ids, "¡Error! Las codificaciones difieren tras la carga del modelo."
    print("✓ ¡Carga y verificación del modelo completada correctamente!")
    
    # Limpiamos el archivo del modelo para no dejar basura local.
    if os.path.exists(model_filename):
        os.remove(model_filename)
        print(f"✓ Archivo temporal '{model_filename}' eliminado de forma limpia.")
    print("-" * 60)

if __name__ == "__main__":
    run_example()
