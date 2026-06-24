import os
import tempfile
import unittest
from tokenizer import BPETokenizer

class TestBPETokenizer(unittest.TestCase):
    """
    Casos de prueba para verificar la robustez y correctitud de la clase BPETokenizer.
    """

    def setUp(self) -> None:
        self.tokenizer = BPETokenizer()
        # Un corpus simple en español para entrenar de forma rápida en los tests.
        self.corpus = (
            "El procesamiento de lenguaje natural es fascinante. "
            "BPE es un algoritmo de tokenización a nivel de bytes. "
            "¿Funciona correctamente con eñes, acentos y emojis? ¡Sí, 🌟 funciona!"
        )

    def test_minimum_vocab_size_enforced(self) -> None:
        """
        Verifica que se lance una excepción si se intenta entrenar
        con un tamaño de vocabulario inferior al mínimo de 256 bytes.
        """
        with self.assertRaises(ValueError):
            self.tokenizer.train(self.corpus, vocab_size=250)

    def test_train_and_vocab_expansion(self) -> None:
        """
        Verifica que el vocabulario se expanda correctamente al valor deseado.
        """
        target_vocab_size = 300
        self.tokenizer.train(self.corpus, vocab_size=target_vocab_size)
        
        # El vocabulario final debe coincidir exactamente con el tamaño objetivo,
        # o ser menor si no hay suficientes pares frecuentes (aunque en este caso el corpus es grande).
        self.assertEqual(len(self.tokenizer.vocab), target_vocab_size)
        self.assertEqual(len(self.tokenizer.merges), target_vocab_size - 256)

    def test_identity_property(self) -> None:
        """
        Propiedad fundamental de identidad: decode(encode(text)) == text.
        Esto garantiza que la tokenización no destruye ni altera información.
        """
        self.tokenizer.train(self.corpus, vocab_size=320)
        
        test_texts = [
            "Hola Mundo",
            "Esta es una prueba con eñes (España) y acentos (canción).",
            "Emojis y símbolos especiales: 🚀 🚀 🤖 !!! ???",
            "",  # Caso límite de texto vacío
            "   Espacios   múltiples   ",
            "12345 y números combinados con texto"
        ]
        
        for text in test_texts:
            encoded = self.tokenizer.encode(text)
            decoded = self.tokenizer.decode(encoded)
            self.assertEqual(decoded, text, f"Fallo de identidad para el texto: {text!r}")

    def test_save_and_load_persistence(self) -> None:
        """
        Verifica que el modelo pueda guardarse a disco en formato JSON
        y cargarse de vuelta conservando la misma lógica de codificación.
        """
        self.tokenizer.train(self.corpus, vocab_size=310)
        original_encoded = self.tokenizer.encode("BPE es un algoritmo increíble.")

        # Usamos un archivo temporal para evitar dejar residuos en el sistema.
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            # Guardamos el modelo.
            self.tokenizer.save(tmp_path)
            
            # Instanciamos un nuevo tokenizador y cargamos el modelo.
            new_tokenizer = BPETokenizer()
            new_tokenizer.load(tmp_path)
            
            # Verificamos que el estado interno sea idéntico.
            self.assertEqual(len(new_tokenizer.vocab), len(self.tokenizer.vocab))
            self.assertEqual(new_tokenizer.merges, self.tokenizer.merges)
            
            # Verificamos que la codificación produzca exactamente los mismos IDs.
            new_encoded = new_tokenizer.encode("BPE es un algoritmo increíble.")
            self.assertEqual(new_encoded, original_encoded)
            
        finally:
            # Limpiamos el archivo temporal.
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def test_unknown_token_id_raises_error(self) -> None:
        """
        Verifica que la decodificación de un ID de token fuera del vocabulario
        lance una excepción KeyError explícita en lugar de fallar silenciosamente.
        """
        self.tokenizer.train(self.corpus, vocab_size=260)
        # ID 999 no existe en un vocabulario de tamaño 260.
        with self.assertRaises(KeyError):
            self.tokenizer.decode([999])

if __name__ == "__main__":
    unittest.main()
