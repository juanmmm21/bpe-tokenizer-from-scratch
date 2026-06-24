import json
import re
from typing import Dict, List, Tuple

# Patrón de pre-tokenización basado en el estándar de GPT-2.
# Evita que se fusionen caracteres de distintas categorías (letras, números, signos de puntuación, espacios).
# Se utiliza el módulo 're' estándar con la flag re.UNICODE. El uso de [^\W\d_] permite coincidir
# con caracteres alfabéticos Unicode sin importar el idioma (soportando acentos, eñes, etc.)
# sin requerir la dependencia externa 'regex'.
GPT2_SPLIT_PATTERN = re.compile(
    r"'s|'t|'re|'ve|'m|'ll|'d| ?[^\W\d_]+| ?\d+| ?[^\s\w]+|\s+(?!\S)|\s+",
    re.UNICODE
)

class BPETokenizer:
    """
    Tokenizador BPE (Byte Pair Encoding) a nivel de bytes desarrollado desde cero.
    
    Esta implementación trabaja directamente sobre bytes UTF-8 para garantizar
    que cualquier secuencia de texto pueda ser codificada sin pérdida de información
    (evitando el uso de tokens especiales <unk> para caracteres fuera de vocabulario).
    """

    def __init__(self) -> None:
        # Vocabulario base: mapea ID (int) -> bytes.
        # Inicializado con los 256 bytes posibles (0 a 255).
        self.vocab: Dict[int, bytes] = {i: bytes([i]) for i in range(256)}
        
        # Reglas de fusión aprendidas durante el entrenamiento.
        # Mapea tupla de IDs (int, int) -> nuevo ID (int).
        self.merges: Dict[Tuple[int, int], int] = {}
        
        # Mapeo inverso de vocabulario para optimizar la reconstrucción o debugging.
        self.inverse_vocab: Dict[bytes, int] = {v: k for k, v in self.vocab.items()}

    def _get_stats(self, word_ids_list: List[List[int]]) -> Dict[Tuple[int, int], int]:
        """
        Calcula las frecuencias de todos los pares de IDs consecutivos
        dentro de cada una de las secuencias del corpus de entrenamiento.
        """
        counts: Dict[Tuple[int, int], int] = {}
        for word_ids in word_ids_list:
            for pair in zip(word_ids, word_ids[1:]):
                counts[pair] = counts.get(pair, 0) + 1
        return counts

    def _merge(self, word_ids_list: List[List[int]], pair: Tuple[int, int], new_id: int) -> List[List[int]]:
        """
        Reemplaza todas las apariciones del par de IDs (pair) por el nuevo ID (new_id)
        en todo el corpus estructurado.
        
        Se procesa palabra por palabra para evitar fusiones que crucen los límites
        de la pre-tokenización (los espacios o puntuación).
        """
        new_word_ids_list = []
        for word_ids in word_ids_list:
            if len(word_ids) < 2:
                new_word_ids_list.append(word_ids)
                continue
            
            new_ids: List[int] = []
            i = 0
            while i < len(word_ids):
                # Si encontramos el par a fusionar, añadimos el nuevo ID y avanzamos 2 posiciones.
                if i < len(word_ids) - 1 and word_ids[i] == pair[0] and word_ids[i+1] == pair[1]:
                    new_ids.append(new_id)
                    i += 2
                else:
                    new_ids.append(word_ids[i])
                    i += 1
            new_word_ids_list.append(new_ids)
        return new_word_ids_list

    def train(self, text: str, vocab_size: int, verbose: bool = False) -> None:
        """
        Entrena el tokenizador utilizando Byte Pair Encoding.
        
        Args:
            text: Corpus de texto para el entrenamiento.
            vocab_size: Tamaño objetivo del vocabulario final (debe ser >= 256).
            verbose: Si es True, imprime información del progreso del entrenamiento.
            
        Raises:
            ValueError: Si vocab_size es menor que 256.
        """
        if vocab_size < 256:
            raise ValueError("El tamaño de vocabulario mínimo es 256 (para cubrir todos los bytes base).")
        
        num_merges = vocab_size - 256
        if num_merges <= 0:
            return

        # Pre-tokenización: dividimos el texto en fragmentos lógicos según el patrón regex.
        # Esto previene que se agrupen letras con signos de puntuación o espacios.
        words = GPT2_SPLIT_PATTERN.findall(text)
        
        # Convertimos cada palabra en una secuencia de sus bytes constituyentes (IDs iniciales 0..255).
        word_ids_list: List[List[int]] = [list(w.encode("utf-8")) for w in words]

        if verbose:
            print(f"Iniciando entrenamiento BPE. Vocabulario base: 256. Objetivo: {vocab_size} (fusiones a realizar: {num_merges})")

        for step in range(num_merges):
            # Obtener las frecuencias de pares consecutivos en el corpus actual.
            stats = self._get_stats(word_ids_list)
            if not stats:
                if verbose:
                    print("No quedan más pares de tokens para fusionar. Finalizando entrenamiento prematuramente.")
                break

            # Seleccionar el par más frecuente.
            best_pair = max(stats, key=stats.get)
            new_id = 256 + step

            # Registrar la fusión y actualizar el vocabulario.
            self.merges[best_pair] = new_id
            
            # La representación en bytes del nuevo token es la concatenación de los bytes
            # representados por los dos tokens del par fusionado.
            concat_bytes = self.vocab[best_pair[0]] + self.vocab[best_pair[1]]
            self.vocab[new_id] = concat_bytes
            self.inverse_vocab[concat_bytes] = new_id

            # Aplicar la fusión en el corpus de entrenamiento.
            word_ids_list = self._merge(word_ids_list, best_pair, new_id)

            if verbose and (step + 1) % max(1, num_merges // 10) == 0:
                print(f"Fusión {step + 1}/{num_merges}: {best_pair} -> {new_id} ({self.vocab[new_id]!r}) | Frecuencia: {stats[best_pair]}")

    def encode(self, text: str) -> List[int]:
        """
        Codifica un texto plano en una lista de IDs de tokens.
        
        Utiliza el algoritmo de codificación codiciosa (greedy) aplicando las fusiones
        aprendidas en el orden cronológico en que fueron descubiertas durante el entrenamiento.
        """
        if not text:
            return []

        # Aplicamos la misma pre-tokenización para no romper los límites lógicos de las palabras.
        words = GPT2_SPLIT_PATTERN.findall(text)
        final_ids: List[int] = []

        for word in words:
            # Inicializar cada palabra como bytes individuales.
            ids = list(word.encode("utf-8"))
            
            while len(ids) >= 2:
                # Buscamos todos los pares posibles en la secuencia actual.
                pairs = list(zip(ids, ids[1:]))
                
                # Encontramos cuál de esos pares tiene la fusión de mayor prioridad (menor ID asignado en merges).
                # Esto es crucial: debemos aplicar las fusiones en el mismo orden que en el entrenamiento.
                best_pair = None
                best_rank = float("inf")
                
                for pair in pairs:
                    rank = self.merges.get(pair)
                    if rank is not None and rank < best_rank:
                        best_rank = rank
                        best_pair = pair
                
                # Si ningún par está en nuestras reglas de fusión, no podemos simplificar más esta palabra.
                if best_pair is None:
                    break
                
                # Fusionamos el par con mejor prioridad.
                # Nota: realizamos la fusión local en la palabra de forma iterativa y directa.
                new_id = self.merges[best_pair]
                new_ids: List[int] = []
                i = 0
                while i < len(ids):
                    if i < len(ids) - 1 and ids[i] == best_pair[0] and ids[i+1] == best_pair[1]:
                        new_ids.append(new_id)
                        i += 2
                    else:
                        new_ids.append(ids[i])
                        i += 1
                ids = new_ids
            
            final_ids.extend(ids)
            
        return final_ids

    def decode(self, ids: List[int]) -> str:
        """
        Decodifica una lista de IDs de tokens de vuelta a texto plano en formato string (UTF-8).
        """
        if not ids:
            return ""
        
        # Mapeamos cada ID a su representación en bytes. Si el ID no existe en el vocabulario,
        # lanzamos un KeyError para evitar silenciar errores de tokenización inconsistente.
        byte_chunks = []
        for idx in ids:
            if idx not in self.vocab:
                raise KeyError(f"El ID de token {idx} no se encuentra en el vocabulario.")
            byte_chunks.append(self.vocab[idx])
            
        byte_data = b"".join(byte_chunks)
        
        # Decodificación UTF-8 robusta. Usamos 'replace' para no colapsar el programa
        # en caso de recibir secuencias de bytes truncadas accidentalmente.
        return byte_data.decode("utf-8", errors="replace")

    def save(self, file_path: str) -> None:
        """
        Guarda las reglas de fusión y el vocabulario en un archivo JSON.
        Esto permite reconstruir el tokenizador exactamente con las mismas reglas.
        """
        # Para serializar a JSON, convertimos las claves tupla de merges a strings "id1,id2".
        serialized_merges = {f"{k[0]},{k[1]}": v for k, v in self.merges.items()}
        
        # El vocabulario se puede derivar a partir de las fusiones si se cargan secuencialmente,
        # pero también guardaremos la información explícita para validación.
        # Dado que los bytes no son serializables a JSON por defecto, los guardamos como cadenas hexadecimales.
        serialized_vocab = {str(k): v.hex() for k, v in self.vocab.items()}
        
        model_data = {
            "merges": serialized_merges,
            "vocab": serialized_vocab
        }
        
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(model_data, f, indent=4, ensure_ascii=False)

    def load(self, file_path: str) -> None:
        """
        Carga las reglas de fusión y el vocabulario desde un archivo JSON.
        """
        with open(file_path, "r", encoding="utf-8") as f:
            model_data = json.load(f)
            
        # Reconstruir merges: mapear "id1,id2" de vuelta a Tuple[int, int].
        self.merges = {}
        for k, v in model_data["merges"].items():
            id1, id2 = map(int, k.split(","))
            self.merges[(id1, id2)] = int(v)
            
        # Reconstruir vocab: mapear ID (int) -> bytes.
        self.vocab = {}
        for k, v in model_data["vocab"].items():
            self.vocab[int(k)] = bytes.fromhex(v)
            
        # Reconstruir inverse_vocab.
        self.inverse_vocab = {v: k for k, v in self.vocab.items()}
