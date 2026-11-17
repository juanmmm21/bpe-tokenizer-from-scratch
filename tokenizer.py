"""Compatibilidad: el código vive en el paquete `bpe_tokenizer_from_scratch`. Este módulo se
conserva para quien añade esta carpeta a `sys.path` e importa `tokenizer`
directamente."""

from bpe_tokenizer_from_scratch.tokenizer import *  # noqa: F401,F403
from bpe_tokenizer_from_scratch.tokenizer import BPETokenizer  # noqa: F401
