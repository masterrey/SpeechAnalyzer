import logging
import os
import tempfile
import time
from typing import Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from google import genai
from google.genai import errors as genai_errors
from google.genai import types


app = FastAPI(title="SpeechAnalyzer")
logger = logging.getLogger(__name__)

API_KEY = os.environ.get("GEMINI_API_KEY")
if not API_KEY:
    raise RuntimeError("GEMINI_API_KEY não configurada.")
client = genai.Client(api_key=API_KEY)


class AnaliseDiscurso(BaseModel):
    postura_nota: int = Field(..., ge=1, le=10)
    postura_feedback: str
    diccao_nota: int = Field(..., ge=1, le=10)
    diccao_feedback: str
    conteudo_nota: int = Field(..., ge=1, le=10)
    conteudo_feedback: str
    veredicto_final: str


PROMPT_ANALISE = """
Você é um especialista em oratória.
Analise o vídeo enviado considerando linguagem corporal, dicção e conteúdo.
Retorne exclusivamente JSON válido de acordo com o schema fornecido.
"""
MAX_UPLOAD_BYTES = 500 * 1024 * 1024
MAX_WAIT_FILE_ACTIVE_SECONDS = 60
FILE_ACTIVE_POLL_SECONDS = 2
MODOS_VALIDOS = {"rapido", "profundo"}
CONFIG_POR_MODO = {
    "rapido": {
        "model": "gemini-3-flash-preview",
        "thinking_config": types.ThinkingConfig(thinking_level="low"),
    },
    "profundo": {
        "model": "gemini-3.1-pro-preview",
        "thinking_config": types.ThinkingConfig(thinking_level="high"),
    },
}


@app.post("/analisar-video/", response_model=AnaliseDiscurso)
async def analisar_video(file: UploadFile = File(...), modo: str = "profundo") -> AnaliseDiscurso:
    if modo not in MODOS_VALIDOS:
        raise HTTPException(status_code=400, detail="Parâmetro 'modo' inválido. Use 'rapido' ou 'profundo'.")

    if not file.filename:
        raise HTTPException(status_code=400, detail="Arquivo inválido.")

    extensao = os.path.splitext(file.filename)[1].lower()
    if extensao not in {".mp4", ".mov"}:
        raise HTTPException(status_code=400, detail="Envie um arquivo .mp4 ou .mov.")

    temp_path: Optional[str] = None
    arquivo_gemini = None

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=extensao) as temp_file:
            temp_path = temp_file.name
            conteudo = await file.read(MAX_UPLOAD_BYTES + 1)
            if len(conteudo) > MAX_UPLOAD_BYTES:
                raise HTTPException(status_code=413, detail="Arquivo excede o tamanho máximo permitido (500MB).")
            temp_file.write(conteudo)

        try:
            arquivo_gemini = client.files.upload(file=temp_path)
        except Exception:
            logger.exception("Erro no upload para Gemini")
            raise HTTPException(status_code=502, detail="Erro no upload para o Gemini.")

        # O arquivo pode levar alguns segundos para ficar pronto para inferencia.
        deadline = time.time() + MAX_WAIT_FILE_ACTIVE_SECONDS
        while True:
            estado = getattr(arquivo_gemini, "state", None)
            estado_str = str(estado).upper() if estado is not None else ""

            if "ACTIVE" in estado_str:
                break

            if "FAILED" in estado_str:
                raise HTTPException(status_code=502, detail="Falha no processamento do arquivo pelo Gemini.")

            if time.time() >= deadline:
                raise HTTPException(
                    status_code=504,
                    detail="Tempo limite aguardando processamento do arquivo no Gemini.",
                )

            time.sleep(FILE_ACTIVE_POLL_SECONDS)
            arquivo_gemini = client.files.get(name=arquivo_gemini.name)

        config_modo = CONFIG_POR_MODO[modo]

        try:
            resposta = client.models.generate_content(
                model=config_modo["model"],
                contents=[arquivo_gemini, PROMPT_ANALISE],
                config=types.GenerateContentConfig(
                    thinking_config=config_modo["thinking_config"],
                    response_mime_type="application/json",
                    response_schema=AnaliseDiscurso,
                ),
            )
        except genai_errors.ClientError as exc:
            logger.exception("Erro na análise com Gemini")
            status_code = getattr(exc, "code", None)

            if status_code == 429:
                raise HTTPException(
                    status_code=429,
                    detail="Quota da Gemini API excedida. Verifique billing/limites ou tente novamente mais tarde.",
                )

            if status_code == 400:
                raise HTTPException(
                    status_code=400,
                    detail="Requisicao rejeitada pela Gemini API. Verifique formato do arquivo e parametros.",
                )

            raise HTTPException(status_code=502, detail="Erro na análise com Gemini.")

        except Exception:
            logger.exception("Erro na análise com Gemini")
            raise HTTPException(status_code=502, detail="Erro na análise com Gemini.")

        try:
            parsed = resposta.parsed
            if parsed is None:
                raise ValueError("Resposta sem conteúdo estruturado.")
            return parsed
        except Exception:
            logger.exception("Erro ao interpretar resposta estruturada")
            raise HTTPException(status_code=502, detail="Erro ao interpretar resposta do Gemini.")

    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError:
                logger.warning("Falha ao remover arquivo temporário local.", exc_info=True)

        if arquivo_gemini is not None:
            try:
                client.files.delete(name=arquivo_gemini.name)
            except Exception:
                logger.warning("Falha ao remover arquivo temporário no Gemini.", exc_info=True)
