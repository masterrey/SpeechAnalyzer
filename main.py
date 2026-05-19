import os
import tempfile
from typing import Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from google import genai
from google.genai import types


app = FastAPI(title="SpeechAnalyzer")

# Inicialização da SDK moderna do Gemini
client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))


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


@app.post("/analisar-video/", response_model=AnaliseDiscurso)
async def analisar_video(file: UploadFile = File(...)) -> AnaliseDiscurso:
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
            conteudo = await file.read()
            temp_file.write(conteudo)

        try:
            arquivo_gemini = client.files.upload(file=temp_path)
        except Exception as exc:
            raise HTTPException(status_code=502, detail=f"Erro no upload para Gemini: {exc}")

        try:
            resposta = client.models.generate_content(
                model="gemini-1.5-pro",
                contents=[arquivo_gemini, PROMPT_ANALISE],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=AnaliseDiscurso,
                ),
            )
        except Exception as exc:
            raise HTTPException(status_code=502, detail=f"Erro na análise com Gemini: {exc}")

        try:
            parsed = resposta.parsed
            if parsed is None:
                raise ValueError("Resposta sem conteúdo estruturado.")
            return parsed
        except Exception as exc:
            raise HTTPException(status_code=502, detail=f"Erro ao interpretar resposta: {exc}")

    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError:
                pass

        if arquivo_gemini is not None:
            try:
                client.files.delete(name=arquivo_gemini.name)
            except Exception:
                pass
