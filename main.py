import os
import uuid
import tempfile
from typing import Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from google import genai
from google.genai import types


# ---------------------------------------------------------------------------
# Pydantic v2 – response schema (Structured Output)
# ---------------------------------------------------------------------------

class AnaliseOratoria(BaseModel):
    postura_nota: int = Field(
        ...,
        ge=1,
        le=10,
        description="Nota de 1 a 10 para linguagem corporal, gestos e contato visual.",
    )
    postura_feedback: str = Field(
        ...,
        description="Feedback detalhado sobre postura, gestos e contato visual.",
    )
    diccao_nota: int = Field(
        ...,
        ge=1,
        le=10,
        description="Nota de 1 a 10 para ritmo, clareza, pausas e vícios de linguagem.",
    )
    diccao_feedback: str = Field(
        ...,
        description="Feedback detalhado sobre dicção, ritmo e clareza.",
    )
    conteudo_nota: int = Field(
        ...,
        ge=1,
        le=10,
        description="Nota de 1 a 10 para estrutura do discurso, retórica e coerência.",
    )
    conteudo_feedback: str = Field(
        ...,
        description="Feedback detalhado sobre conteúdo, estrutura e retórica.",
    )
    veredicto_final: str = Field(
        ...,
        description="Resumo dos pontos fortes e principais melhorias sugeridas.",
    )


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="SpeechAnalyzer API",
    description="Análise de oratória via Google Gemini (multimodal).",
    version="1.0.0",
)

ALLOWED_CONTENT_TYPES = {"video/mp4", "video/quicktime"}
ALLOWED_EXTENSIONS = {".mp4", ".mov"}

# Maximum accepted upload size
MAX_UPLOAD_MB = 500
MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1024 * 1024

# Both MP4 and MOV use the ISO Base Media File Format; the first box is either
# "ftyp" (File Type Box) or "moov"/"wide" in older QuickTime files.
_FTYP = b"ftyp"
_MOOV = b"moov"
_WIDE = b"wide"  # older QuickTime marker

PROMPT = """
Você é um coach especialista em oratória e comunicação. Analise cuidadosamente
o vídeo fornecido — avaliando tanto a imagem quanto o áudio — e retorne uma
avaliação estruturada nos seguintes critérios:

1. POSTURA (linguagem corporal, gestos, contato visual com a câmera)
2. DICÇÃO (ritmo, clareza, pausas estratégicas, vícios de linguagem)
3. CONTEÚDO (estrutura do discurso, argumentação, retórica e coerência)
4. VEREDICTO FINAL (pontos fortes + principais melhorias)

Para cada critério de nota (postura, dicção, conteúdo) atribua um valor inteiro
de 1 a 10 e escreva um feedback construtivo em português.
"""


@app.post("/analisar-video/", response_model=AnaliseOratoria, summary="Analisa oratória de um vídeo")
async def analisar_video(file: UploadFile = File(..., description="Arquivo de vídeo (.mp4 ou .mov)")):
    """
    Recebe um arquivo de vídeo, envia ao Google Gemini 1.5 Pro para análise
    multimodal de oratória e retorna a avaliação estruturada em JSON.
    """

    # ------------------------------------------------------------------
    # 1. Validate file extension and declared content-type
    # ------------------------------------------------------------------
    original_filename = file.filename or ""
    ext = os.path.splitext(original_filename)[1].lower()

    if ext not in ALLOWED_EXTENSIONS or not file.content_type or file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=415,
            detail="Formato de arquivo não suportado. Envie um arquivo .mp4 ou .mov.",
        )

    # ------------------------------------------------------------------
    # 2. Initialise Gemini client (before the try/finally so it is always
    #    in scope for the cleanup block)
    # ------------------------------------------------------------------
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=500,
            detail="GEMINI_API_KEY não configurada no servidor.",
        )

    client = genai.Client(api_key=api_key)

    # ------------------------------------------------------------------
    # 3. Read upload with size guard, then verify magic bytes
    # ------------------------------------------------------------------
    tmp_path: Optional[str] = None
    gemini_file: Optional[types.File] = None

    try:
        content = await file.read(MAX_UPLOAD_BYTES + 1)
        if len(content) > MAX_UPLOAD_BYTES:
            raise HTTPException(
                status_code=413,
                detail=f"Arquivo muito grande. Tamanho máximo permitido: {MAX_UPLOAD_MB} MB.",
            )

        # Server-side magic-byte check (first 12 bytes cover the ISO box header)
        header = content[:12]
        if len(header) >= 8 and header[4:8] not in (_FTYP, _MOOV, _WIDE):
            raise HTTPException(
                status_code=415,
                detail="O conteúdo do arquivo não corresponde a um vídeo MP4/MOV válido.",
            )

        # ------------------------------------------------------------------
        # 4. Save to a temporary local file
        # ------------------------------------------------------------------
        suffix = f"_{uuid.uuid4().hex}{ext}"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp_path = tmp.name
            tmp.write(content)

        # ------------------------------------------------------------------
        # 5. Upload to Gemini Files API
        # ------------------------------------------------------------------
        try:
            gemini_file = client.files.upload(
                file=tmp_path,
                config=types.UploadFileConfig(
                    display_name=original_filename,
                    mime_type=file.content_type,
                ),
            )
        except Exception as exc:
            raise HTTPException(
                status_code=502,
                detail=f"Falha ao enviar vídeo para a API do Gemini: {exc}",
            )

        # ------------------------------------------------------------------
        # 6. Run multimodal analysis with Structured Output
        # ------------------------------------------------------------------
        try:
            response = client.models.generate_content(
                model="gemini-1.5-pro",
                contents=[gemini_file, PROMPT],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=AnaliseOratoria,
                ),
            )
        except Exception as exc:
            raise HTTPException(
                status_code=502,
                detail=f"Falha ao obter resposta do Gemini: {exc}",
            )

        # ------------------------------------------------------------------
        # 7. Parse and return structured result
        # ------------------------------------------------------------------
        try:
            result: AnaliseOratoria = response.parsed
            if result is None:
                raise ValueError("Resposta vazia do modelo.")
        except Exception as exc:
            raise HTTPException(
                status_code=502,
                detail=f"Resposta do Gemini não pôde ser interpretada: {exc}",
            )

        return result

    finally:
        # ------------------------------------------------------------------
        # 8. Cleanup – local temp file
        # ------------------------------------------------------------------
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass  # best-effort cleanup

        # ------------------------------------------------------------------
        # 9. Cleanup – Gemini cloud file
        # ------------------------------------------------------------------
        if gemini_file is not None:
            try:
                client.files.delete(name=gemini_file.name)
            except Exception:
                pass  # best-effort cleanup
