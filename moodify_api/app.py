from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import tempfile
import shutil
import logging

from api_core.predict import predict_from_file

# keeping logs simple so I can track what's happening in real time
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("moodify-api")

app = FastAPI(title="moodify-api")

# loosening CORS for local testing (can tighten later)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/predict")
async def predict(file: UploadFile | None = File(default=None), path: str | None = Form(default=None)):
    # either user uploads a file or points to an existing one
    if not file and not path:
        raise HTTPException(status_code=400, detail="Need either a file or a file path")

    temp_file = None
    try:
        # handle upload by dropping it in a temp file
        if file:
            suffix = Path(file.filename).suffix or ".wav"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                temp_file = Path(tmp.name)
                shutil.copyfileobj(file.file, tmp)
            audio_path = str(temp_file)
        else:
            audio_path = path
            if not Path(audio_path).exists():
                raise HTTPException(status_code=400, detail="Path does not exist")

        # model prediction
        emotion = predict_from_file(audio_path)
        return JSONResponse({"emotion": emotion})

    except Exception as e:
        log.exception("Something went wrong")
        raise HTTPException(status_code=500, detail="Prediction failed")

    finally:
        # cleaning up temp files so they don’t pile up
        if temp_file and temp_file.exists():
            try:
                temp_file.unlink()
            except:
                pass
