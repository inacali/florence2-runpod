import base64
import io
from typing import Optional

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM

MODEL_ID = "microsoft/Florence-2-large"

device = "cuda" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if device == "cuda" else torch.float32

print(f"Loading Florence-2 model {MODEL_ID} on {device} with dtype={torch_dtype} ...")

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch_dtype,
    trust_remote_code=True,
).to(device)

processor = AutoProcessor.from_pretrained(
    MODEL_ID,
    trust_remote_code=True,
)

app = FastAPI(title="Florence-2-large API")


# ---------- Schemas ----------

class FlorenceRequest(BaseModel):
    image_b64: str          # base64-encoded PNG/JPEG
    task: str               # e.g. "<REGION_PROPOSAL>", "<OD>", "<OCR>", "<CAPTION>"
    text_input: Optional[str] = None  # extra text for some tasks (e.g. caption grounding)
    max_new_tokens: int = 512


class FlorenceResponse(BaseModel):
    raw_text: str           # raw generated text from the model
    parsed: dict            # post-processed structure from processor.post_process_generation


# ---------- Helpers ----------

def decode_image(b64: str) -> Image.Image:
    try:
        data = base64.b64decode(b64)
        img = Image.open(io.BytesIO(data)).convert("RGB")
        return img
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image data: {e}")


# ---------- Routes ----------

@app.get("/healthz")
def healthz():
    return {"status": "ok"}


@app.post("/florence", response_model=FlorenceResponse)
def run_florence(req: FlorenceRequest):
    image = decode_image(req.image_b64)

    # Build Florence prompt
    if req.text_input is None:
        prompt = req.task
    else:
        prompt = req.task + req.text_input

    inputs = processor(
        text=prompt,
        images=image,
        return_tensors="pt",
    ).to(device, torch_dtype)

    with torch.no_grad():
        generated_ids = model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=req.max_new_tokens,
            num_beams=3,
            do_sample=False,
        )

    generated_text = processor.batch_decode(
        generated_ids, skip_special_tokens=False
    )[0]

    # Post-process according to the task
    try:
        parsed = processor.post_process_generation(
            generated_text,
            task=req.task,
            image_size=(image.width, image.height),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Post-processing error: {e}")

    # Make sure parsed is JSON-serializable
    if not isinstance(parsed, dict):
        parsed = {"result": parsed}

    return FlorenceResponse(raw_text=generated_text, parsed=parsed)

