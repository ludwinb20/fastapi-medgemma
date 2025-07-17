# 🩺 MedGemma API – Medical Image Analysis with FastAPI

This is a demo API built using **FastAPI** and **Docker** to interface with the `google/medgemma-4b-it` vision-language model for medical image analysis. The model is loaded via Hugging Face Transformers and accepts medical image uploads to produce structured insights in Spanish.

---

## 🚀 Features

- Analyze medical images (X-rays, MRIs, CTs) using MedGemma.
- Two endpoints:
  - `POST /analyze`: basic analysis with a fixed prompt.
  - `POST /analyze-async`: contextual analysis with exam type and clinical notes.
- Responses include model result, processing time, and model name.
- GPU-accelerated (requires CUDA).

---

## 📁 Project Structure

```
├── app/
│   ├── main.py         # FastAPI entry point
│   └── requirements.txt
├── Dockerfile
└── README.md
```

---

## 🐋 Run with Docker

### 1. Set up Hugging Face Token

You must have a valid Hugging Face token to access the model.

Create a `.env` file or export it before running Docker:

```bash
export HF_TOKEN=your_huggingface_token
```

### 2. Build and run the container

```bash
docker build -t medgemma-api .
docker run --gpus all -p 8000:8000 --env HF_TOKEN=$HF_TOKEN medgemma-api
```

> **Note**: Make sure you have `nvidia-container-toolkit` installed to allow GPU access.

---

## 🧪 API Usage

### ➤ `POST /analyze`

Analyze an image using a default prompt.

#### Request

- **file**: image (e.g. PNG, JPG)
- **prompt** *(optional)*: custom prompt (defaults to Spanish medical findings)

#### Example using curl

```bash
curl -X POST http://localhost:8000/analyze \
  -F "file=@chest_xray.jpg"
```

---

### ➤ `POST /analyze-async`

Analyze an image with context (exam type + clinical notes).

#### Request

- **file**: image
- **exam_type**: one of:
  - `radiografía de tórax`
  - `resonancia magnética cerebral`
  - `tomografía abdominal`
- **clinical_notes**: free text in Spanish

#### Example

```bash
curl -X POST http://localhost:8000/analyze-async \
  -F "file=@mri.png" \
  -F "exam_type=resonancia magnética cerebral" \
  -F "clinical_notes=Paciente con antecedentes de epilepsia"
```

---

## ✅ Requirements (for local dev)

If you prefer running locally (outside Docker):

```bash
pip install -r requirements.txt
```

And then:

```bash
uvicorn app.main:app --reload
```

Make sure you set `HF_TOKEN` as an environment variable.

---

## ⚠️ Notes

- This API is a **demo** and not intended for clinical use.
- Ensure you have a GPU with enough memory to load the MedGemma model (~16GB+ recommended).
- This project uses `torch_dtype=torch.float32` for broader compatibility, but can be adapted to `bfloat16` if your hardware allows.

---

## 📬 Contact

For questions or collaboration, reach out to the project maintainer.
