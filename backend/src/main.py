from fastapi import FastAPI

app = FastAPI(title="KishoLens API")

@app.get("/health")
def health_check():
    return {"status": "ok", "service": "backend"}
