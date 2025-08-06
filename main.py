
from fastapi import FastAPI, status, Request
from fastapi.responses import JSONResponse
import requests, uvicorn
from model import AI_model

BASE_URL = "/api/v1"
app = FastAPI()
model = AI_model()

@app.get(BASE_URL)
async def root():
    return JSONResponse(content={"message": "running"})

@app.post(BASE_URL+"/hackrx/run")
async def hackrx_run(request: Request):
    auth_header = request.headers.get("Authorization")
    if auth_header != "Bearer c6b3c0cec7814bce812bd3ef0758f641d815fee92abbb9a61eb6ec428762de43":
        return JSONResponse(
            status_code=status.HTTP_401_UNAUTHORIZED,
            content={"detail": "Unauthorized"}
        )
    
    data = await request.json()
    document_url = data["documents"]
    questions = data["questions"]

    # get document from link and upload for preprocessing by model
    response = requests.get(document_url)
    if response.status_code != 200:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"detail": "Failed to fetch document"}
        )

    with open("documents/temp_document.pdf", "wb") as f: # if changing file name, update model.upload_docs() to have filename param
        f.write(response.content)
    upload_response = model.upload_docs()
    print(upload_response)

    # run model for each question
    answers = model.run_model(questions)

    return JSONResponse(content={"answers": answers})

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
