import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.routers import pacs_routes, diagnostics_routes, test_routes

app = FastAPI()

origins = ["*","http://localhost:3000"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["Content-Disposition"]
)

# config router
app.include_router(pacs_routes.router)

app.include_router(diagnostics_routes.router)

app.include_router(test_routes.router)

@app.get("/")
async def root():
    return {"message": "Hello Bigger Applications!"}
