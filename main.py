from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List
from meme_generator import run_workflow

app = FastAPI(title="Meme Generator API", version="1.0.0")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",  # Next.js default port
        "http://127.0.0.1:3000",
        "https://meme-generator-frontend-five.vercel.app/",  # Add your production domain here
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

class UserQuery(BaseModel):
    query: str = Field(..., min_length=1, max_length=500, description="User query for meme generation")

class MemeResponse(BaseModel):
    success: bool
    message: str
    meme_urls: List[str]

@app.get("/")
async def root():
    return {"message": "Welcome to Meme Generator API!"}

@app.post('/generate-meme', response_model=MemeResponse)
async def handle_generate_meme(user_query: UserQuery):
    try:
        # Call your meme generation function
        result = await run_workflow(user_query.query)
        print("result", result)
        return MemeResponse(
            success=True,
            message="Meme generated successfully!",
            meme_urls=result
        )
    except Exception as e:
        # Handle any errors from the meme generation
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to generate meme: {str(e)}"
        )

# To run this server, use the command:
# uvicorn main:app --reload