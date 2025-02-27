from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(root_path='/api')

# list of allowed origins
origins = [
    "http://localhost:5173",
    "http://vcm-45508.vm.duke.edu"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "Hello world!"}

@app.get("/mean")
def query_mean_model(query: str):
    """
    Query endpoint for the mean model
    """
    # Pass query to some function
    answer = f"Response to the mean query : {query}"
    # answer = f(query) 
    return {"answer": answer}

@app.get("/traditional")
def query_traditional_model(query: str):
    """
    Query endpoint for the traditional model
    """
    # Pass query to some function
    answer = f"Response to the traditional query : {query}"
    # answer = f(query) 
    return {"answer": answer}


@app.get("/deep-learning")
def query_deep_learning_model(query: str):
    """
    Query endpoint for the deep learning model
    """
    # Pass query to some function
    answer = f"Response to the deep learning model query : {query}"
    # answer = f(query) 
    return {"answer": answer}
