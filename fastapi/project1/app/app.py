from fastapi import FastAPI, HTTPException

app = FastAPI()

text_posts = {
    1: {"title": "New Post", "content": "cool test post"},
    2: {"title": "Second Post", "content": "This is the second test post"},
    3: {"title": "Third Post", "content": "Another dummy post for testing"},
    4: {"title": "Fourth Post", "content": "Testing the API with dummy data"},
    5: {"title": "Fifth Post", "content": "Lorem ipsum dolor sit amet"},
    6: {"title": "Sixth Post", "content": "More sample content here"},
    7: {"title": "Seventh Post", "content": "This is a test post"},
    8: {"title": "Eighth Post", "content": "Sample content for post 8"},
    9: {"title": "Ninth Post", "content": "Testing with more data"},
    10: {"title": "Tenth Post", "content": "Final dummy post"}
}
 
@app.get("/hello-world")
def hello_world():
    return {"message": "Hello World!"}

@app.get("/posts")
def get_all_posts(limit: int = None):
    if limit:
        return list(text_posts.values())[:limit]
    return text_posts

@app.get("/posts/{id}")
def get_post(id: int):
    if id not in text_posts:
        raise HTTPException(status_code=404, detail = "Page not Found")
    return text_posts.get(id)


@app.post("/posts")
def create_post(title: str, content: str):
    pass