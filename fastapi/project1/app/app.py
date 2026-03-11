from fastapi import FastAPI, HTTPException, File, UploadFile, Form,  Depends
from app.schemas import PostCreate, PostResponse
from app.db import Post, create_db_and_tables, get_async_session
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from contextlib import asynccontextmanager
from app.images import imagekit
# from imagekitio.resources.

@asynccontextmanager
async def lifespan(app: FastAPI):
    await create_db_and_tables()
    yield


app = FastAPI(lifespan=lifespan)



@app.post("/upload")
async def upload_file(
    file: UploadFile = File(...),
    caption: str = Form(""),
    session: AsyncSession = Depends(get_async_session)

):
    post = Post(
        caption = caption,
        url="dummy url",
        file_type="photo",
        file_name = "dummy_name"
    )
    session.add(post)
    await session.commit()
    await session.refresh(post)

    return post

@app.get("/feed")
async def get_feed(
    session: AsyncSession = Depends(get_async_session)
):
    result = await session.execute(
        select(Post).order_by(Post.created_at.desc())
    )
    posts = [row[0] for row in result.all()]
    posts_data = []
    for post in  posts:
        posts_data.append(
            {            
                "id": str(post.id),
                "caption": post.caption,
                "url": post.url,
                "file_type": post.file_type,
                "file_name": post.file_name,
                "created_at" : post.created_at.isoformat()
            }
        )
    return {
        "posts": posts_data
    }




































# --------------------------------------------------------------------
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
def get_post(id: int) -> PostResponse:
    if id not in text_posts:
        raise HTTPException(status_code=404, detail = "Page not Found")
    return text_posts.get(id)


@app.post("/posts")
def create_post(post: PostCreate) -> PostResponse:
    new_post = {
        "title": post.title,
        "contetnt": post.content
    }
    text_posts[max(text_posts.keys()) + 1]  = new_post
    return new_post

# @app.delete