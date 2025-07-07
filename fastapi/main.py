from fastapi import FastAPI

app = FastAPI()

items = []

@app.get("/")
def root():
    return {"Hello": "World"}

@app.post("/items")
def create_item(item: str):
    items.append(item)
    return {"item": item}

@app.get("/items/{item_id}")
def get_items(item_id: int = None):
    if item_id is not None:
        return {"item": items[item_id]} if item_id < len(items) else {"error": "Item not found"}