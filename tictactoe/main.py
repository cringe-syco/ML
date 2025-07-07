

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi import Request
import uvicorn

from game_logic import Game
from probabilities import minimax_probabilities

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

game = Game(prob_calc=minimax_probabilities)

@app.get("/", response_class=HTMLResponse)
async def get(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

from fastapi.responses import JSONResponse

@app.post("/move")
async def move(pos: int):
    game.move(pos)
    return JSONResponse(game.state())

@app.post("/reset")
async def reset():
    game.reset()
    return JSONResponse(game.state())

@app.get("/state")
async def state():
    return JSONResponse(game.state())

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)

