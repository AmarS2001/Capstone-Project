from typing import Union
from fastapi import FastAPI
from pydantic import BaseModel


class Item(BaseModel):
    input: int


app = FastAPI()

@app.post("/node1/")
async def create_item(item: Item):
    output = item.input + 10
    return output