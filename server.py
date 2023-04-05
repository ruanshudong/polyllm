#!/usr/bin/env python
from transformers import AutoTokenizer, AutoModel
import asyncio
import websockets


path_or_name = "chatglm-6b-int4-qe"
tokenizer = AutoTokenizer.from_pretrained(
    path_or_name, trust_remote_code=True)
model = AutoModel.from_pretrained(
    path_or_name, trust_remote_code=True).half().cuda()
model.eval()


async def handle(websocket, path):
    async for message in websocket:
        print(message)
        response, history = model.chat(tokenizer,
                                       prompt,
                                       history=message.history,
                                       max_length=message.max_length if message.max_length else 2048,
                                       top_p=message.top_p if message.top_p else 0.7,
                                       temperature=message.temperature if message.temperature else 0.95)
        answer = {
            "prompt": response,
            "history": history,
        }
        torch_gc()
        await websocket.send(message)


asyncio.get_event_loop().run_until_complete(
    websockets.serve(handle, '0.0.0.0', 8765))
asyncio.get_event_loop().run_forever()
