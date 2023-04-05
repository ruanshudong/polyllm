#!/usr/bin/env python
from transformers import AutoTokenizer, AutoModel
import asyncio
import websockets
import json


path_or_name = "chatglm-6b-int4-qe"
tokenizer = AutoTokenizer.from_pretrained(
    path_or_name, trust_remote_code=True)
model = AutoModel.from_pretrained(
    path_or_name, trust_remote_code=True).half().cuda()
model.eval()


async def handle(websocket, path):
    async for message in websocket:
        print(message)
        message_dict = json.loads(message)
        response, history = model.chat(tokenizer,
                                       message_dict["prompt"],
                                       history=message_dict["history"],
                                       max_length=message_dict.get(
                                           "max_length", 2048),
                                       top_p=message_dict.get("top_p", 0.7),
                                       temperature=message_dict.get("temperature", 0.7))
        answer = {
            "prompt": response,
            "history": history,
        }
        torch_gc()
        await websocket.send(json.dumps(answer))

asyncio.get_event_loop_policy().get_event_loop().run_until_complete(
    websockets.serve(handle, '0.0.0.0', 8765))

# asyncio.get_event_loop().run_until_complete(websockets.serve(handle, '0.0.0.0', 8765))
# asyncio.get_event_loop().run_forever()
