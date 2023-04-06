#!/usr/bin/env python
from transformers import AutoTokenizer, AutoModel
import asyncio
import websockets
import json
import torch

DEVICE = "cuda"
DEVICE_ID = "0"
CUDA_DEVICE = f"{DEVICE}:{DEVICE_ID}" if DEVICE_ID else DEVICE


def torch_gc():
    if torch.cuda.is_available():
        with torch.cuda.device(CUDA_DEVICE):
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()


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


async def handle_stream(websocket, path):
    async for message in websocket:
        print(message)
        message_dict = json.loads(message)
        for response, history in model.stream_chat(tokenizer, message_dict["prompt"], history=message_dict["history"], max_length=message_dict.get(
                "max_length", 2048),
                top_p=message_dict.get("top_p", 0.7),
                temperature=message_dict.get("temperature", 0.7)):

            print(response)
            print(history)
            # query, response = history[-1]
            answer = {
                "prompt": response,
                "history": [],
                "eof": False,
            }

            await websocket.send(json.dumps(answer))

        answer = {
            "prompt": "",
            "history": history,
            "eof": True,
        }

        await websocket.send(json.dumps(answer))

        torch_gc()

asyncio.get_event_loop_policy().get_event_loop().run_until_complete(
    websockets.serve(handle_stream, '0.0.0.0', 8765))
asyncio.get_event_loop().run_forever()
