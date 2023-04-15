#!/usr/bin/env python
from transformers import AutoTokenizer, AutoModel
import asyncio
import websockets
import json
import torch
import argparse
import os


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--host",
        type=str,
        nargs="+",
        default=("0.0.0.0"),
        help="server host",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="server port",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="model path",
    )
    parser.add_argument(
        "--cuda",
        type=str,
        default="0",
        help="cuda device",
    )
    args = parser.parse_args()

    return args


args = parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda

# DEVICE = "cuda"
# DEVICE_ID = "0"
# CUDA_DEVICE = f"{DEVICE}:{DEVICE_ID}" if DEVICE_ID else DEVICE


def torch_gc():
    if torch.cuda.is_available():
        with torch.cuda.device(CUDA_DEVICE):
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()


# 载入Tokenizer
tokenizer = AutoTokenizer.from_pretrained("models", trust_remote_code=True)
config = AutoConfig.from_pretrained(
    "models", trust_remote_code=True, pre_seq_len=128)
model = AutoModel.from_pretrained(
    "models", config=config, trust_remote_code=True)
prefix_state_dict = torch.load(os.path.join(args.path, "pytorch_model.bin"))
new_prefix_state_dict = {}
for k, v in prefix_state_dict.items():
    if k.startswith("transformer.prefix_encoder."):
        new_prefix_state_dict[k[len("transformer.prefix_encoder."):]] = v
        model.transformer.prefix_encoder.load_state_dict(
            new_prefix_state_dict)

model = model.quantize(4)
model = model.half().cuda()
model.transformer.prefix_encoder.float()
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
    websockets.serve(handle, args.host, args.port))
asyncio.get_event_loop().run_forever()
