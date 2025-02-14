import os
import asyncio
import json
import random
import time
import uuid
from typing import Dict, List, Set

import torch
import torch.nn as nn
import torch.nn.functional as F
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from pydantic import BaseModel

app = FastAPI()

# --- Model ---
class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 5)
        self.fc2 = nn.Linear(5, 2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# --- Global Model and State ---
global_model = SimpleNet()
global_model_lock = asyncio.Lock()  # For thread-safe model updates
connected_clients: Dict[str, WebSocket] = {}
client_system_info: Dict[str, dict] = {}  # Store client system info
ready_clients: Set[str] = set()
training_rounds = 10  # Total number of training rounds
min_number_of_clients = 2
clients_per_round = 1.0 # Select clients 50% in each round
current_round = 0
selected_clients_for_round: Set[str] = set()
received_updates: Dict[str, dict] = {}  # Store updates from clients for current round

# --- Checkpointing ---
CHECKPOINT_DIR = "checkpoints"  # Directory to store checkpoints
CHECKPOINT_FILE = os.path.join(CHECKPOINT_DIR, "global_model.pth")

def save_checkpoint(model, round_num):
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)  # Ensure directory exists
    checkpoint = {
        'round': round_num,
        'model_state_dict': serialize_model(model),
    }
    torch.save(checkpoint, CHECKPOINT_FILE)
    print(f"Checkpoint saved to {CHECKPOINT_FILE}")

def load_checkpoint(model):
    global current_round  # Need to modify the global variable
    if os.path.exists(CHECKPOINT_FILE):
        checkpoint = torch.load(CHECKPOINT_FILE)
        deserialize_model(checkpoint['model_state_dict'], model)
        current_round = checkpoint['round']
        print(f"Checkpoint loaded from {CHECKPOINT_FILE}, resuming from round {current_round}")
        return True
    else:
        print("No checkpoint found, starting from scratch.")
        return False
    
# --- Pydantic Models (for data validation) ---
class ClientSystemInfo(BaseModel):
    cpu_count: int
    gpu_available: bool
    ram: str

class JoinRequest(BaseModel):
    client_id: str
    system_info: ClientSystemInfo

class ReadyMessage(BaseModel):    #Message for client to notify server he is ready
    client_id: str

class ModelUpdate(BaseModel):
    client_id: str
    model_state_dict: dict
    data_size: int    #Number of samples which were used for local training


# --- Helper Functions ---
def serialize_model(model: nn.Module) -> dict:
    return {k: v.tolist() for k, v in model.state_dict().items()}

def deserialize_model(model_state_dict: dict, model: nn.Module):
    state_dict = {}
    for k, v in model_state_dict.items():
        if isinstance(v, list):
            # Convert lists back to tensors
            state_dict[k] = torch.tensor(v)
        else:
            # If it was already a tensor (shouldn't normally happen
            # with tolist() serialization, but good to be safe)
            state_dict[k] = v.clone().detach()  # Or just v if you are sure
    model.load_state_dict(state_dict)

async def aggregate_updates(updates: Dict[str, ModelUpdate]):
    """Aggregates model updates from multiple clients (FedAvg)."""
    global global_model

    # Use a weighted average based on the number of data samples.
    total_data_size = sum(update.data_size for update in updates.values())
    averaged_state_dict = {}

    for client_id, update in updates.items():
        weight = update.data_size / total_data_size
        for k, v in update.model_state_dict.items():
            if isinstance(v, list):
                v_tensor = torch.tensor(v)  # Convert list to tensor
            else:
                v_tensor = v  # Already a tensor, no need to convert.
            
            if k not in averaged_state_dict:
                averaged_state_dict[k] = v_tensor * weight
            else:
                averaged_state_dict[k] += v_tensor * weight

    async with global_model_lock:
        deserialize_model(averaged_state_dict, global_model)

# --- WebSocket Endpoints ---
@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    await websocket.accept()
    connected_clients[client_id] = websocket
    print(f"Client {client_id} connected. Total clients: {len(connected_clients)}")
    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            print(f"Received message from {client_id}: {message["type"]}")
            
            if "type" not in message:
                print(f"Ignoring the message from client {client_id}, message type not present")
                continue
            
            if message["type"] == "join_request":
                try:
                    join_request = JoinRequest(**message)
                    # Store client system information
                    client_system_info[join_request.client_id] = join_request.system_info.dict()
                    # Send config to the client
                    config = {"learning_rate": 0.01, "batch_size": 32}
                    await websocket.send_json({"type": "config", "data": config})
                except Exception as e:
                    print(f"An exception occured while parsing JoinRequest, {str(e)}")
                    await websocket.close(code=4000, reason=str(e)) #4000 custom code for parsing error
                    return
            
            elif message["type"] == "ready":
                try:
                    ready_message = ReadyMessage.model_validate(message)
                    ready_clients.add(ready_message.client_id)
                    await handle_training_start()
                except Exception as e:
                    print(f"An exception occured while parsing ReadyMessage, {str(e)}")
                    await websocket.close(code=4000, reason=str(e))
                    return  # Close and remove the connection.

            elif message["type"] == "model_update":
                try:
                    update = ModelUpdate(**message)
                    received_updates[update.client_id] = update
                    await check_and_aggregate()
                except Exception as e:
                    print(f"An exception occured while parsing ReadyMessage, {str(e)}")
                    await websocket.close(code=4000, reason=str(e))
                    return  # Close and remove the connection.
            else:
                print(f"Ignoring the message from client {client_id}, message type: {message.get('type')} not supported")


    except WebSocketDisconnect:
        print(f"Client {client_id} disconnected.")
    finally:
        # Clean up on disconnect
        if client_id in connected_clients:
            del connected_clients[client_id]
        if client_id in ready_clients:
            ready_clients.remove(client_id)
        if client_id in client_system_info:
            del client_system_info[client_id]
        # If client disconnects during selection, remove it.
        if client_id in selected_clients_for_round:
            selected_clients_for_round.remove(client_id)
        if client_id in received_updates:
            del received_updates[client_id]

async def handle_training_start():
    global current_round, selected_clients_for_round
    if len(ready_clients) == min_number_of_clients and current_round < training_rounds:
        # All clients are ready, start a training round.
        current_round += 1
        print(f"[Round: {str(current_round).zfill(4)}] Starting training round {current_round}/{training_rounds}")
        selected_clients = random.sample(
            list(ready_clients), k=int(min(len(ready_clients), len(ready_clients) * clients_per_round))
        )
        selected_clients_for_round = set(selected_clients)
        received_updates.clear()  # Clear previous updates

        for client_id in selected_clients:
            websocket = connected_clients[client_id]
            await websocket.send_json({"type": "start_training"})
            print(f"[Round: {str(current_round).zfill(4)}] sent start training to {client_id}")
        ready_clients.clear()


async def check_and_aggregate():
    global current_round, selected_clients_for_round, received_updates

    if len(received_updates) == len(selected_clients_for_round):
        print(f"[Round: {str(current_round).zfill(4)}] Aggregating updates...")
        model_updates =  received_updates #All updates stored correctly.
        await aggregate_updates(model_updates)
        received_updates.clear() #Clear the updates
        selected_clients_for_round.clear() # Clear the selection
        # Clear ready clients to force clients to send ready message again.
        # ready_clients.clear()
        print(f"[Round: {str(current_round).zfill(4)}] Round {current_round} completed.")
        
        # Save checkpoint after each round
        save_checkpoint(global_model, current_round) #Save the checkpoint
        
        # Prepare for the next round or finish.
        if current_round < training_rounds:
            await asyncio.sleep(1)  # Give clients time to get ready
            # Start next round, server is contentiously listing for client messages
        else:
            await finish_training()


async def finish_training():
    print("Training completed. Sending finish message to all clients.")
    for client_id, websocket in connected_clients.items():
        await websocket.send_json({"type": "training_finished"})
    print("All rounds completed server can exit now...")


@app.get("/get_global_model")
async def get_global_model_endpoint():
    async with global_model_lock:
        serialized_model_data = serialize_model(global_model)
    return {"model_state_dict": serialized_model_data}
