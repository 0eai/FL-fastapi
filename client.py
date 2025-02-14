import asyncio
import json
import time
import uuid
import httpx
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from pydantic import BaseModel
import asyncio
import websockets
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from pydantic import BaseModel
import platform, psutil #For system info

# --- Model (same as server) ---
class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 5)
        self.fc2 = nn.Linear(5, 2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# --- Helper functions ---
def generate_dummy_data(num_samples: int = 100):
    data = torch.randn(num_samples, 10)
    labels = torch.randint(0, 2, (num_samples,)).long()  # Binary classification
    return data, labels


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

# --- Pydantic Models (for data validation) ---
class ClientSystemInfo(BaseModel):
    cpu_count: int
    gpu_available: bool
    ram: str

class JoinRequest(BaseModel):
    client_id: str
    system_info: ClientSystemInfo

class ReadyMessage(BaseModel):
    client_id: str

class ModelUpdate(BaseModel):
    client_id: str
    model_state_dict: dict
    data_size: int


# --- Client Class ---

class FederatedClient:
    def __init__(self, server_url: str):
        print("Initializing FederatedClient")
        self.server_url = server_url
        self.client_id = str(uuid.uuid4())
        self.local_model = SimpleNet()
        self.optimizer = optim.Adam(self.local_model.parameters(), lr=0.01)  # Default LR
        self.config = {}  # Store configuration received from server.
        self.websocket: websockets.WebSocketClientProtocol | None = None # Store the WebSocket

    async def connect(self):
        """Connects to the server and handles the main client logic."""
        print("Entering connect()")
        try:
            async with websockets.connect(f"{self.server_url}/{self.client_id}") as websocket:
                self.websocket = websocket
                print("Connected to server")
                await self.send_join_request()
                await self.listen_for_server_messages()
        except ConnectionRefusedError:
            print(f"Failed to connect to the server at {self.server_url}.  Is the server running?")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")

    async def send_join_request(self):
        """Sends a join request to the server."""
        print("Sending join request")
        system_info = self.get_system_info()
        join_request = JoinRequest(client_id=self.client_id, system_info=system_info)
        await self.websocket.send(json.dumps({"type": "join_request", **json.loads(join_request.model_dump_json())}))
        print("Join request sent.")


    async def listen_for_server_messages(self):
        """Listens for messages from the server and calls the appropriate handler."""
        print("Listening for server messages")
        try:
            async for message_str in self.websocket:
                message = json.loads(message_str)
                print(f"Received from server: {message}")
                if message["type"] == "config":
                    self.config = message["data"]
                    print(f"Received config: {self.config}")
                    #Update the optimiser
                    self.optimizer = optim.Adam(self.local_model.parameters(), lr=self.config.get("learning_rate", 0.01))
                    await self.send_ready_message()

                elif message["type"] == "start_training":
                    await self.train_and_send_update()
                elif message["type"] == "training_finished":
                    print("Training finished. Exiting.")
                    return  # Exit the listening loop.

            else:
                print(f"Unknown message type: {message['type']}")
        except Exception as e:
            print("Exception in listen for server message ", e)
            return #Exit the listen loop.

    async def send_ready_message(self):
        """Sends a 'ready' message to the server."""
        print("Sending ready message")
        ready_message = ReadyMessage(client_id=self.client_id)
        await self.websocket.send(json.dumps({"type": "ready", **json.loads(ready_message.model_dump_json())}))
        print("Ready message sent.")


    async def train_and_send_update(self):
        """Downloads the global model, trains locally, and sends the update."""
        #Get Global Model
        print("Getting global model")
        await self.get_global_model()
        # Train
        print("Starting local training...")
        data, labels = generate_dummy_data(128)  # Use more data
        loss = await self.train(data, labels, epochs=3, batch_size=self.config.get("batch_size", 32)) #Use batch size from config, train multiple epochs
        print(f"Local training Loss: {loss:.4f}")
        await self.send_model_update(data.size(0)) #Send data size
        await self.send_ready_message()



    async def get_global_model(self):
        print("Requesting global model")
        async with httpx.AsyncClient() as client:
            url = f"http://localhost:8008/get_global_model"  
            response = await client.get(url)
            if response.status_code == 200:
                model_data = response.json()
                deserialize_model(model_data['model_state_dict'], self.local_model)
                print("Global model downloaded and loaded successfully.")
            else:
                print(f"Error downloading global model. Status code: {response.status_code}")
                raise HTTPException(status_code=response.status_code, detail="Failed to fetch global model")



    async def train(self, data, labels, epochs=1, batch_size=32):
        """Trains the local model for a specified number of epochs."""
        print("Entering train()")
        self.local_model.train()  # Set the model to training mode
        criterion = nn.CrossEntropyLoss()
        total_loss = 0

        for epoch in range(epochs):
            for i in range(0, len(data), batch_size):
                batch_data = data[i:i + batch_size]
                batch_labels = labels[i:i + batch_size]

                self.optimizer.zero_grad()
                outputs = self.local_model(batch_data)
                loss = criterion(outputs, batch_labels)
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
        return total_loss / (len(data) / batch_size)

    async def send_model_update(self, data_size: int):
        """Sends the model update to the server."""
        print("Sending model update")
        serialized_update = serialize_model(self.local_model)
        model_update = ModelUpdate(client_id=self.client_id, model_state_dict=serialized_update, data_size=data_size)
        await self.websocket.send(json.dumps({"type": "model_update", **json.loads(model_update.model_dump_json())}))
        print("Model update sent.")


    def get_system_info(self) -> ClientSystemInfo:
        """Collects system information."""
        print("Getting system info")
        cpu_count = psutil.cpu_count()
        gpu_available = torch.cuda.is_available()
        ram = str(round(psutil.virtual_memory().total / (1024.0 **3)))+"GB"
        return ClientSystemInfo(cpu_count=cpu_count, gpu_available=gpu_available, ram=ram)



# --- Main Execution (Client) ---

async def main_client():
    print("Staring main client")
    client = FederatedClient(server_url="ws://localhost:8008/ws")  # Replace with your server URL
    await client.connect()

if __name__ == "__main__":
    asyncio.run(main_client())