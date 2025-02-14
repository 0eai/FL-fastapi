# Federated Learning with FastAPI and WebSockets

This project demonstrates a basic federated learning system using FastAPI, WebSockets, and PyTorch. It allows multiple clients to collaboratively train a machine learning model without sharing their raw data.

## Overview

The system consists of a central server and multiple clients.

*   **Server:**
    *   Manages the global model.
    *   Coordinates training rounds.
    *   Aggregates model updates from clients (using Federated Averaging - FedAvg).
    *   Provides a REST endpoint for clients to download the initial global model.
    *   Handles client connections and disconnections.
    *   Saves and loads model checkpoints.

*   **Client:**
    *   Connects to the server via WebSockets.
    *   Downloads the global model.
    *   Trains the model on local (dummy) data.
    *   Sends model updates to the server.
    *   Reports system information (CPU, GPU availability, RAM).

## Technologies Used

*   **FastAPI:** A modern, fast (high-performance) web framework for building APIs.
*   **WebSockets:** For real-time, bidirectional communication between server and clients.
*   **PyTorch:**  A popular deep learning framework.
*   **Pydantic:**  For data validation and settings management.
*   **httpx:** An asynchronous HTTP client (used for downloading the global model).
*   **psutil:** For retrieving system information.
*   **uvicorn:**  An ASGI server for running FastAPI.
*   **asyncio:** For asynchronous programming.
*   **websockets:** For WebSocket server and client implementation.

## Project Structure

*   `server.py`: Contains the FastAPI server code.
*   `client.py`: Contains the federated learning client code.
*   `requirements.txt`: Lists the project dependencies.
*   `checkpoints/`: Directory to store model checkpoints (created automatically).
*   `README.md`: This file.

## Setup and Installation

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/0eai/FL-fastapi.git
    cd FL-fastapi
    ```

2.  **Create a virtual environment (recommended):**

    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Linux/macOS
    venv\Scripts\activate  # On Windows
    ```

3.  **Install dependencies:**

    ```bash
    pip install -r requirements.txt
    ```
    **Important:** If you have a GPU and want to use it with PyTorch, make sure you install the correct `torch` version that's compatible with your CUDA version. See the PyTorch website for instructions.  The `requirements.txt` file has the CPU version by default. You might need to change it, for example, to: `torch==2.1.1+cu118`

## Running the System

1.  **Start the server:**

    ```bash
    uvicorn server:app --reload --host 0.0.0.0 --port 8008
    ```
   *   `--reload`:  Enables automatic server reloading on code changes (useful during development).
   * `--host 0.0.0.0`: Allows the server to be accessible from other machines on your network.  If you are only running the client and server on the same machine, you can omit this.
   * `--port 8008`: Specifies the port the server will listen on. You can change this if needed.

2.  **Start the clients (in separate terminals):**

    ```bash
    python client.py
    ```

    You can run multiple instances of `client.py` to simulate multiple clients.  Each client will have a unique ID.

3. **Observe Training**
    The server and client terminals will display log messages showing the training process, client connections, model updates, and aggregation.  Checkpoints will be saved in the `checkpoints/` directory.

## Stopping the System

*   **Clients:** Press Ctrl+C in each client terminal.
*   **Server:** Press Ctrl+C in the server terminal.

## Configuration

*   **`training_rounds` (server.py):**  The total number of training rounds.
*   **`min_number_of_clients` (server.py):** The minimum number of clients required to start a round.
*   **`clients_per_round` (server.py):** The fraction of available clients to select for each round.
*   **`learning_rate` and `batch_size` (sent as `config` from server.py):**  These are sent to the client and can be adjusted.
* **`CHECKPOINT_DIR` and `CHECKPOINT_FILE` (server.py)** Checkpoint file and dir locations.

You can modify these parameters directly in the `server.py` and `client.py` files.  For larger projects, consider using a separate configuration file (e.g., YAML).

## Further Improvements

This project provides a basic framework.  Here are some areas for improvement:

*   **Robust Error Handling:**  Add more comprehensive error handling and logging.
*   **Security:** Implement authentication, authorization, and encryption for production use.
*   **Data Loading:** Replace the dummy data with real datasets.
*   **Advanced Algorithms:** Implement more sophisticated federated learning algorithms (e.g., FedProx, FedAdam).
*   **Differential Privacy:**  Add differential privacy mechanisms for enhanced privacy protection.
*   **Client Selection:**  Implement more sophisticated client selection strategies.
*   **Model Evaluation:**  Add code to evaluate the global model's performance.
*   **User Interface:** Create a UI for monitoring and management.
*   **Scalability:** Optimize for a larger number of clients (e.g., using a message queue).
*   **Testing:** Write unit and integration tests.

## Contributing

Contributions are welcome! Please submit pull requests with clear descriptions of your changes.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.