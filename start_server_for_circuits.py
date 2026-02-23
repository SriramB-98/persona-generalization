from circuit_tracer.utils import create_graph_files
from circuit_tracer.frontend.local_server import serve
import argparse
import time
import logging

# Configure the logging system
logging.basicConfig(level=logging.DEBUG)

parser = argparse.ArgumentParser(description="Start a local server to visualize circuit tracing results.")
parser.add_argument("--graph-dir", type=str, required=True, help="Directory containing graph files (output from visualize_circuits.py).")
parser.add_argument("--port", type=int, default=8041, help="Port to run the local server on.")
args = parser.parse_args()

server = serve(data_dir=str(args.graph_dir), port=args.port)
print(f"Server running at http://localhost:{args.port}")
print("Press Ctrl+C to stop.")
try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    server.stop()