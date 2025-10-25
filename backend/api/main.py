# from fastapi import FastAPI, WebSocket, WebSocketDisconnect
# from fastapi.middleware.cors import CORSMiddleware
# import asyncio
# import json

# app = FastAPI()

# # --- CORS configuration ---
# origins = [
#     "http://localhost:3000",
#     "http://127.0.0.1:3000"
# ]

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=origins,
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # --- WebSocket endpoint ---
# @app.websocket("/ws")
# async def websocket_endpoint(websocket: WebSocket):
#     await websocket.accept()
#     print("✅ WebSocket connected")

#     num_players = None
#     try:
#         while True:
#             # Receive JSON message
#             message = await websocket.receive_text()
#             data = json.loads(message)

#             # Handle number of players
#             if "numPlayers" in data:
#                 num_players = data["numPlayers"]
#                 print(f"Number of players: {num_players}")

#             # Handle video data
#             if "frame" in data:
#                 frame_data = data["frame"]
#                 # TODO: decode and process (OpenCV, etc.)
#                 print(f"Received frame ({len(frame_data)} bytes)")

#             # Handle close request
#             if data.get("action") == "close":
#                 print("🚪 Client requested to close connection.")
#                 await websocket.close()
#                 break

#             # Optional: send acknowledgment
#             await websocket.send_json({
#                 "status": "ok",
#                 "players": num_players
#             })

#     except WebSocketDisconnect:
#         print("❌ WebSocket disconnected")
#     except Exception as e:
#         print(f"⚠️ Error: {e}")
#         await websocket.close()

# import base64, cv2, numpy as np

# def decode_frame(base64_data):
#     img_data = base64.b64decode(base64_data.split(",")[1])
#     np_arr = np.frombuffer(img_data, np.uint8)
#     return cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import os, json, base64, cv2, numpy as np, time

app = FastAPI()

# Allow CORS for your frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# last_time = time.time()

# Helper: decode base64 -> OpenCV image
def decode_frame(base64_data: str):
    try:
        if "," in base64_data:
            base64_data = base64_data.split(",")[1]
        img_bytes = base64.b64decode(base64_data)
        np_arr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        return frame
    except Exception as e:
        print(f"⚠️ Error decoding frame: {e}")
        return None

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("✅ WebSocket connected")

    # Prepare output directory
    save_dir = "frames"
    os.makedirs(save_dir, exist_ok=True)

    frame_count = 0
    try:
        while True:
            msg = await websocket.receive_text()
            data = json.loads(msg)

            if "frame" in data:
                frame_count += 1
                frame = decode_frame(data["frame"])

                # now = time.time()
                # print(f"Frame {frame_count} received after {(now - last_time)*1000:.1f} ms")
                # last_time = now

                if frame is not None:
                    h, w, _ = frame.shape
                    print(f"📸 Received frame {frame_count} — {w}x{h}px")

                    # Save every 30th frame (or whatever rate you want)
                    if frame_count % 10 == 0:
                        filename = os.path.join(save_dir, f"frame_{frame_count:04d}.jpg")
                        cv2.imwrite(filename, frame)
                        print(f"💾 Saved frame to {filename}")

                else:
                    print("⚠️ Failed to decode frame")

            # Handle close request
            if data.get("action") == "close":
                print("🚪 Client requested close.")
                break

            await websocket.send_json({"frames_received": frame_count})

    except WebSocketDisconnect:
        print("❌ WebSocket disconnected")
    except Exception as e:
        print(f"⚠️ Error: {e}")
    finally:
        print(f"✅ Total frames processed: {frame_count}")


