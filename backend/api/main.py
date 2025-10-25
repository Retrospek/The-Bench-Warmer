from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import os, json, base64, cv2, numpy as np, time
from computerVision.newtrack import BasketballTrackerSageMaker
import JSON

app = FastAPI()

# Allow CORS for your frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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

                # Process frame

                if frame is not None:
                    h, w, _ = frame.shape
                    print(f"📸 Received frame {frame_count} — {w}x{h}px")

                    # Save every 30th frame (or whatever rate you want)
                    if frame_count % 10 == 0:
                        filename = os.path.join(save_dir, f"frame_{frame_count:04d}.jpg")
                        cv2.imwrite(filename, frame)
                        print(f"💾 Saved frame to {filename}")


                    annotatedImage, shotMade, posession = BasketballTrackerSageMaker.process_frame(frame)
                    message = {
                        "shotMade": 1 if shotMade else 0,
                        "possession": posession,
                    }
                    await websocket.send(JSON.stringify(message))

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


