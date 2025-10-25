// "use client";

// import { useState, useRef } from "react";
// import { useRouter } from 'next/navigation'; // For App Router

// export default function PlayGame() {
//   const router = useRouter();
//   const [team1Score, setTeam1Score] = useState(0);
//   const [team2Score, setTeam2Score] = useState(0);
//   const [team1Percentage, setTeam1Percentage] = useState(0);
//   const [team2Percentage, setTeam2Percentage] = useState(0);
//   const [paused, setPaused] = useState(false);
//   const videoRef = useRef<HTMLVideoElement | null>(null);

//   const togglePause = () => setPaused((prev) => !prev);

//   const endGame = () => {
//     if (confirm("Are you sure you want to end the game?")) {
//       alert("Game ended!");
//     }
//     router.push("/stats");
//   };

//   return (
//     <div className="flex flex-col min-h-screen bg-[#3b3b3b] text-white">
//       {/* Top section - stats */}
//       <div className="flex flex-col items-center justify-center bg-[#4a4a4a] text-center py-10 shadow-md">
//         <div className="grid grid-cols-3 w-full max-w-2xl text-xl font-semibold mb-8">
//           <div></div>
//           <div className="text-2xl font-bold">Team 1</div>
//           <div className="text-2xl font-bold">Team 2</div>

//           <div className="text-right pr-4 text-gray-300">Score</div>
//           <div>{team1Score}</div>
//           <div>{team2Score}</div>

//           <div className="text-right pr-4 text-gray-300">Percentage</div>
//           <div>{team1Percentage}%</div>
//           <div>{team2Percentage}%</div>
//         </div>

//         <div className="flex justify-center gap-8 mt-4">
//           <button
//             onClick={togglePause}
//             className="bg-[#ffb300] hover:bg-[#e0a000] text-black font-semibold px-6 py-2 rounded-md shadow-md transition"
//           >
//             {paused ? "Resume" : "Pause"}
//           </button>

//           <button
//             onClick={endGame}
//             className="bg-[#ff5a00] hover:bg-[#e14f00] text-white font-semibold px-6 py-2 rounded-md shadow-md transition"
//           >
//             End
//           </button>
//         </div>
//       </div>

//       {/* Bottom section - live video feed */}
//       <div className="flex-1 bg-[#2f2f2f] relative">
//         <video
//           ref={videoRef}
//           className="w-full h-full object-cover"
//           autoPlay
//           muted
//           playsInline
//         />
//         {paused && (
//           <div className="absolute inset-0 bg-black/70 flex items-center justify-center">
//             <span className="text-4xl font-bold text-white">Paused</span>
//           </div>
//         )}
//       </div>
//     </div>
//   );
// }
"use client";

import { useState, useRef, useEffect } from "react";
import { useRouter } from "next/navigation";

export default function PlayGame() {
  const router = useRouter();
  const [team1Score, setTeam1Score] = useState(0);
  const [team2Score, setTeam2Score] = useState(0);
  const [team1Percentage, setTeam1Percentage] = useState(0);
  const [team2Percentage, setTeam2Percentage] = useState(0);
  const [paused, setPaused] = useState(false);

  const videoRef = useRef<HTMLVideoElement | null>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const captureIntervalRef = useRef<NodeJS.Timeout | null>(null);

  const togglePause = () => setPaused((prev) => !prev);

  const endGame = () => {
    if (confirm("Are you sure you want to end the game?")) {
      // Tell backend to close
      wsRef.current?.send(JSON.stringify({ action: "close" }));
      wsRef.current?.close();
      alert("Game ended!");
      router.push("/stats");
    }
  };

  // 🎥 Set up camera and WebSocket when component mounts
  useEffect(() => {
    let stream: MediaStream;

    const initCameraAndWebSocket = async () => {
      try {
        // Start camera
        stream = await navigator.mediaDevices.getUserMedia({ video: true });
        if (videoRef.current) videoRef.current.srcObject = stream;

        // Connect to FastAPI WebSocket
        const ws = new WebSocket("ws://localhost:8000/ws");
        wsRef.current = ws;

        ws.onopen = () => {
          console.log("✅ WebSocket connected");
          // Send initial game info
          ws.send(JSON.stringify({ numPlayers: 5 }));
        };

        ws.onmessage = (event) => {
          console.log("📩 Server:", event.data);
        };

        ws.onclose = () => {
          console.log("🔌 WebSocket closed");
        };

        // Send 5 frames per second (every 200ms)
        captureIntervalRef.current = setInterval(() => {
          if (!paused && videoRef.current && ws.readyState === WebSocket.OPEN) {
            const frameBase64 = captureFrame(videoRef.current);
            ws.send(JSON.stringify({ frame: frameBase64 }));
          }
        }, 10); // 1000 ms / 5 fps = 200 ms per frame (at least 1 fps per second)
      } catch (err) {
        console.error("Camera/WebSocket setup failed:", err);
      }
    };

    initCameraAndWebSocket();

    // Cleanup on unmount
    return () => {
      if (captureIntervalRef.current) clearInterval(captureIntervalRef.current);
      wsRef.current?.send(JSON.stringify({ action: "close" }));
      wsRef.current?.close();
      stream?.getTracks().forEach((track) => track.stop());
    };
  }, [paused]);

  // 🧠 Utility function to capture a video frame as Base64 image
  const captureFrame = (video: HTMLVideoElement): string => {
    const canvas = document.createElement("canvas");
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const ctx = canvas.getContext("2d");
    if (!ctx) return "";
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    return canvas.toDataURL("image/jpeg", 0.1); // compress to ~10% quality
  };

  return (
    <div className="flex flex-col min-h-screen bg-[#3b3b3b] text-white">
      {/* Top section - stats */}
      <div className="flex flex-col items-center justify-center bg-[#4a4a4a] text-center py-10 shadow-md">
        <div className="grid grid-cols-3 w-full max-w-2xl text-xl font-semibold mb-8">
          <div></div>
          <div className="text-2xl font-bold">Team 1</div>
          <div className="text-2xl font-bold">Team 2</div>

          <div className="text-right pr-4 text-gray-300">Score</div>
          <div>{team1Score}</div>
          <div>{team2Score}</div>

          <div className="text-right pr-4 text-gray-300">Percentage</div>
          <div>{team1Percentage}%</div>
          <div>{team2Percentage}%</div>
        </div>

        <div className="flex justify-center gap-8 mt-4">
          <button
            onClick={togglePause}
            className="bg-[#ffb300] hover:bg-[#e0a000] text-black font-semibold px-6 py-2 rounded-md shadow-md transition"
          >
            {paused ? "Resume" : "Pause"}
          </button>

          <button
            onClick={endGame}
            className="bg-[#ff5a00] hover:bg-[#e14f00] text-white font-semibold px-6 py-2 rounded-md shadow-md transition"
          >
            End
          </button>
        </div>
      </div>

      {/* Bottom section - live video feed */}
      <div className="flex-1 bg-[#2f2f2f] relative">
        <video
          ref={videoRef}
          className="w-full h-full object-cover"
          autoPlay
          muted
          playsInline
        />
        {paused && (
          <div className="absolute inset-0 bg-black/70 flex items-center justify-center">
            <span className="text-4xl font-bold text-white">Paused</span>
          </div>
        )}
      </div>
    </div>
  );
}
