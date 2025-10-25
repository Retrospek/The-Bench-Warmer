"use client";

import { useState, useRef, useEffect } from "react";
import { useRouter, useSearchParams } from "next/navigation";

function timeToSeconds(time: string): number {
  const [minutes, seconds] = time.split(":").map(Number);

  if (isNaN(minutes) || isNaN(seconds)) {
    throw new Error("Invalid time format. Use mm:ss (e.g., '1:30').");
  }

  return minutes * 60 + seconds;
}


export default function PlayGame() {
  const router = useRouter();
  const searchParams = useSearchParams(); 

  const [team1Score, setTeam1Score] = useState(0);
  const [team2Score, setTeam2Score] = useState(0);
  const [team1Percentage, setTeam1Percentage] = useState(0);
  const [team2Percentage, setTeam2Percentage] = useState(0);
  const [paused, setPaused] = useState(false);
  const [timeLeft, setTimeLeft] = useState(60 * 2); // 🕒 2 minutes
  const [initalTime, setInitalTime] = useState(0);

  const videoRef = useRef<HTMLVideoElement | null>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const captureIntervalRef = useRef<NodeJS.Timeout | null>(null);
  const timerRef = useRef<NodeJS.Timeout | null>(null);

  const togglePause = () => setPaused((prev) => !prev);

  const endGame = () => {
    if (confirm("Are you sure you want to end the game?")) {
      wsRef.current?.send(JSON.stringify({ action: "close" }));
      wsRef.current?.close();
      alert("⏹️ Game ended!");
      console.log(initalTime, timeLeft);
      router.push(`/stats?timeElapsed=${initalTime - timeLeft}`);
    }
  };

  useEffect(() => {
    const timeParam = searchParams.get("time");
    console.log("Time Param", timeParam);
    if (timeParam) {
      setInitalTime(timeToSeconds(timeParam));
      setTimeLeft(timeToSeconds(timeParam)); // convert string -> number
    } else {
      setTimeLeft(120); // default fallback
    }
  }, [])

  // ⏱ Countdown timer
  useEffect(() => {
    if (paused) {
      if (timerRef.current) clearInterval(timerRef.current);
      return;
    }

    timerRef.current = setInterval(() => {
      setTimeLeft((prev) => {
        if (prev <= 1) { // bug when continues but can ignore
          clearInterval(timerRef.current!);
          endGame();
          return 0;
        }
        return prev - 1;
      });
    }, 1000);
    return () => {
      if (timerRef.current) clearInterval(timerRef.current);
    };
  }, [paused, searchParams]);

  // 🎥 Camera + WebSocket setup
  useEffect(() => {
    let stream: MediaStream;

    const initCameraAndWebSocket = async () => {
      try {
        stream = await navigator.mediaDevices.getUserMedia({ video: true });
        if (videoRef.current) videoRef.current.srcObject = stream;

        const ws = new WebSocket("ws://localhost:8000/ws");
        wsRef.current = ws;

        ws.onmessage = (event) => {
          try {
            const data = JSON.parse(event.data);

            // 🎥 Update annotated frame if present
            // if (data.frame) {
            //   if (typeof data.frame === "string" && data.frame.startsWith("data:image")) {
            //     setAnnotatedFrame(data.frame);
            //   } else {
            //     setAnnotatedFrame(`data:image/jpeg;base64,${data.frame}`);
            //   }
            // }

            // 🧮 Update other game data if present
            // if (typeof data.numPlayersTeam1 === "number") {
            //   setTeam1Percentage(data.numPlayersTeam1);
            // }

            // if (typeof data.numPlayersTeam2 === "number") {
            //   setTeam2Percentage(data.numPlayersTeam2);
            // }

            if ((typeof data.shotMade === "number") || (typeof data.possession === "number")) {
              // You can increment score or just track shots
              if(data.shotMade) {
                if(data.posession == 0) {
                  setTeam1Score(s => s + 2);
                } else {
                  setTeam2Score(s => s + 2);
                }
              }
            }

            if (typeof data.possession === "number") {
              // For example, 1 = Team 1, 2 = Team 2
              console.log("Team in possession:", data.possession);
              // You could highlight possession in UI
            }
          } catch (err) {
            // In case backend sends a plain string instead of JSON
            console.error("Failed to parse WebSocket message:", err);

            const raw = event.data;
            // if (typeof raw === "string" && raw.startsWith("data:image")) {
            //   setAnnotatedFrame(raw);
            // } else {
            //   setAnnotatedFrame(`data:image/jpeg;base64,${raw}`);
            // }
          }
        };


        ws.onopen = () => {
          console.log("✅ WebSocket connected");
          ws.send(JSON.stringify({ numPlayers: 5 }));
        };

        ws.onmessage = (event) => console.log("📩 Server:", event.data);
        ws.onclose = () => console.log("🔌 WebSocket closed");

        captureIntervalRef.current = setInterval(() => {
          if (!paused && videoRef.current && ws.readyState === WebSocket.OPEN) {
            const frameBase64 = captureFrame(videoRef.current);
            ws.send(JSON.stringify({ frame: frameBase64 }));
          }
        }, 200); // 5 fps
      } catch (err) {
        console.error("Camera/WebSocket setup failed:", err);
      }
    };

    initCameraAndWebSocket();

    return () => {
      if (captureIntervalRef.current) clearInterval(captureIntervalRef.current);
      wsRef.current?.send(JSON.stringify({ action: "close" }));
      wsRef.current?.close();
      stream?.getTracks().forEach((t) => t.stop());
    };
  }, [paused]);

  // 🎨 Capture full-size video frame (not affected by CSS scaling)
  const captureFrame = (video: HTMLVideoElement): string => {
    const canvas = document.createElement("canvas");
    const width = video.videoWidth;
    const height = video.videoHeight;
    canvas.width = width;
    canvas.height = height;
    const ctx = canvas.getContext("2d");
    if (!ctx) return "";
    ctx.drawImage(video, 0, 0, width, height);
    return canvas.toDataURL("image/jpeg", 0.1);
  };

  // Format timer
  const formatTime = (seconds: number) => {
    const m = Math.floor(seconds / 60).toString().padStart(2, "0");
    const s = (seconds % 60).toString().padStart(2, "0");
    return `${m}:${s}`;
  };

  return (
    <div className="flex flex-col min-h-screen bg-[#3b3b3b] text-white">
      {/* Top section - stats */}
      <div className="flex flex-col items-center justify-center bg-[#4a4a4a] text-center py-10 shadow-md relative">
        {/* Timer */}
        <div className="absolute top-4 right-6 text-2xl font-bold bg-[#ff5a00] px-4 py-2 rounded-md shadow-md">
          ⏱ {formatTime(timeLeft)}
        </div>

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
      <div className="flex-1 bg-[#2f2f2f] flex items-center justify-center relative">
        <div className="relative w-[640px] h-[360px] border-4 border-[#ff5a00] rounded-lg overflow-hidden shadow-lg">
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
    </div>
  );
}


// "use client";

// import { useState, useRef, useEffect } from "react";
// import { useRouter, useSearchParams } from "next/navigation";

// function timeToSeconds(time: string): number {
//   const [minutes, seconds] = time.split(":").map(Number);
//   if (isNaN(minutes) || isNaN(seconds)) throw new Error("Invalid time format. Use mm:ss (e.g., '1:30').");
//   return minutes * 60 + seconds;
// }

// export default function PlayGame() {
//   const router = useRouter();
//   const searchParams = useSearchParams();

//   const [team1Score, setTeam1Score] = useState(0);
//   const [team2Score, setTeam2Score] = useState(0);
//   const [team1Percentage, setTeam1Percentage] = useState(0);
//   const [team2Percentage, setTeam2Percentage] = useState(0);
//   const [paused, setPaused] = useState(false);
//   const [timeLeft, setTimeLeft] = useState(60 * 2);
//   const [initalTime, setInitalTime] = useState(0);
//   const [annotatedFrame, setAnnotatedFrame] = useState<string | null>(null); // 🟢 annotated frame

//   const videoRef = useRef<HTMLVideoElement | null>(null);
//   const wsRef = useRef<WebSocket | null>(null);
//   const captureIntervalRef = useRef<NodeJS.Timeout | null>(null);
//   const timerRef = useRef<NodeJS.Timeout | null>(null);

//   const togglePause = () => setPaused((prev) => !prev);

//   const endGame = () => {
//     if (confirm("Are you sure you want to end the game?")) {
//       wsRef.current?.send(JSON.stringify({ action: "close" }));
//       wsRef.current?.close();
//       alert("⏹️ Game ended!");
//       router.push(`/stats?timeElapsed=${initalTime - timeLeft}`);
//     }
//   };

//   // 🕒 Initialize time from URL param
//   useEffect(() => {
//     const timeParam = searchParams.get("time");
//     if (timeParam) {
//       setInitalTime(timeToSeconds(timeParam));
//       setTimeLeft(timeToSeconds(timeParam));
//     } else {
//       setTimeLeft(120);
//     }
//   }, []);

//   // ⏱ Countdown timer
//   useEffect(() => {
//     if (paused) {
//       if (timerRef.current) clearInterval(timerRef.current);
//       return;
//     }

//     timerRef.current = setInterval(() => {
//       setTimeLeft((prev) => {
//         if (prev <= 1) {
//           clearInterval(timerRef.current!);
//           endGame();
//           return 0;
//         }
//         return prev - 1;
//       });
//     }, 1000);

//     return () => {
//       if (timerRef.current) clearInterval(timerRef.current);
//     };
//   }, [paused, searchParams]);

//   // 🎥 Camera + WebSocket setup
//   useEffect(() => {
//     let stream: MediaStream;

//     const initCameraAndWebSocket = async () => {
//       try {
//         // 🎦 Start webcam
//         stream = await navigator.mediaDevices.getUserMedia({ video: true });
//         if (videoRef.current) videoRef.current.srcObject = stream;

//         // 🌐 Connect WebSocket
//         const ws = new WebSocket("ws://localhost:8000/ws");
//         wsRef.current = ws;

//         ws.onopen = () => {
//           console.log("✅ WebSocket connected");
//           ws.send(JSON.stringify({ numPlayers: 5 }));
//         };

//         // 📩 Receive annotated frames (base64-encoded)
//         ws.onmessage = (event) => {
//           try {
//             const data = event.data;
//             if (typeof data === "string" && data.startsWith("data:image")) {
//               // Already a dataURL (base64 image)
//               setAnnotatedFrame(data);
//             } else {
//               // If it's just base64, add prefix
//               setAnnotatedFrame(`data:image/jpeg;base64,${data}`);
//             }
//           } catch (err) {
//             console.error("Failed to decode annotated frame:", err);
//           }
//         };

//         ws.onclose = () => console.log("🔌 WebSocket closed");

//         // 🎞 Send frames every 200ms
//         captureIntervalRef.current = setInterval(() => {
//           if (!paused && videoRef.current && ws.readyState === WebSocket.OPEN) {
//             const frameBase64 = captureFrame(videoRef.current);
//             ws.send(JSON.stringify({ frame: frameBase64 }));
//           }
//         }, 200); // 5 fps
//       } catch (err) {
//         console.error("Camera/WebSocket setup failed:", err);
//       }
//     };

//     initCameraAndWebSocket();

//     return () => {
//       if (captureIntervalRef.current) clearInterval(captureIntervalRef.current);
//       wsRef.current?.send(JSON.stringify({ action: "close" }));
//       wsRef.current?.close();
//       stream?.getTracks().forEach((t) => t.stop());
//     };
//   }, [paused]);

//   // 🎨 Capture frame
//   const captureFrame = (video: HTMLVideoElement): string => {
//     const canvas = document.createElement("canvas");
//     const width = video.videoWidth;
//     const height = video.videoHeight;
//     canvas.width = width;
//     canvas.height = height;
//     const ctx = canvas.getContext("2d");
//     if (!ctx) return "";
//     ctx.drawImage(video, 0, 0, width, height);
//     return canvas.toDataURL("image/jpeg", 0.5); // 50% quality
//   };

//   // 🕓 Format timer
//   const formatTime = (seconds: number) => {
//     const m = Math.floor(seconds / 60).toString().padStart(2, "0");
//     const s = (seconds % 60).toString().padStart(2, "0");
//     return `${m}:${s}`;
//   };

//   return (
//     <div className="flex flex-col min-h-screen bg-[#3b3b3b] text-white">
//       {/* Top section - stats */}
//       <div className="flex flex-col items-center justify-center bg-[#4a4a4a] text-center py-10 shadow-md relative">
//         <div className="absolute top-4 right-6 text-2xl font-bold bg-[#ff5a00] px-4 py-2 rounded-md shadow-md">
//           ⏱ {formatTime(timeLeft)}
//         </div>

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
//       <div className="flex-1 bg-[#2f2f2f] flex items-center justify-center relative">
//         <div className="relative w-[640px] h-[360px] border-4 border-[#ff5a00] rounded-lg overflow-hidden shadow-lg">
//           {/* Show annotated frame if available */}
//           {annotatedFrame ? (
//             <img
//               src={annotatedFrame}
//               alt="Annotated Frame"
//               className="w-full h-full object-cover"
//             />
//           ) : (
//             <video
//               ref={videoRef}
//               className="w-full h-full object-cover"
//               autoPlay
//               muted
//               playsInline
//             />
//           )}

//           {paused && (
//             <div className="absolute inset-0 bg-black/70 flex items-center justify-center">
//               <span className="text-4xl font-bold text-white">Paused</span>
//             </div>
//           )}
//         </div>
//       </div>
//     </div>
//   );
// }
