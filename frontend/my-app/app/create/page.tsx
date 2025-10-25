"use client";

import { useRouter } from 'next/navigation'; // For App Router
import { useState } from "react";

const generateTimeOptions = () => {
  const times: string[] = [];
  for (let totalSeconds = 30; totalSeconds <= 20 * 60; totalSeconds += 30) {
    const minutes = Math.floor(totalSeconds / 60);
    const seconds = totalSeconds % 60;
    const formatted = `${minutes}:${seconds.toString().padStart(2, "0")}`;
    times.push(formatted);
  }
  return times;
};

export default function CreateGame() {
  const router = useRouter();

  const pointOptions = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21];
  const timeOptions = generateTimeOptions();
  const gameOptions = ["1v1", "2v2", "3v3", "4v4", "5v5"];

  const [points, setPoints] = useState(10);
  const [time, setTime] = useState("1:30");
  const [game, setGame] = useState("1v1");

  // helpers for cycling through arrays
  const cycle = (arr: any[], current: any, dir: "next" | "prev") => {
    const idx = arr.indexOf(current);
    if (dir === "next") return arr[(idx + 1) % arr.length];
    if (dir === "prev") return arr[(idx - 1 + arr.length) % arr.length];
  };

  const handlePlay = () => {
    alert(`Starting ${game} game to ${points} points, ${time} limit`);
    // You can route to your game page here using next/navigation:
    router.push(`/play?points=${points}&time=${time}&type=${game}`);
  };

  return (
    <div className="relative flex items-center justify-center min-h-screen text-center">
      {/* Background */}
      <div className="absolute inset-0">
        <img
          src="/court-bg.jpg"
          alt="Basketball court"
          className="object-cover w-full h-full"
        />
        <div className="absolute inset-0 bg-black/70" />
      </div>

      {/* Card */}
      <div className="relative z-10 bg-white/10 backdrop-blur-md p-10 rounded-xl shadow-lg text-white w-[90%] max-w-md">
        <h1 className="text-3xl font-bold mb-8">Create Game</h1>

        <div className="flex flex-col gap-6 text-lg font-semibold">
          {/* Points */}
          <div className="flex items-center justify-between">
            <span className="w-24 text-left">Points</span>
            <div className="flex items-center gap-2">
              <button
                onClick={() => setPoints(cycle(pointOptions, points, "prev"))}
                className="bg-orange-500 hover:bg-orange-600 px-3 py-1 rounded text-white text-xl"
              >
                ◀
              </button>
              <div className="px-6 py-1 bg-orange-500 rounded-md text-lg">
                {points}
              </div>
              <button
                onClick={() => setPoints(cycle(pointOptions, points, "next"))}
                className="bg-orange-500 hover:bg-orange-600 px-3 py-1 rounded text-white text-xl"
              >
                ▶
              </button>
            </div>
          </div>

          {/* Time */}
          <div className="flex items-center justify-between">
            <span className="w-24 text-left">Time</span>
            <div className="flex items-center gap-2">
              <button
                onClick={() => setTime(cycle(timeOptions, time, "prev"))}
                className="bg-orange-500 hover:bg-orange-600 px-3 py-1 rounded text-white text-xl"
              >
                ◀
              </button>
              <div className="px-6 py-1 bg-orange-500 rounded-md text-lg">
                {time}
              </div>
              <button
                onClick={() => setTime(cycle(timeOptions, time, "next"))}
                className="bg-orange-500 hover:bg-orange-600 px-3 py-1 rounded text-white text-xl"
              >
                ▶
              </button>
            </div>
          </div>

          {/* Game Type */}
          <div className="flex items-center justify-between">
            <span className="w-24 text-left">Game</span>
            <div className="flex items-center gap-2">
              <button
                onClick={() => setGame(cycle(gameOptions, game, "prev"))}
                className="bg-orange-500 hover:bg-orange-600 px-3 py-1 rounded text-white text-xl"
              >
                ◀
              </button>
              <div className="px-6 py-1 bg-orange-500 rounded-md text-lg">
                {game}
              </div>
              <button
                onClick={() => setGame(cycle(gameOptions, game, "next"))}
                className="bg-orange-500 hover:bg-orange-600 px-3 py-1 rounded text-white text-xl"
              >
                ▶
              </button>
            </div>
          </div>
        </div>

        {/* Play Button */}
        <button
          onClick={handlePlay}
          className="mt-10 w-full bg-orange-600 hover:bg-orange-700 text-white text-lg font-semibold py-3 rounded-md shadow-lg transition-transform transform hover:scale-105"
        >
          Play
        </button>
      </div>
    </div>
  );
}
