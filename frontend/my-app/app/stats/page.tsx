"use client"

import Heatmap from "@/components/HeatMap"
import Link from "next/link"; 

import { useSearchParams } from "next/navigation";
import {useEffect, useState} from "react";

function secondsToTime(totalSeconds: number): string {
  const minutes = Math.floor(totalSeconds / 60);
  const seconds = totalSeconds % 60;

  // Pad with leading zero if needed (e.g. "1:05" instead of "1:5")
  const formattedSeconds = seconds.toString().padStart(2, "0");

  return `${minutes}:${formattedSeconds}`;
}

export default function GameStats() {
  const searchParams = useSearchParams(); 

  const [timeElapsed, setTimeElapsed] = useState("");
  useEffect(() => {
    let time = searchParams.get("timeElapsed");
    if(time) {
      setTimeElapsed(secondsToTime(parseInt(time)));
    }
  }, [])

  return (
    <div className="flex flex-col items-center justify-center min-h-screen bg-[#2e2e2e] text-white">
      <div className="bg-[#bfbfbf] text-white rounded-lg p-8 w-[600px]">
        {/* Header */}
        <div className="flex justify-between mb-6 text-center">
          <div className="flex-1">
            <h2 className="text-xl font-semibold">Team 1</h2>
            <p className="text-2xl font-bold">0</p>
            <p className="text-sm">Shot Percentage 0%</p>
          </div>
          <div className="w-px bg-white mx-4" />
          <div className="flex-1">
            <h2 className="text-xl font-semibold">Team 2</h2>
            <p className="text-2xl font-bold">0</p>
            <p className="text-sm">Shot Percentage 0%</p>
          </div>
        </div>

        {/* Divider */}
        <hr className="border-white mb-6" />

        {/* Timer and Heatmap */}
        <div className="flex justify-between items-start mt-8">
          {/* Left Section */}
          <div className="flex flex-col justify-between h-full">
            <div>
              <p className="text-lg font-semibold mb-1">Time Elapsed</p>
              <p className="text-3xl font-bold">{timeElapsed}</p>
            </div>

            {/* Separate Play Again button below */}
            <div className="mt-10">
              <Link
                href="/create"
                className="inline-block px-8 py-3 bg-orange-500 hover:bg-orange-600 transition-colors rounded-md font-semibold text-white shadow-md"
              >
                Play Again
              </Link>
            </div>
          </div>

          {/* Heatmap Section */}
          <div className="w-1/2 ml-8">
            <div className="rounded-lg overflow-hidden shadow-lg">
              <Heatmap />
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
