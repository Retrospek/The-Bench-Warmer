"use client";

import Link from "next/link";

export default function Home() {
  return (
    <div className="relative flex items-center justify-center min-h-screen text-center">
      {/* Background Video or Image */}
      <div className="absolute inset-0 overflow-hidden">
        {/* For video background */}
        {/* <video
          autoPlay
          loop
          muted
          playsInline
          className="object-cover w-full h-full"
        >
          <source src="/court-bg.mp4" type="video/mp4" />
        </video> */}

        {/* For image background */}
        {/* <img
          src="/court-bg.jpg"
          alt="Basketball court background"
          className="object-cover w-full h-full"
        /> */}

        {/* Overlay for contrast */}
        <div className="absolute inset-0 bg-black/60" />
      </div>

      {/* Main Content */}
      <div className="relative z-10 px-6">
        <h1 className="text-4xl sm:text-5xl font-extrabold text-white drop-shadow-lg mb-6">
          The Bench Warmer
        </h1>

        <Link href="/create" className="px-8 py-3 bg-orange-500 hover:bg-orange-600 text-white text-lg font-semibold rounded-md shadow-lg transition-transform transform hover:scale-105">
          Get Started
        </Link>
      </div>
    </div>
  );
}
