"use client";
import React from "react";
import MediaRecorderComponent from "@/components/MediaRecorderComponent";

export default function Home() {
  return (
    <main className="w-full h-screen grid grid-cols-8">
      <div className="flex justify-center col-start-2 gap-6 col-span-6 h-full w-full mt-4">
        <div className="h-full w-fit">
          <MediaRecorderComponent />
        </div>
      </div>
    </main>
  );
}
