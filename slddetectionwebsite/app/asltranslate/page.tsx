"use client";
import React from "react";
import MediaRecorderComponent from "@/components/MediaRecorderComponent";

const AsltranslatePage = () => {
  return (
    <div className="grid grid-cols-8 h-screen">
      <div className="flex justify-center col-start-2 gap-6 col-span-6 h-full w-full mt-4">
        <div className="h-full w-fit">
          <MediaRecorderComponent />
        </div>
      </div>
    </div>
  );
};
export default AsltranslatePage;
