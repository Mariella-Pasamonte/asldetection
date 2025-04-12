import React, { useEffect, useRef, useState } from "react";
import Image from "next/image";
import axios from "axios";

const MediaRecorderComponent: React.FC = () => {
  const videoRef = useRef<HTMLVideoElement>(null);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const frameIntervalRef = useRef<NodeJS.Timeout | null>(null);
  const [chunks, setChunks] = useState<Blob[]>([]);
  const [recording, setRecording] = useState(false);
  const [isVideoReady, setIsVideoReady] = useState(false);
  const [activeStream, setActiveStream] = useState<MediaStream | null>(null);
  const [subtitle, setSubtitle] = useState("");
  const [errorMessage, setErrorMessage] = useState("");
  const subtitleRef = useRef(subtitle);

  const startWebcam = async () => {
    try {
      stopCurrentStream();
      const stream = await navigator.mediaDevices.getUserMedia({
        video: true,
        audio: false,
      });
      attachStreamToVideo(stream);
    } catch (err) {
      console.error("Error starting webcam:", err);
      setErrorMessage(
        "Camera not available. Please check your permissions or device."
      );
    }
  };

  const startScreenShare = async () => {
    try {
      stopCurrentStream();
      const stream = await navigator.mediaDevices.getDisplayMedia({
        video: true,
        audio: false,
      });
      attachStreamToVideo(stream);
    } catch (err) {
      console.error("Error starting screen share:", err);
      setErrorMessage("Screen sharing failed.");
    }
  };

  const stopCurrentStream = () => {
    activeStream?.getTracks().forEach((track) => track.stop());
    setActiveStream(null);
    setErrorMessage("");
  };

  const attachStreamToVideo = (stream: MediaStream) => {
    if (videoRef.current) {
      videoRef.current.srcObject = stream;
      videoRef.current.play(); // <-- Force play
      setActiveStream(stream);
      setIsVideoReady(true);
      setErrorMessage("");
    }
  };

  const startRecording = () => {
    const stream = videoRef.current?.srcObject as MediaStream;
    if (!stream) return;

    // const recorder = new MediaRecorder(stream);
    // mediaRecorderRef.current = recorder;
    // setChunks([]);

    // recorder.ondataavailable = (e) => {
    //   if (e.data.size > 0) {
    //     setChunks((prev) => [...prev, e.data]);
    //   }
    // };

    // recorder.onstop = () => {
    //   const blob = new Blob(chunks, { type: "video/webm" });
    //   const url = URL.createObjectURL(blob);
    //   const a = document.createElement("a");
    //   a.href = url;
    //   a.download = "recording.webm";
    //   a.click();
    //   URL.revokeObjectURL(url);
    // };

    // recorder.start();
    setRecording(true);

    const intervalId = startFrameCapture();
    frameIntervalRef.current = intervalId; // store to stop later
  };

  const stopRecording = () => {
    if (frameIntervalRef.current) {
      stopFrameCapture(frameIntervalRef.current);
      frameIntervalRef.current = null;
    }
    // mediaRecorderRef.current?.stop();
    setRecording(false);
  };

  const startFrameCapture = () => {
    let count = 0;

    const interval = setInterval(async () => {
      if (!videoRef.current || !canvasRef.current) return;

      const ctx = canvasRef.current.getContext("2d");
      if (!ctx) return;

      canvasRef.current.width = videoRef.current.videoWidth;
      canvasRef.current.height = videoRef.current.videoHeight;

      ctx.drawImage(videoRef.current, 0, 0);

      const dataUrl = canvasRef.current.toDataURL("image/png");
      const blob = await (await fetch(dataUrl)).blob(); // convert base64 to blob
      const formData = new FormData();
      formData.append("file", blob, "frame.png");

      try {
        const response = await axios.post(
          "http://192.168.1.11:8000/predict",
          formData
        );
        const label = response.data.result;
        console.log("Prediction:", label);
        setErrorMessage("");
        setSubtitle(label); // update current subtitle
        speakText(label);
      } catch (err) {}

      count++;
    }, 5000); // every 1 second or 30th frame

    return interval;
  };

  const stopFrameCapture = (intervalId: NodeJS.Timeout) => {
    clearInterval(intervalId);
  };

  const speakText = (text: string) => {
    if ("speechSynthesis" in window) {
      // Cancel any ongoing speech
      if (window.speechSynthesis.speaking) {
        window.speechSynthesis.cancel();
      }

      const utterance = new SpeechSynthesisUtterance(text);
      utterance.lang = "en-US";
      utterance.rate = 1; // Speed (0.1 - 10)
      utterance.pitch = 1; // Pitch (0 - 2)
      window.speechSynthesis.speak(utterance);
    } else {
      console.warn("Text-to-Speech not supported in this browser.");
    }
  };

  useEffect(() => {
    startWebcam();
    return () => {
      stopCurrentStream();
    };
  }, []);

  useEffect(() => {
    subtitleRef.current = subtitle;
  }, [subtitle]);

  return (
    <div className="h-full w-full">
      <div className="relative w-fit h-[500px]">
        {!isVideoReady && (
          <div className="w-[700px] h-full bg-[#222222] text-white p-2 rounded-lg">
            {"No video available"}
          </div>
        )}
        <video
          ref={videoRef}
          autoPlay
          muted
          className={`rounded-lg h-full ${isVideoReady ? "block" : "hidden "}`}
        />
        {subtitle && (
          <div className="absolute bottom-20 w-full text-center">
            <span className="text-white text-2xl font-bold bg-black bg-opacity-50 px-4 py-1 rounded">
              {subtitle}
            </span>
          </div>
        )}
        {errorMessage && (
          <div className="absolute bottom-20 left-1/2 transform -translate-x-1/2 bg-red-500 text-white px-4 py-2 rounded">
            {errorMessage}
          </div>
        )}

        <canvas ref={canvasRef} style={{ display: "none" }} />
        <button
          onClick={recording ? stopRecording : startRecording}
          disabled={isVideoReady == false}
          className={`absolute bottom-4 left-1/2 transform -translate-x-1/2`}
        >
          {recording ? (
            <Image
              src="/stop record.png"
              alt="Stop Recording"
              width={50}
              height={50}
            />
          ) : (
            <Image
              src="/record.png"
              alt="Start Recording"
              width={50}
              height={50}
            />
          )}
        </button>
      </div>
      <div className="grid grid-cols-4 mt-4 space-x-2 content-center place-items-center">
        <button
          onClick={startWebcam}
          className="col-start-2 bg-blue-500 text-white px-4 py-2 rounded"
        >
          Webcam
        </button>
        <button
          onClick={startScreenShare}
          className="bg-green-600 text-white px-4 py-2 rounded"
        >
          Screen Share
        </button>
      </div>
    </div>
  );
};

export default MediaRecorderComponent;
