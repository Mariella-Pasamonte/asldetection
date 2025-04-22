import React, { useEffect, useRef, useState } from "react";
import axios from "axios";
import {
  FilesetResolver,
  GestureRecognizer,
  GestureRecognizerResult,
  DrawingUtils,
} from "@mediapipe/tasks-vision";

const MediaRecorderComponent: React.FC = () => {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const drawIntervalRef = useRef<NodeJS.Timeout | null>(null);
  const predictIntervalRef = useRef<NodeJS.Timeout | null>(null);
  let gestureRecognizer: GestureRecognizer;
  const [ASLDetecting, setASLDetecting] = useState(false);
  const [isVideoReady, setIsVideoReady] = useState(false);
  const [activeStream, setActiveStream] = useState<MediaStream | null>(null);
  const [subtitle, setSubtitle] = useState("");
  const [errorMessage, setErrorMessage] = useState("");
  const subtitleRef = useRef(subtitle);
  const cancelPredictRef = useRef<(() => void) | null>(null);

  const startWebcam = async (): Promise<void> => {
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
        `Camera not available. ${err}. Please check your permissions or device.`
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
      setErrorMessage(`Screen sharing failed. ${err}.`);
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

  const startASLDetection = (): void => {
    const stream = videoRef.current?.srcObject as MediaStream;
    if (!stream) return;

    setASLDetecting(true);

    const { drawInterval, predictInterval, cancel } = startFrameCapture();
    drawIntervalRef.current = drawInterval;
    predictIntervalRef.current = predictInterval;
    cancelPredictRef.current = cancel;
  };

  const stopASLDetection = (): void => {
    if (drawIntervalRef.current) {
      clearInterval(drawIntervalRef.current);
      drawIntervalRef.current = null;
    }

    if (predictIntervalRef.current) {
      clearTimeout(predictIntervalRef.current);
      predictIntervalRef.current = null;
    }

    if (cancelPredictRef.current) {
      cancelPredictRef.current(); // Stops recursive prediction loop
      cancelPredictRef.current = null;
    }

    setASLDetecting(false);
    setSubtitle("");
  };

  const createGestureRecognizer = async () => {
    const vision = await FilesetResolver.forVisionTasks(
      "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3/wasm"
    );
    gestureRecognizer = await GestureRecognizer.createFromOptions(vision, {
      baseOptions: {
        modelAssetPath:
          "https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/1/gesture_recognizer.task",
        delegate: "GPU",
      },
      runningMode: "IMAGE",
      minHandDetectionConfidence: 0.1,
      minTrackingConfidence: 0.1,
      numHands: 2,
    });
  };
  createGestureRecognizer();

  const startFrameCapture = () => {
    let results: GestureRecognizerResult;
    let predictTimeout: NodeJS.Timeout | null = null;
    let isCancelled = false;

    const drawInterval = setInterval(() => {
      if (!videoRef.current || !canvasRef.current) return;

      results = gestureRecognizer.recognize(videoRef.current);

      const ctx = canvasRef.current.getContext("2d");
      if (!ctx) return;

      canvasRef.current.width = videoRef.current.videoWidth;
      canvasRef.current.height = videoRef.current.videoHeight;

      ctx.drawImage(videoRef.current, 0, 0);

      const drawingUtils = new DrawingUtils(ctx);
      if (results.landmarks) {
        for (const landmarks of results.landmarks) {
          drawingUtils.drawConnectors(
            landmarks,
            GestureRecognizer.HAND_CONNECTIONS,
            {
              color: "#00FF00",
              lineWidth: 5,
            }
          );
          drawingUtils.drawLandmarks(landmarks, {
            color: "#FF0000",
            lineWidth: 2,
          });
        }
      }
    }, 100);

    const predictLoop = async () => {
      if (isCancelled) return;

      if (!results || !results.landmarks || results.landmarks.length === 0) {
        predictTimeout = setTimeout(predictLoop, 1000); // Try again later
        return;
      }

      const landmarkData: number[] = [];
      for (const landmarks of results.landmarks) {
        for (const landmark of landmarks) {
          landmarkData.push(landmark.x);
          landmarkData.push(landmark.y);
          landmarkData.push(landmark.z);
        }
      }

      try {
        const response = await axios.post(
          "https://asldetection.onrender.com/predict",
          landmarkData
        );
        const label = response.data.result;
        setErrorMessage("");
        setSubtitle(label);
        speakText(label);
      } catch (err) {
        console.log(err);
      } finally {
        // Wait 4 seconds before running next prediction
        if (!isCancelled) {
          predictTimeout = setTimeout(predictLoop, 3000);
        }
      }
    };

    predictLoop();

    return {
      drawInterval,
      predictInterval: predictTimeout,
      cancel: () => {
        isCancelled = true;
        if (predictTimeout) clearTimeout(predictTimeout);
      },
    };
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
          <div className="absolute bottom-20 w-full text-center z-20">
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
        <canvas
          ref={canvasRef}
          className={`absolute top-0 left-0 z-10 ${
            ASLDetecting ? "block" : "hidden"
          }`}
          style={{
            width: "100%",
            height: "100%",
            pointerEvents: "none",
          }}
        />
        <label className="absolute bottom-2 left-2 inline-flex items-center cursor-pointer z-20">
          <input
            type="checkbox"
            onChange={(e) => {
              const checked = e.target.checked;
              if (checked) {
                startASLDetection();
              } else {
                stopASLDetection();
              }
            }}
            disabled={isVideoReady == false}
            className="sr-only peer"
          />
          <div
            className={`relative w-11 h-6 ${
              isVideoReady ? "bg-gray-200" : "bg-gray-800"
            } peer-focus:outline-none peer-focus:ring-4 peer-focus:ring-blue-300 dark:peer-focus:ring-blue-800 rounded-full peer dark:bg-gray-700 peer-checked:after:translate-x-full rtl:peer-checked:after:-translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:start-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all dark:border-gray-600 peer-checked:bg-blue-600 dark:peer-checked:bg-blue-600`}
          ></div>
          <span
            className={`ms-3 text-sm font-medium ${
              isVideoReady ? "text-black" : "text-white"
            } dark:text-gray-300`}
          >
            {ASLDetecting ? "Hand Detection is on" : "Hand Detection is off"}
          </span>
        </label>
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
