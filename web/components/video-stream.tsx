"use client";
import React, { useEffect, useRef } from "react";
import io from "socket.io-client";

export function VideoStream() {
  const videoRef = useRef<HTMLImageElement | null>(null);

  useEffect(() => {
    const socket = io("http://<AWS_SERVER_IP>:5000"); // Connect to the Flask WebSocket server

    socket.on("video-frame", (data) => {
      if (videoRef.current) {
        // Convert base64 string back to a Blob URL for display
        const imgBlob = `data:image/jpeg;base64,${data}`;
        videoRef.current.src = imgBlob;
      }
    });

    return () => {
      socket.disconnect();
    };
  }, []);

  return (
    <div>
      <h1>Live Object Detection Stream</h1>
      <img
        ref={videoRef}
        alt="Processed Stream"
        style={{ width: "100%", maxHeight: "500px" }}
      />
    </div>
  );
}
