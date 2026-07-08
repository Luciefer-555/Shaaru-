import { useState, useRef, useCallback } from "react";

export interface VoiceSTTResult {
  transcribed_text?: string;
  reply?: string;
  model?: string;
}

export interface UseVoiceInputOptions {
  onResult?: (result: VoiceSTTResult) => void;
  onError?: (err: any) => void;
}

export function useVoiceInput(options?: UseVoiceInputOptions) {
  const [isRecording, setIsRecording] = useState(false);
  const [isTranscribing, setIsTranscribing] = useState(false);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioChunksRef = useRef<Blob[]>([]);

  const startRecording = useCallback(async () => {
    if (isRecording || isTranscribing) return;
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      audioChunksRef.current = [];
      const mediaRecorder = new MediaRecorder(stream);
      
      mediaRecorder.ondataavailable = (event) => {
        if (event.data && event.data.size > 0) {
          audioChunksRef.current.push(event.data);
        }
      };

      mediaRecorderRef.current = mediaRecorder;
      mediaRecorder.start();
      setIsRecording(true);
    } catch (err) {
      console.error("[useVoiceInput] Mic access error:", err);
      options?.onError?.(err);
    }
  }, [isRecording, isTranscribing, options]);

  const stopRecordingAndSend = useCallback(async (scanContext?: string, userId: string = "user") => {
    if (!mediaRecorderRef.current || !isRecording) return;
    const mediaRecorder = mediaRecorderRef.current;

    return new Promise<void>((resolve) => {
      mediaRecorder.onstop = async () => {
        mediaRecorder.stream.getTracks().forEach((track) => track.stop());
        const audioBlob = new Blob(audioChunksRef.current, { type: "audio/webm" });
        setIsRecording(false);
        setIsTranscribing(true);

        try {
          const formData = new FormData();
          formData.append("file", audioBlob, "recording.webm");
          formData.append("user_id", userId);
          formData.append("enable_tts", "false"); // No TTS needed for live camera text overlay!
          if (scanContext) {
            formData.append("scan_context", scanContext);
          }

          const apiUrl = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
          const res = await fetch(`${apiUrl}/api/voice/stt`, {
            method: "POST",
            body: formData,
          });

          if (res.ok) {
            const data = await res.json();
            // Discard audio_base64 if any, pass only text results
            options?.onResult?.({
              transcribed_text: data.transcribed_text,
              reply: data.reply,
              model: data.model,
            });
          } else {
            console.error("[useVoiceInput] STT endpoint error:", res.status);
            options?.onError?.(new Error(`STT error: ${res.status}`));
          }
        } catch (err) {
          console.error("[useVoiceInput] STT request failed:", err);
          options?.onError?.(err);
        } finally {
          setIsTranscribing(false);
          resolve();
        }
      };

      mediaRecorder.stop();
    });
  }, [isRecording, options]);

  const toggleRecording = useCallback((scanContext?: string, userId?: string) => {
    if (isRecording) {
      stopRecordingAndSend(scanContext, userId);
    } else {
      startRecording();
    }
  }, [isRecording, startRecording, stopRecordingAndSend]);

  return {
    isRecording,
    isTranscribing,
    startRecording,
    stopRecordingAndSend,
    toggleRecording,
  };
}
