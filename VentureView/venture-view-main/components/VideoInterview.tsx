import React, { useState, useRef, useEffect, useCallback } from 'react';
import { VIDEO_QUESTIONS } from '../constants';
import { VideoAnalysisResult } from '../types';
import { analyzeVideo } from '../services/mockAnalysisService';
import CameraIcon from './icons/CameraIcon';
import RecordingIcon from './icons/RecordingIcon';

interface VideoInterviewProps {
  onComplete: (results: VideoAnalysisResult[]) => void;
}

type PermissionStatus = 'idle' | 'pending' | 'granted' | 'denied';
type InterviewStatus = 'setup' | 'permission_denied' | 'ready' | 'recording' | 'analyzing' | 'finished';

const VideoInterview: React.FC<VideoInterviewProps> = ({ onComplete }) => {
  const [permissionStatus, setPermissionStatus] = useState<PermissionStatus>('idle');
  const [interviewStatus, setInterviewStatus] = useState<InterviewStatus>('setup');
  // This round now only handles the first question.
  const [currentQuestionIndex] = useState(0); 
  const [timeLeft, setTimeLeft] = useState(180); // 3 minutes per question
  
  const videoRef = useRef<HTMLVideoElement>(null);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const recordedChunksRef = useRef<Blob[]>([]);
  const streamRef = useRef<MediaStream | null>(null);

  const requestPermissions = useCallback(async () => {
    setPermissionStatus('pending');
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: true });
      streamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
      }
      setPermissionStatus('granted');
      setInterviewStatus('ready');
    } catch (err) {
      console.error("Error accessing media devices.", err);
      setPermissionStatus('denied');
      setInterviewStatus('permission_denied');
    }
  }, []);

  const stopRecording = useCallback(() => {
    if (mediaRecorderRef.current && mediaRecorderRef.current.state === 'recording') {
        mediaRecorderRef.current.stop();
    }
  }, []);

  const beginRecordingForCurrentQuestion = useCallback(() => {
    if (streamRef.current) {
      recordedChunksRef.current = [];
      const recorder = new MediaRecorder(streamRef.current, { mimeType: 'video/webm' });
      mediaRecorderRef.current = recorder;
      
      recorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          recordedChunksRef.current.push(event.data);
        }
      };

      recorder.onstop = async () => {
        const blob = new Blob(recordedChunksRef.current, { type: 'video/webm' });
        
        setInterviewStatus('analyzing');
        const currentQuestion = VIDEO_QUESTIONS[currentQuestionIndex];
        const result = await analyzeVideo(blob, currentQuestion);

        // After analyzing the first (and only) question for this round,
        // call onComplete to proceed to the next round (HR interview).
        onComplete([result]);
      };
      
      try {
        recorder.start();
        setInterviewStatus('recording');
        setTimeLeft(180);
      } catch (e) {
        console.error("Failed to start MediaRecorder:", e);
        setInterviewStatus('permission_denied');
      }
    }
  }, [currentQuestionIndex, onComplete]);
  
  const handleStartInterview = useCallback(() => {
    beginRecordingForCurrentQuestion();
  }, [beginRecordingForCurrentQuestion]);

  const handleFinishQuestion = useCallback(() => {
    stopRecording();
  }, [stopRecording]);

  useEffect(() => {
    if (interviewStatus === 'recording' && timeLeft > 0) {
      const timer = setTimeout(() => setTimeLeft(prev => prev - 1), 1000);
      return () => clearTimeout(timer);
    } else if (interviewStatus === 'recording' && timeLeft === 0) {
      handleFinishQuestion();
    }
  }, [timeLeft, interviewStatus, handleFinishQuestion]);

  const cleanup = useCallback(() => {
    stopRecording();
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => track.stop());
    }
  }, [stopRecording]);

  useEffect(() => {
    // Cleanup on unmount
    return () => {
      cleanup();
    };
  }, [cleanup]);

  const renderSetup = () => (
    <div className="text-center p-8 bg-gray-800 rounded-lg shadow-xl max-w-2xl mx-auto">
      <CameraIcon className="w-16 h-16 mx-auto mb-4 text-cyan-400" />
      <h2 className="text-3xl font-bold text-white mb-2">Technical Video Interview</h2>
      <p className="text-gray-300 mb-6">This is the second round of your interview. You will be presented with 1 technical question. You will have 3 minutes to answer it.</p>
      <button 
        onClick={requestPermissions}
        disabled={permissionStatus === 'pending'}
        className="px-6 py-3 text-lg font-semibold text-white bg-cyan-600 rounded-md hover:bg-cyan-500 disabled:bg-cyan-800 transition-colors"
      >
        {permissionStatus === 'pending' ? 'Requesting...' : 'Setup Camera & Microphone'}
      </button>
    </div>
  );

  const renderPermissionDenied = () => (
    <div className="text-center p-8 bg-red-900/50 border border-red-700 rounded-lg shadow-xl max-w-2xl mx-auto">
      <h2 className="text-3xl font-bold text-red-300 mb-2">Permissions Denied</h2>
      <p className="text-red-200">Camera and microphone access is required for the video interview. Please enable permissions in your browser settings and refresh the page.</p>
    </div>
  );
  
  const renderReady = () => (
    <div className="text-center p-8 bg-gray-800 rounded-lg shadow-xl max-w-2xl mx-auto flex flex-col items-center">
        <h2 className="text-2xl font-bold text-white mb-4">You're All Set!</h2>
        <div className="w-full max-w-md bg-black rounded-lg overflow-hidden mb-4 border-2 border-gray-700">
            <video ref={videoRef} autoPlay muted playsInline className="w-full h-auto"></video>
        </div>
        <p className="text-gray-400 mb-6">Click the button below to start the interview when you're ready.</p>
        <button 
            onClick={handleStartInterview}
            className="px-8 py-3 text-lg font-semibold text-white bg-green-600 rounded-md hover:bg-green-500 transition-colors"
        >
            Start Interview
        </button>
    </div>
  );
  
  const renderAnalyzing = () => (
      <div className="text-center p-8 bg-gray-800 rounded-lg shadow-xl max-w-2xl mx-auto">
        <h2 className="text-3xl font-bold text-white mb-4">Analyzing your answer...</h2>
        <div className="flex items-center justify-center space-x-2 text-cyan-400">
            <div className="w-8 h-8 border-4 border-t-transparent border-cyan-400 rounded-full animate-spin"></div>
            <span className="text-lg">Please wait while we process your video.</span>
      </div>
    </div>
  );


  const renderRecording = () => {
    const question = VIDEO_QUESTIONS[currentQuestionIndex];
    const mins = String(Math.floor(timeLeft / 60)).padStart(2, '0');
    const secs = String(timeLeft % 60).padStart(2, '0');
    return (
        <div className="w-full h-full flex flex-col p-4 gap-4">
            <div className="w-full bg-gray-800 rounded-lg p-4 border border-gray-700">
                <p className="text-sm text-cyan-400 font-semibold">{question.category}</p>
                <h3 className="text-xl font-bold text-white mt-1">{question.question}</h3>
            </div>
            <div className="flex-1 flex flex-col md:flex-row gap-4 overflow-hidden">
                <div className="flex-1 bg-black rounded-lg overflow-hidden relative border-2 border-cyan-500">
                    <video ref={videoRef} autoPlay muted playsInline className="w-full h-full object-cover"></video>
                    <div className="absolute top-3 left-3 flex items-center space-x-2 bg-red-600/90 text-white px-3 py-1 rounded-md text-sm font-semibold">
                       <RecordingIcon className="w-4 h-4" />
                       <span>REC</span>
                    </div>
                </div>
                <div className="w-full md:w-64 flex flex-col gap-4">
                    <div className="bg-gray-800 p-4 rounded-lg flex-1 flex flex-col items-center justify-center text-center">
                        <p className="text-gray-400 text-sm">Technical Question</p>
                        <p className={`font-mono text-5xl font-bold mt-2 ${timeLeft < 30 ? 'text-red-500' : 'text-white'}`}>{mins}:{secs}</p>
                        <p className="text-gray-500 text-xs mt-1">Time Remaining</p>
                    </div>
                    <button 
                        onClick={handleFinishQuestion}
                        className="w-full py-3 text-base font-semibold text-white bg-cyan-600 rounded-md hover:bg-cyan-500 transition-colors"
                    >
                        Finish & Proceed to HR Round
                    </button>
                </div>
            </div>
        </div>
    )
  };

  const renderContent = () => {
    switch(interviewStatus) {
      case 'setup': return renderSetup();
      case 'permission_denied': return renderPermissionDenied();
      case 'ready': return renderReady();
      case 'recording': return renderRecording();
      case 'analyzing': return renderAnalyzing();
      // The finished state is no longer used here as onComplete transitions out of this component
      default: return renderSetup();
    }
  };

  return (
    <div className="flex flex-col h-screen font-sans bg-gray-900 text-gray-200">
      <header className="flex items-center justify-between p-3 bg-gray-800 border-b border-gray-700 shadow-md">
        <h1 className="text-xl font-bold text-cyan-400">AI-Powered Interview Platform</h1>
        <div className="flex items-center space-x-4">
          <span className="text-sm font-semibold px-3 py-1 bg-cyan-900/50 text-cyan-300 rounded-full">Round 2 of 3: Technical Interview</span>
        </div>
      </header>
      <main className="flex-1 flex items-center justify-center p-4">
        {renderContent()}
      </main>
    </div>
  );
};

export default VideoInterview;