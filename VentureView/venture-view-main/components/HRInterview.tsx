import React, { useState, useRef, useEffect, useCallback } from 'react';
import { HR_QUESTION } from '../constants';
import CameraIcon from './icons/CameraIcon';
import RecordingIcon from './icons/RecordingIcon';

interface HRInterviewProps {
  onComplete: () => void;
}

type PermissionStatus = 'idle' | 'pending' | 'granted' | 'denied';
type InterviewStatus = 'setup' | 'permission_denied' | 'ready' | 'recording' | 'analyzing' | 'finished';

const HRInterview: React.FC<HRInterviewProps> = ({ onComplete }) => {
  const [permissionStatus, setPermissionStatus] = useState<PermissionStatus>('idle');
  const [interviewStatus, setInterviewStatus] = useState<InterviewStatus>('setup');
  const [timeLeft, setTimeLeft] = useState(300); // 5 minutes per question
  
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

  const beginRecording = useCallback(() => {
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
        // const blob = new Blob(recordedChunksRef.current, { type: 'video/webm' });
        // Analysis is now done on the results page.
        setInterviewStatus('finished');
        // Just signal that the recording is complete.
        onComplete();
      };
      
      try {
        recorder.start();
        setInterviewStatus('recording');
        setTimeLeft(300);
      } catch (e) {
        console.error("Failed to start MediaRecorder:", e);
        setInterviewStatus('permission_denied');
      }
    }
  }, [onComplete]);
  
  const handleFinishInterview = useCallback(() => {
    stopRecording();
  }, [stopRecording]);

  useEffect(() => {
    if (interviewStatus === 'recording' && timeLeft > 0) {
      const timer = setTimeout(() => setTimeLeft(prev => prev - 1), 1000);
      return () => clearTimeout(timer);
    } else if (interviewStatus === 'recording' && timeLeft === 0) {
      handleFinishInterview();
    }
  }, [timeLeft, interviewStatus, handleFinishInterview]);

  const cleanup = useCallback(() => {
    stopRecording();
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => track.stop());
    }
  }, [stopRecording]);

  useEffect(() => {
    return () => cleanup();
  }, [cleanup]);
  
  const renderFinished = () => (
      <div className="text-center p-8 bg-gray-800 rounded-lg shadow-xl max-w-2xl mx-auto">
        <h2 className="text-3xl font-bold text-white mb-4">Interview Round Complete</h2>
        <p className="text-gray-400 mb-6">Your answer has been recorded. Please proceed to the final results dashboard.</p>
        <div className="w-8 h-8 mx-auto border-4 border-t-transparent border-cyan-400 rounded-full animate-spin"></div>
      </div>
  );


  const renderSetup = () => (
    <div className="text-center p-8 bg-gray-800 rounded-lg shadow-xl max-w-2xl mx-auto">
      <CameraIcon className="w-16 h-16 mx-auto mb-4 text-cyan-400" />
      <h2 className="text-3xl font-bold text-white mb-2">HR Interview</h2>
      <p className="text-gray-300 mb-6">This is the final round. You will be asked one behavioral question. You will have 5 minutes to answer.</p>
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
        <button 
            onClick={beginRecording}
            className="px-8 py-3 text-lg font-semibold text-white bg-green-600 rounded-md hover:bg-green-500 transition-colors"
        >
            Start HR Interview
        </button>
    </div>
  );
  
  // This state is no longer used, as the component transitions immediately on completion.
  const renderAnalyzing = () => (
      <div className="text-center p-8 bg-gray-800 rounded-lg shadow-xl max-w-2xl mx-auto">
        <h2 className="text-3xl font-bold text-white mb-4">Finalizing...</h2>
        <div className="flex items-center justify-center space-x-2 text-cyan-400">
            <div className="w-8 h-8 border-4 border-t-transparent border-cyan-400 rounded-full animate-spin"></div>
            <span className="text-lg">Preparing final report.</span>
      </div>
    </div>
  );

  const renderRecording = () => {
    const question = HR_QUESTION;
    const mins = String(Math.floor(timeLeft / 60)).padStart(2, '0');
    const secs = String(timeLeft % 60).padStart(2, '0');
    return (
        <div className="w-full h-full flex flex-col p-4 gap-4">
            <div className="w-full bg-gray-800 rounded-lg p-4 border border-gray-700">
                <p className="text-sm text-violet-400 font-semibold">{question.category}</p>
                <h3 className="text-xl font-bold text-white mt-1">{question.question}</h3>
            </div>
            <div className="flex-1 flex flex-col md:flex-row gap-4 overflow-hidden">
                <div className="flex-1 bg-black rounded-lg overflow-hidden relative border-2 border-violet-500">
                    <video ref={videoRef} autoPlay muted playsInline className="w-full h-full object-cover"></video>
                    <div className="absolute top-3 left-3 flex items-center space-x-2 bg-red-600/90 text-white px-3 py-1 rounded-md text-sm font-semibold">
                       <RecordingIcon className="w-4 h-4" />
                       <span>REC</span>
                    </div>
                </div>
                <div className="w-full md:w-64 flex flex-col gap-4">
                    <div className="bg-gray-800 p-4 rounded-lg flex-1 flex flex-col items-center justify-center text-center">
                        <p className={`font-mono text-5xl font-bold mt-2 ${timeLeft < 60 ? 'text-red-500' : 'text-white'}`}>{mins}:{secs}</p>
                        <p className="text-gray-500 text-xs mt-1">Time Remaining</p>
                    </div>
                    <button 
                        onClick={handleFinishInterview}
                        className="w-full py-3 text-base font-semibold text-white bg-green-600 rounded-md hover:bg-green-500 transition-colors"
                    >
                        Finish Interview
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
      case 'finished': return renderFinished();
      default: return renderSetup();
    }
  };

  return (
    <div className="flex flex-col h-screen font-sans bg-gray-900 text-gray-200">
      <header className="flex items-center justify-between p-3 bg-gray-800 border-b border-gray-700 shadow-md">
        <h1 className="text-xl font-bold text-cyan-400">AI-Powered Interview Platform</h1>
        <div className="flex items-center space-x-4">
          <span className="text-sm font-semibold px-3 py-1 bg-violet-900/50 text-violet-300 rounded-full">Round 3 of 3: HR Interview</span>
        </div>
      </header>
      <main className="flex-1 flex items-center justify-center p-4">
        {renderContent()}
      </main>
    </div>
  );
};

export default HRInterview;
