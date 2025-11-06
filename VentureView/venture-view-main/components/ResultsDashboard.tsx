import React, { useState } from 'react';
import { VideoAnalysisResult, Emotion, HRVideoAnalysisResult } from '../types';
import { VIDEO_QUESTIONS, SCORING_WEIGHTS, HR_QUESTION } from '../constants';
import { analyzeHRTranscript } from '../services/mockAnalysisService';
import ChartIcon from './icons/ChartIcon';
import CheckIcon from './icons/CheckIcon';
import XIcon from './icons/XIcon';
import DownloadIcon from './icons/DownloadIcon';
import UploadIcon from './icons/UploadIcon';
import ThumbUpIcon from './icons/ThumbUpIcon';
import ThumbDownIcon from './icons/ThumbDownIcon';

interface ResultsDashboardProps {
  codingScore: number;
  videoResults: VideoAnalysisResult[];
  onFinishInterview: () => void;
}

const ProgressBar: React.FC<{ value: number }> = ({ value }) => (
    <div className="w-full bg-gray-700 rounded-full h-2.5">
        <div className="bg-cyan-500 h-2.5 rounded-full" style={{ width: `${value}%` }}></div>
    </div>
);

const TechnicalResult: React.FC<{ result: VideoAnalysisResult }> = ({ result }) => {
    const question = VIDEO_QUESTIONS.find(q => q.id === result.questionId);
    if (!question) return null;

    const totalScore = (result.emotionAnalysis.score * 0.4) + (result.keywordAnalysis.score * 0.6);

    return (
        <div className="bg-gray-800 p-6 rounded-lg border border-gray-700">
            <h3 className="text-lg font-bold text-white">{question.question}</h3>
            <p className="text-xs text-cyan-400 font-semibold mt-1">{question.category}</p>

            <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mt-4 text-center">
                <div className="bg-gray-700/50 p-3 rounded-lg">
                    <p className="text-sm text-gray-400">Emotion Score</p>
                    <p className="text-2xl font-bold text-sky-400">{result.emotionAnalysis.score}/100</p>
                </div>
                <div className="bg-gray-700/50 p-3 rounded-lg">
                    <p className="text-sm text-gray-400">Keyword Score</p>
                    <p className="text-2xl font-bold text-teal-400">{result.keywordAnalysis.score}/100</p>
                </div>
                <div className="bg-gray-900 p-3 rounded-lg border border-cyan-500">
                    <p className="text-sm text-gray-300">Question Score</p>
                    <p className="text-2xl font-bold text-cyan-400">{totalScore.toFixed(0)}/100</p>
                </div>
            </div>
             <div className="mt-4">
                <h4 className="text-sm font-semibold text-gray-300 mb-2">Answer Transcript</h4>
                <p className="text-sm text-gray-400 bg-gray-900/50 p-3 rounded-lg max-h-24 overflow-y-auto">{result.transcript}</p>
            </div>
        </div>
    );
};

const HRAnalysisSection: React.FC<{ onHrScoreCalculated: (score: number) => void }> = ({ onHrScoreCalculated }) => {
    const [hrAnalysis, setHrAnalysis] = useState<HRVideoAnalysisResult | null>(null);
    const [isAnalyzing, setIsAnalyzing] = useState(false);
    const [decision, setDecision] = useState<'none' | 'approved' | 'rejected'>('none');
    const fileInputRef = React.useRef<HTMLInputElement>(null);

    const handleDownloadTranscript = () => {
        const blob = new Blob([HR_QUESTION.mockTranscript], { type: 'text/plain' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = 'hr_interview_transcript.txt';
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
    };

    const handleUploadClick = () => {
        fileInputRef.current?.click();
    };

    const handleFileChange = async (event: React.ChangeEvent<HTMLInputElement>) => {
        const file = event.target.files?.[0];
        if (!file) return;

        setIsAnalyzing(true);
        // In a real app, you'd read the file content. Here we'll just use the mock transcript.
        const transcript = HR_QUESTION.mockTranscript; // await file.text();
        const result = await analyzeHRTranscript(transcript);
        setHrAnalysis(result);
        const hrScore = (result.confidenceAnalysis.score * 0.6 + result.responseAnalysis.score * 0.4);
        onHrScoreCalculated(hrScore);
        setIsAnalyzing(false);
    };
    
    const handleDecision = (newDecision: 'approved' | 'rejected') => {
        setDecision(newDecision);
    }

    if (!hrAnalysis) {
        return (
             <div className="bg-gray-800 p-6 rounded-lg border border-gray-700 text-center">
                <h3 className="text-lg font-bold text-white">HR Interview Analysis Required</h3>
                <p className="text-sm text-gray-400 mt-2 mb-4">Download the transcript, verify its contents, and upload it to generate the final analysis and make a hiring decision.</p>
                <div className="flex justify-center gap-4">
                    <button onClick={handleDownloadTranscript} className="flex items-center px-4 py-2 text-sm font-semibold text-white bg-gray-600 rounded-md hover:bg-gray-500 transition-colors">
                        <DownloadIcon className="w-4 h-4 mr-2"/>
                        Download Transcript
                    </button>
                    <button onClick={handleUploadClick} disabled={isAnalyzing} className="flex items-center px-4 py-2 text-sm font-semibold text-white bg-cyan-600 rounded-md hover:bg-cyan-500 disabled:bg-cyan-800 transition-colors">
                        <UploadIcon className="w-4 h-4 mr-2"/>
                        {isAnalyzing ? 'Analyzing...' : 'Upload & Analyze'}
                    </button>
                    <input type="file" ref={fileInputRef} onChange={handleFileChange} className="hidden" accept=".txt" />
                </div>
                {isAnalyzing && <div className="w-6 h-6 border-2 mt-4 mx-auto border-t-transparent border-cyan-400 rounded-full animate-spin"></div>}
            </div>
        )
    }

    const totalScore = (hrAnalysis.confidenceAnalysis.score * 0.6) + (hrAnalysis.responseAnalysis.score * 0.4);

    return (
        <div className="bg-gray-800 p-6 rounded-lg border border-gray-700">
            <div className="flex justify-between items-start">
                 <div>
                    <h3 className="text-lg font-bold text-white">{HR_QUESTION.question}</h3>
                    <p className="text-xs text-violet-400 font-semibold mt-1">{HR_QUESTION.category}</p>
                 </div>
                 {decision !== 'none' && (
                    <span className={`px-4 py-1.5 rounded-full font-semibold text-sm ${decision === 'approved' ? 'bg-green-500/20 text-green-300' : 'bg-red-500/20 text-red-300'}`}>
                        {decision === 'approved' ? 'Approved' : 'Rejected'}
                    </span>
                 )}
            </div>
            
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mt-4 text-center">
                <div className="bg-gray-700/50 p-3 rounded-lg">
                    <p className="text-sm text-gray-400">Confidence Score</p>
                    <p className="text-2xl font-bold text-sky-400">{hrAnalysis.confidenceAnalysis.score}/100</p>
                </div>
                <div className="bg-gray-700/50 p-3 rounded-lg">
                    <p className="text-sm text-gray-400">Response Quality</p>
                    <p className="text-2xl font-bold text-teal-400">{hrAnalysis.responseAnalysis.score}/100</p>
                </div>
                <div className="bg-gray-900 p-3 rounded-lg border border-violet-500">
                    <p className="text-sm text-gray-300">Question Score</p>
                    <p className="text-2xl font-bold text-violet-400">{totalScore.toFixed(0)}/100</p>
                </div>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mt-6">
                <div className="bg-gray-900/50 p-4 rounded-lg">
                     <h4 className="text-md font-semibold text-gray-200 mb-3">STAR Method Analysis</h4>
                     <div className="space-y-2">
                        {Object.entries(hrAnalysis.responseAnalysis.starMethod).map(([key, value]) => (
                            <div key={key} className={`flex items-center text-sm font-medium p-2 rounded-md ${value ? 'bg-green-800/40' : 'bg-red-800/40'}`}>
                                {value ? <CheckIcon className="w-5 h-5 mr-2 text-green-400"/> : <XIcon className="w-5 h-5 mr-2 text-red-400"/>}
                                <span className={value ? 'text-green-300' : 'text-red-300'}>{key.charAt(0).toUpperCase() + key.slice(1)}</span>
                                <span className="ml-auto text-xs text-gray-400">{value ? 'Detected' : 'Not Detected'}</span>
                            </div>
                        ))}
                     </div>
                </div>
                 <div className="bg-gray-900/50 p-4 rounded-lg">
                    <h4 className="text-md font-semibold text-gray-200 mb-3">Soft Skills Identified</h4>
                    <div className="flex flex-wrap gap-2">
                      {hrAnalysis.responseAnalysis.softSkills.matched.map(skill => (
                        <span key={skill} className="px-3 py-1 bg-cyan-900/80 text-cyan-300 text-sm rounded-full">{skill}</span>
                      ))}
                      {hrAnalysis.responseAnalysis.softSkills.matched.length === 0 && <p className="text-sm text-gray-500">None detected</p>}
                    </div>
                </div>
            </div>
             <div className="mt-6 bg-gray-900 border-t-2 border-cyan-500 p-4 rounded-b-lg">
                <h4 className="text-md font-semibold text-center text-gray-200 mb-3">Final Decision</h4>
                <div className="flex justify-center gap-4">
                     <button onClick={() => handleDecision('approved')} disabled={decision !== 'none'} className="flex items-center px-6 py-2 text-base font-semibold text-white bg-green-600 rounded-md hover:bg-green-500 disabled:bg-gray-600 disabled:cursor-not-allowed transition-colors">
                        <ThumbUpIcon className="w-5 h-5 mr-2" />
                        Approve Candidate
                    </button>
                    <button onClick={() => handleDecision('rejected')} disabled={decision !== 'none'} className="flex items-center px-6 py-2 text-base font-semibold text-white bg-red-600 rounded-md hover:bg-red-500 disabled:bg-gray-600 disabled:cursor-not-allowed transition-colors">
                        <ThumbDownIcon className="w-5 h-5 mr-2" />
                        Reject Candidate
                    </button>
                </div>
            </div>
        </div>
    );
};


const ResultsDashboard: React.FC<ResultsDashboardProps> = ({ codingScore, videoResults, onFinishInterview }) => {
    const [hrScore, setHrScore] = useState(0);
    
    const technicalScore = videoResults.length > 0
        ? (videoResults[0].emotionAnalysis.score * 0.4 + videoResults[0].keywordAnalysis.score * 0.6)
        : 0;
        
    const finalScore = (codingScore * SCORING_WEIGHTS.coding) + (technicalScore * SCORING_WEIGHTS.technical) + (hrScore * SCORING_WEIGHTS.hr);

    const getRecommendation = (score: number) => {
        if (score >= 85) return { text: "Highly Recommended", color: "text-green-400" };
        if (score >= 70) return { text: "Recommended", color: "text-cyan-400" };
        if (score >= 50) return { text: "Consider", color: "text-yellow-400" };
        return { text: "Not Recommended", color: "text-red-400" };
    };
    
    const recommendation = getRecommendation(finalScore);

    return (
        <div className="min-h-screen font-sans bg-gray-900 text-gray-200">
            <header className="flex items-center justify-between p-3 bg-gray-800 border-b border-gray-700 shadow-md sticky top-0 z-10">
                <h1 className="text-xl font-bold text-cyan-400">AI-Powered Interview Platform</h1>
                <button className="flex items-center px-3 py-1.5 text-xs font-semibold text-white bg-gray-600 rounded-md hover:bg-gray-500 transition-colors">
                    <DownloadIcon className="w-4 h-4 mr-2"/>
                    Export as PDF
                </button>
            </header>
            
            <main className="p-4 md:p-8 max-w-7xl mx-auto">
                <div className="bg-gray-800/50 p-6 rounded-lg border border-gray-700 flex flex-col md:flex-row justify-between items-center gap-6">
                    <div>
                        <h2 className="text-3xl font-bold text-white">Final Performance Report</h2>
                        <p className="text-gray-400 mt-1">An AI-generated report on the candidate's performance across all rounds.</p>
                    </div>
                    <div className="text-center">
                         <p className="text-sm text-gray-300">Final Score</p>
                         <p className="text-5xl font-bold text-cyan-400">{finalScore.toFixed(0)}<span className="text-3xl text-gray-500">/100</span></p>
                    </div>
                    <div className="text-center">
                        <p className="text-sm text-gray-300">Recommendation</p>
                        <p className={`text-2xl font-bold ${recommendation.color}`}>{recommendation.text}</p>
                    </div>
                </div>

                <div className="mt-8 grid grid-cols-1 lg:grid-cols-3 gap-8">
                    <div className="lg:col-span-1 flex flex-col gap-8">
                        <div className="bg-gray-800 p-6 rounded-lg border border-gray-700">
                           <h3 className="font-semibold text-white mb-4 text-lg">Score Breakdown</h3>
                           <div className="space-y-4">
                               <div>
                                   <div className="flex justify-between items-center mb-1">
                                       <p>Coding Challenge ({SCORING_WEIGHTS.coding * 100}%)</p>
                                       <p className="font-bold text-lg">{codingScore.toFixed(0)}/100</p>
                                   </div>
                                   <ProgressBar value={codingScore} />
                               </div>
                               <div>
                                   <div className="flex justify-between items-center mb-1">
                                       <p>Technical Interview ({SCORING_WEIGHTS.technical * 100}%)</p>
                                       <p className="font-bold text-lg">{technicalScore.toFixed(0)}/100</p>
                                   </div>
                                   <ProgressBar value={technicalScore} />
                               </div>
                               <div>
                                   <div className="flex justify-between items-center mb-1">
                                       <p>HR Interview ({SCORING_WEIGHTS.hr * 100}%)</p>
                                       <p className="font-bold text-lg">{hrScore.toFixed(0)}/100</p>
                                   </div>
                                   <ProgressBar value={hrScore} />
                               </div>
                           </div>
                        </div>
                         <div className="bg-gray-800 p-6 rounded-lg border border-gray-700">
                           <h3 className="font-semibold text-white mb-3 text-lg">Next Steps</h3>
                           <p className="text-sm text-gray-400 mb-4">You have completed the automated portion of the interview. The hiring team will review your results.</p>
                           <button 
                                onClick={onFinishInterview}
                                className="w-full py-2 text-base font-semibold text-white bg-green-600 rounded-md hover:bg-green-500 transition-colors"
                            >
                                Finish & Exit
                            </button>
                        </div>
                    </div>
                    <div className="lg:col-span-2 space-y-6">
                        <div>
                            <div className="flex items-center gap-2 border-b border-gray-700 mb-4">
                                <ChartIcon className="w-5 h-5 text-cyan-400"/>
                                <h3 className="text-xl font-bold text-white py-2">Technical Video Analysis</h3>
                            </div>
                            {videoResults.map(res => <TechnicalResult key={res.questionId} result={res} />)}
                        </div>
                         <div>
                            <div className="flex items-center gap-2 border-b border-gray-700 mb-4">
                                <ChartIcon className="w-5 h-5 text-violet-400"/>
                                <h3 className="text-xl font-bold text-white py-2">HR Interview Analysis</h3>
                            </div>
                            <HRAnalysisSection onHrScoreCalculated={setHrScore} />
                        </div>
                    </div>
                </div>
            </main>
        </div>
    );
};

export default ResultsDashboard;