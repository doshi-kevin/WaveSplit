import React, { useState, useEffect, useCallback } from 'react';
import ProblemStatement from './components/ProblemStatement';
import CodeEditor from './components/CodeEditor';
import TestResults from './components/TestResults';
import VideoInterview from './components/VideoInterview';
import HRInterview from './components/HRInterview';
import ResultsDashboard from './components/ResultsDashboard';
import { mockCodeExecution } from './services/mockCodeExecution';
import { Language, TestCaseResult, ExecutionStatus, VideoAnalysisResult, HRVideoAnalysisResult } from './types';
import { LANGUAGES, TEST_CASES, PROBLEM } from './constants';

type InterviewRound = 'coding' | 'video' | 'hr' | 'results' | 'finished';

const App: React.FC = () => {
  const [language, setLanguage] = useState<Language>(LANGUAGES[1]); // Default to JavaScript
  const [code, setCode] = useState<string>(language.boilerplate);
  const [executionStatus, setExecutionStatus] = useState<ExecutionStatus>('idle');
  const [testResults, setTestResults] = useState<TestCaseResult[]>([]);
  const [score, setScore] = useState<number>(0);
  const [activeTab, setActiveTab] = useState<'results' | 'problem'>('problem');
  const [round, setRound] = useState<InterviewRound>('coding');
  const [isSubmissionComplete, setIsSubmissionComplete] = useState(false);
  const [videoAnalysisResults, setVideoAnalysisResults] = useState<VideoAnalysisResult[]>([]);
  
  // HR Analysis is now handled inside the dashboard
  // const [hrAnalysisResult, setHrAnalysisResult] = useState<HRVideoAnalysisResult | null>(null);


  const handleLanguageChange = (newLanguage: Language) => {
    const savedCode = localStorage.getItem(`code-${newLanguage.id}`);
    setLanguage(newLanguage);
    setCode(savedCode || newLanguage.boilerplate);
    setTestResults([]);
    setScore(0);
  };

  // Auto-save feature
  useEffect(() => {
    const handler = setTimeout(() => {
      localStorage.setItem(`code-${language.id}`, code);
    }, 1000);

    return () => {
      clearTimeout(handler);
    };
  }, [code, language.id]);
  
  // Load saved code on initial mount
  useEffect(() => {
    const savedCode = localStorage.getItem(`code-${language.id}`);
    if (savedCode) {
      setCode(savedCode);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const runCode = useCallback(async (isSubmission: boolean) => {
    setExecutionStatus('running');
    setActiveTab('results');
    setIsSubmissionComplete(false);
    const casesToRun = isSubmission ? TEST_CASES : TEST_CASES.slice(0, 8);
    
    try {
      const results = await mockCodeExecution(code, language.id, casesToRun);
      setTestResults(results);
      
      const passedCount = results.filter(r => r.passed).length;
      const newScore = (passedCount / TEST_CASES.length) * 100;
      if (isSubmission) {
        setScore(newScore);
        setIsSubmissionComplete(true);
      } else {
        setScore(0);
      }
      setExecutionStatus('completed');
    } catch (error) {
      console.error('Execution failed:', error);
      setExecutionStatus('error');
    }
  }, [code, language.id]);
  
  const handleRun = () => runCode(false);
  const handleSubmit = () => runCode(true);
  
  const handleVideoComplete = (results: VideoAnalysisResult[]) => {
    setVideoAnalysisResults(results);
    setRound('hr');
  };

  const handleHRComplete = () => {
    // HR recording is done, now move to the results page where the recruiter will analyze it.
    setRound('results');
  };

  const handleFinishInterview = () => {
    setRound('finished');
  }

  if (round === 'video') {
    return <VideoInterview onComplete={handleVideoComplete} />;
  }

  if (round === 'hr') {
    return <HRInterview onComplete={handleHRComplete} />;
  }
  
  if (round === 'results') {
    return <ResultsDashboard 
      codingScore={score} 
      videoResults={videoAnalysisResults}
      onFinishInterview={handleFinishInterview}
    />
  }

  if (round === 'finished') {
    return (
      <div className="flex flex-col h-screen font-sans bg-gray-900 text-gray-200 items-center justify-center">
        <header className="absolute top-0 left-0 right-0 flex items-center justify-between p-3 bg-gray-800 border-b border-gray-700 shadow-md">
          <h1 className="text-xl font-bold text-cyan-400">AI-Powered Interview Platform</h1>
        </header>
        <div className="text-center p-8 bg-gray-800 rounded-lg shadow-xl">
          <h2 className="text-3xl font-bold text-cyan-400 mb-4">Interview Complete</h2>
          <p className="text-lg text-gray-300">Thank you for completing all rounds of the interview. <br/> We will be in touch with you shortly regarding the results.</p>
        </div>
      </div>
    );
  }

  return (
    <div className="flex flex-col h-screen font-sans bg-gray-900 text-gray-200">
      <header className="flex items-center justify-between p-3 bg-gray-800 border-b border-gray-700 shadow-md">
        <h1 className="text-xl font-bold text-cyan-400">AI-Powered Interview Platform</h1>
        <div className="flex items-center space-x-4">
          <span className="text-sm font-semibold px-3 py-1 bg-cyan-900/50 text-cyan-300 rounded-full">Round 1 of 3: Coding</span>
        </div>
      </header>

      <div className="flex flex-1 overflow-hidden">
        <div className="w-full md:w-2/5 lg:w-1/3 p-4 overflow-y-auto border-r border-gray-700 bg-gray-800/50">
           <ProblemStatement problem={PROBLEM} />
        </div>
        
        <div className="flex flex-col flex-1">
          <CodeEditor
            language={language}
            onLanguageChange={handleLanguageChange}
            code={code}
            setCode={setCode}
            onRun={handleRun}
            onSubmit={handleSubmit}
            status={executionStatus}
          />
          <div className="flex-grow flex flex-col bg-gray-800 border-t border-gray-700 overflow-hidden">
            <div className="flex border-b border-gray-700">
                <button 
                  onClick={() => setActiveTab('results')}
                  className={`px-4 py-2 text-sm font-medium ${activeTab === 'results' ? 'bg-gray-700 text-cyan-400 border-b-2 border-cyan-400' : 'text-gray-400 hover:bg-gray-700/50'}`}
                >
                  Test Results
                </button>
            </div>
            <div className="flex-1 p-4 overflow-y-auto">
              {activeTab === 'results' && (
                <TestResults 
                  results={testResults} 
                  status={executionStatus}
                  score={score}
                  isSubmissionComplete={isSubmissionComplete}
                  onProceed={() => setRound('video')}
                />
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default App;
