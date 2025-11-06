import React from 'react';
import { TestCaseResult, ExecutionStatus } from '../types';
import CheckIcon from './icons/CheckIcon';
import XIcon from './icons/XIcon';
import CameraIcon from './icons/CameraIcon';

interface TestResultsProps {
  results: TestCaseResult[];
  status: ExecutionStatus;
  score: number;
  isSubmissionComplete: boolean;
  onProceed: () => void;
}

const ResultCard: React.FC<{ result: TestCaseResult, index: number }> = ({ result, index }) => {
    const isPassed = result.passed;
    return (
        <div className={`p-4 rounded-lg border ${isPassed ? 'bg-green-900/30 border-green-700/50' : 'bg-red-900/30 border-red-700/50'}`}>
            <div className="flex justify-between items-center mb-2">
                <h4 className="font-semibold text-base text-gray-100">
                    Test Case {index + 1} {result.hidden ? '(Hidden)' : ''}
                </h4>
                <span className={`flex items-center text-sm font-bold ${isPassed ? 'text-green-400' : 'text-red-400'}`}>
                    {isPassed ? <CheckIcon className="w-5 h-5 mr-1" /> : <XIcon className="w-5 h-5 mr-1" />}
                    {isPassed ? 'Passed' : 'Failed'}
                </span>
            </div>
            <div className="text-xs font-mono text-gray-400 space-y-1">
                <p><strong>Input:</strong> {JSON.stringify(result.input)}</p>
                <p><strong>Expected:</strong> {JSON.stringify(result.expected)}</p>
                <p><strong>Your Output:</strong> {JSON.stringify(result.output)}</p>
            </div>
        </div>
    );
};

const TestResults: React.FC<TestResultsProps> = ({ results, status, score, isSubmissionComplete, onProceed }) => {
  if (status === 'idle') {
    return <div className="text-center text-gray-500">Run your code to see the test results.</div>;
  }

  if (status === 'running') {
    return (
      <div className="flex items-center justify-center space-x-2 text-cyan-400">
        <div className="w-4 h-4 border-2 border-t-transparent border-cyan-400 rounded-full animate-spin"></div>
        <span>Running tests...</span>
      </div>
    );
  }

  if (status === 'error') {
    return <div className="text-center text-red-500">An error occurred during execution.</div>;
  }

  return (
    <div className="space-y-4">
      {isSubmissionComplete && (
        <div className="p-4 bg-gray-900/50 rounded-lg text-center flex flex-col items-center gap-4">
            <h3 className="text-xl font-bold">Final Score: <span className="text-cyan-400">{score.toFixed(0)} / 100</span></h3>
            <p className="text-sm text-gray-400">You have completed the coding challenge.</p>
            <button
                onClick={onProceed}
                className="flex items-center px-6 py-2 text-base font-semibold text-white bg-cyan-600 rounded-md hover:bg-cyan-500 transition-colors"
            >
                <CameraIcon className="w-5 h-5 mr-2" />
                Proceed to Video Interview
            </button>
        </div>
      )}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4">
        {results.map((result, index) => (
          <ResultCard key={index} result={result} index={index} />
        ))}
      </div>
    </div>
  );
};

export default TestResults;