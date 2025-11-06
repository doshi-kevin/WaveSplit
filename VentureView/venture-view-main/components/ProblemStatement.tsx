
import React from 'react';
import { Problem } from '../types';

interface ProblemStatementProps {
  problem: Problem;
}

const ProblemStatement: React.FC<ProblemStatementProps> = ({ problem }) => {
  return (
    <div className="text-gray-300 space-y-6">
      <h2 className="text-2xl font-semibold text-white border-b border-gray-600 pb-2">{problem.title}</h2>
      
      <div className="bg-gray-700/50 p-2 rounded-md text-sm">
        <span className={`inline-block px-3 py-1 rounded-full text-xs font-semibold ${
          problem.difficulty === 'Medium' ? 'bg-yellow-500/20 text-yellow-300' : 'bg-green-500/20 text-green-300'
        }`}>
          {problem.difficulty}
        </span>
      </div>

      <p className="text-base leading-relaxed">{problem.description}</p>
      
      {problem.examples.map((example, index) => (
        <div key={index} className="space-y-2">
          <h3 className="font-semibold text-gray-100">Example {index + 1}:</h3>
          <pre className="bg-gray-900 p-3 rounded-lg text-sm font-mono whitespace-pre-wrap">
            <code>
              <strong>Input:</strong> {example.input}<br />
              <strong>Output:</strong> {example.output}<br />
              {example.explanation && <><strong>Explanation:</strong> {example.explanation}</>}
            </code>
          </pre>
        </div>
      ))}
      
      <div>
        <h3 className="font-semibold text-gray-100">Constraints:</h3>
        <ul className="list-disc list-inside mt-2 pl-2 space-y-1 text-sm">
          {problem.constraints.map((constraint, index) => (
            <li key={index}>
              <code className="bg-gray-700/80 px-1 py-0.5 rounded">{constraint}</code>
            </li>
          ))}
        </ul>
      </div>
    </div>
  );
};

export default ProblemStatement;
