
import React from 'react';
import Timer from './Timer';
import PlayIcon from './icons/PlayIcon';
import UploadIcon from './icons/UploadIcon';
import { Language, ExecutionStatus } from '../types';
import { LANGUAGES } from '../constants';

interface CodeEditorProps {
  language: Language;
  onLanguageChange: (language: Language) => void;
  code: string;
  setCode: (code: string) => void;
  onRun: () => void;
  onSubmit: () => void;
  status: ExecutionStatus;
}

const CodeEditor: React.FC<CodeEditorProps> = ({
  language,
  onLanguageChange,
  code,
  setCode,
  onRun,
  onSubmit,
  status,
}) => {
  
  const handleLanguageSelect = (e: React.ChangeEvent<HTMLSelectElement>) => {
    const selectedLanguage = LANGUAGES.find(lang => lang.id === e.target.value);
    if (selectedLanguage) {
      onLanguageChange(selectedLanguage);
    }
  };
  
  return (
    <div className="flex flex-col h-3/5 bg-gray-900">
      <div className="flex items-center justify-between p-2 bg-gray-800 border-b border-gray-700">
        <select
          value={language.id}
          onChange={handleLanguageSelect}
          className="bg-gray-700 text-white text-sm rounded px-3 py-1.5 focus:outline-none focus:ring-2 focus:ring-cyan-500"
        >
          {LANGUAGES.map(lang => (
            <option key={lang.id} value={lang.id}>{lang.name}</option>
          ))}
        </select>
        <Timer initialMinutes={60} />
      </div>

      <div className="flex-1 relative">
        <textarea
          value={code}
          onChange={(e) => setCode(e.target.value)}
          className="w-full h-full p-4 bg-transparent text-gray-200 font-mono text-sm resize-none focus:outline-none"
          placeholder="Write your code here..."
          spellCheck="false"
        />
      </div>

      <div className="flex justify-end items-center p-3 bg-gray-800 border-t border-gray-700 space-x-3">
        <button
          onClick={onRun}
          disabled={status === 'running'}
          className="flex items-center px-4 py-2 text-sm font-semibold text-white bg-gray-600 rounded-md hover:bg-gray-500 disabled:bg-gray-700 disabled:cursor-not-allowed transition-colors"
        >
          <PlayIcon className="w-4 h-4 mr-2"/>
          Run Tests
        </button>
        <button
          onClick={onSubmit}
          disabled={status === 'running'}
          className="flex items-center px-4 py-2 text-sm font-semibold text-white bg-green-600 rounded-md hover:bg-green-500 disabled:bg-green-800 disabled:cursor-not-allowed transition-colors"
        >
          <UploadIcon className="w-4 h-4 mr-2" />
          Submit
        </button>
      </div>
    </div>
  );
};

export default CodeEditor;
