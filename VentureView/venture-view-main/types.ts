export interface Language {
  id: string;
  name: string;
  boilerplate: string;
}

export interface TestCase {
  input: { nums: number[]; target: number };
  expected: number[];
  hidden: boolean;
}

export interface TestCaseResult extends TestCase {
  output: any;
  passed: boolean;
}

export type ExecutionStatus = 'idle' | 'running' | 'completed' | 'error';

export interface Problem {
    title: string;
    difficulty: 'Easy' | 'Medium' | 'Hard';
    description: string;
    examples: {
        input: string;
        output: string;
        explanation?: string;
    }[];
    constraints: string[];
}

export interface VideoQuestion {
    id: number;
    question: string;
    category: string;
}

// Types for Technical AI Analysis
export type Emotion = 'Enthusiasm' | 'Confidence' | 'Calmness' | 'Anxiety' | 'Self-Doubt' | 'Frustration';

export interface EmotionDataPoint {
    timestamp: number; // in seconds
    emotions: Record<Emotion, number>; // percentage
}

export interface EmotionAnalysisResult {
    score: number; // 0-100
    timeline: EmotionDataPoint[];
}

export interface KeywordAnalysisResult {
    score: number; // 0-100
    matchedKeywords: string[];
    fillerWords: string[];
}

export interface VideoAnalysisResult {
    questionId: number;
    transcript: string;
    emotionAnalysis: EmotionAnalysisResult;
    keywordAnalysis: KeywordAnalysisResult;
}

// Types for HR AI Analysis
export interface ConfidenceAnalysis {
    confidence: number; // 0-100
    nervousness: number; // 0-100
    score: number; // 0-100
}

export interface STARMethodAnalysis {
    situation: boolean;
    task: boolean;
    action: boolean;
    result: boolean;
}

export interface SoftSkillsAnalysis {
    matched: string[];
}

export interface HRResponseAnalysis {
    sentiment: 'Positive' | 'Neutral' | 'Negative';
    starMethod: STARMethodAnalysis;
    softSkills: SoftSkillsAnalysis;
    score: number; // 0-100
}

export interface HRVideoAnalysisResult {
    questionId: string;
    transcript: string;
    confidenceAnalysis: ConfidenceAnalysis;
    responseAnalysis: HRResponseAnalysis;
}