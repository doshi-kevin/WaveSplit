import { VideoAnalysisResult, Emotion, EmotionDataPoint, HRVideoAnalysisResult } from '../types';
import { ExtendedVideoQuestion, HR_QUESTION } from '../constants';

// --- Emotion Analysis Simulation (Technical Round) ---

const POSITIVE_EMOTIONS: Emotion[] = ['Enthusiasm', 'Confidence', 'Calmness'];
const NEGATIVE_EMOTIONS: Emotion[] = ['Anxiety', 'Self-Doubt', 'Frustration'];

const generateEmotionDataPoint = (timestamp: number): EmotionDataPoint => {
    const emotions: Record<Emotion, number> = {
        Enthusiasm: 0, Confidence: 0, Calmness: 0,
        Anxiety: 0, 'Self-Doubt': 0, Frustration: 0,
    };
    
    let remaining = 100;
    emotions.Confidence = Math.floor(Math.random() * 40) + 20; // 20-60%
    remaining -= emotions.Confidence;
    emotions.Calmness = Math.floor(Math.random() * (remaining / 2));
    remaining -= emotions.Calmness;
    emotions.Enthusiasm = Math.floor(Math.random() * (remaining / 2));
    remaining -= emotions.Enthusiasm;
    
    emotions.Anxiety = Math.floor(Math.random() * remaining * 0.5);
    remaining -= emotions.Anxiety;
    emotions.Frustration = Math.floor(Math.random() * remaining * 0.5);
    remaining -= emotions.Frustration;
    emotions['Self-Doubt'] = remaining;

    return { timestamp, emotions };
};

const generateEmotionTimeline = (duration: number): EmotionDataPoint[] => {
    const timeline: EmotionDataPoint[] = [];
    for (let i = 0; i < duration; i++) {
        timeline.push(generateEmotionDataPoint(i));
    }
    return timeline;
};

const calculateEmotionScore = (timeline: EmotionDataPoint[]): number => {
    if (timeline.length === 0) return 0;
    let totalPositive = 0;
    let totalNegative = 0;
    timeline.forEach(dp => {
        POSITIVE_EMOTIONS.forEach(e => totalPositive += dp.emotions[e]);
        NEGATIVE_EMOTIONS.forEach(e => totalNegative += dp.emotions[e]);
    });
    const avgPositive = totalPositive / timeline.length;
    const avgNegative = totalNegative / timeline.length;
    const rawScore = avgPositive * 1.2 - avgNegative * 1.5;
    return Math.max(0, Math.min(100, 50 + rawScore));
};

const analyzeKeywords = (transcript: string, question: ExtendedVideoQuestion) => {
    const words = transcript.toLowerCase().replace(/[.,!?]/g, '').split(/\s+/);
    const matchedKeywords = question.expectedKeywords.filter(kw => words.includes(kw.toLowerCase()));
    const fillerWords = words.filter(word => question.fillerWords.includes(word));
    const keywordRatio = matchedKeywords.length / question.expectedKeywords.length;
    const fillerPenalty = Math.min(fillerWords.length * 5, 30);
    const score = Math.max(0, keywordRatio * 100 - fillerPenalty);
    return { score, matchedKeywords, fillerWords };
};

export const analyzeVideo = (videoBlob: Blob, question: ExtendedVideoQuestion): Promise<VideoAnalysisResult> => {
    return new Promise(resolve => {
        setTimeout(() => {
            console.log(`Analyzing technical video for question ${question.id}, blob size: ${videoBlob.size}`);
            const transcript = question.mockTranscript;
            const emotionTimeline = generateEmotionTimeline(150);
            const emotionAnalysis = {
                score: Math.round(calculateEmotionScore(emotionTimeline)),
                timeline: emotionTimeline,
            };
            const keywordAnalysis = analyzeKeywords(transcript, question);
            keywordAnalysis.score = Math.round(keywordAnalysis.score);
            resolve({ questionId: question.id, transcript, emotionAnalysis, keywordAnalysis });
        }, 500 + Math.random() * 500);
    });
};

// --- HR Analysis Simulation ---

export const analyzeHRTranscript = (transcript: string): Promise<HRVideoAnalysisResult> => {
    return new Promise(resolve => {
        setTimeout(() => {
            console.log(`Analyzing HR transcript`);

            // Confidence Analysis
            const confidence = 60 + Math.floor(Math.random() * 25); // 60-85%
            const nervousness = 10 + Math.floor(Math.random() * 15); // 10-25%
            const confidenceScore = Math.max(0, Math.min(100, confidence * 1.1 - nervousness * 1.3));

            // Response Analysis
            const words = transcript.toLowerCase().replace(/[.,!?]/g, '').split(/\s+/);
            const starMethod = {
                situation: HR_QUESTION.starKeywords.situation.some(kw => words.includes(kw)),
                task: HR_QUESTION.starKeywords.task.some(kw => words.includes(kw)),
                action: HR_QUESTION.starKeywords.action.some(kw => words.includes(kw)),
                result: HR_QUESTION.starKeywords.result.some(kw => words.includes(kw)),
            };
            const starScore = (Object.values(starMethod).filter(Boolean).length / 4) * 100;

            const softSkills = {
                matched: HR_QUESTION.softSkills.filter(kw => words.includes(kw)),
            };
            const softSkillsScore = (softSkills.matched.length / HR_QUESTION.softSkills.length) * 100;
            
            // Simple sentiment and response score
            const responseScore = (starScore * 0.7) + (softSkillsScore * 0.3); // Heavier weight on STAR

            resolve({
                questionId: HR_QUESTION.id,
                transcript,
                confidenceAnalysis: {
                    confidence,
                    nervousness,
                    score: Math.round(confidenceScore),
                },
                responseAnalysis: {
                    sentiment: 'Positive',
                    starMethod,
                    softSkills,
                    score: Math.round(responseScore),
                },
            });

        }, 1000 + Math.random() * 500); // Simulate analysis delay
    });
};
