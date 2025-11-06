
import { TestCase, TestCaseResult } from '../types';

// This is a very simplified mock of a code execution service.
// In a real application, this would be a backend service that runs the code in a sandbox.
// Here, we just check for a keyword in the code to determine if the solution is "correct".

const SOLUTIONS: { [key: string]: string } = {
  javascript: 'new Map()',
  python: 'dict()',
  java: 'new HashMap<>()',
  cpp: 'unordered_map',
};

const solveTwoSum = (nums: number[], target: number): number[] => {
    const map = new Map<number, number>();
    for (let i = 0; i < nums.length; i++) {
        const complement = target - nums[i];
        if (map.has(complement)) {
            return [map.get(complement)!, i];
        }
        map.set(nums[i], i);
    }
    return [];
};

export const mockCodeExecution = (code: string, languageId: string, testCases: TestCase[]): Promise<TestCaseResult[]> => {
  return new Promise(resolve => {
    setTimeout(() => {
      const results: TestCaseResult[] = testCases.map(testCase => {
        const isCorrectSolution = code.includes(SOLUTIONS[languageId]);
        let output: any;
        let passed = false;

        if (isCorrectSolution) {
            // For JS, we can actually run the logic. For others, we assume it's correct if the keyword is present.
            if (languageId === 'javascript') {
                try {
                    // This is a simplified evaluation and NOT safe for production.
                    // A real solution would use a sandboxed environment.
                    const userSolution = solveTwoSum(testCase.input.nums, testCase.input.target);
                    output = userSolution.sort();
                    passed = JSON.stringify(output) === JSON.stringify(testCase.expected.sort());
                } catch(e) {
                    output = 'Runtime Error';
                    passed = false;
                }
            } else {
                 output = testCase.expected;
                 passed = true;
            }
        } else {
            output = languageId === 'javascript' ? [] : 'Incorrect output';
            passed = false;
        }
        
        return {
          ...testCase,
          output: passed ? testCase.expected : output,
          passed: passed,
        };
      });
      resolve(results);
    }, 1500); // Simulate network delay
  });
};
