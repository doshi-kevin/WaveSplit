import { Language, TestCase, Problem, VideoQuestion } from './types';

export interface ExtendedVideoQuestion extends VideoQuestion {
    expectedKeywords: string[];
    fillerWords: string[];
    mockTranscript: string;
}

export const LANGUAGES: Language[] = [
  {
    id: 'python',
    name: 'Python',
    boilerplate: `class Solution:
    def twoSum(self, nums: list[int], target: int) -> list[int]:
        # Your code here
        pass
`,
  },
  {
    id: 'javascript',
    name: 'JavaScript',
    boilerplate: `/**
 * @param {number[]} nums
 * @param {number} target
 * @return {number[]}
 */
var twoSum = function(nums, target) {
    // Your code here
};`,
  },
  {
    id: 'java',
    name: 'Java',
    boilerplate: `class Solution {
    public int[] twoSum(int[] nums, int target) {
        // Your code here
        return new int[2];
    }
}`,
  },
  {
    id: 'cpp',
    name: 'C++',
    boilerplate: `#include <vector>
#include <unordered_map>

class Solution {
public:
    std::vector<int> twoSum(std::vector<int>& nums, int target) {
        // Your code here
        return {};
    }
};`,
  },
];

export const TEST_CASES: TestCase[] = [
  // 8 visible test cases
  { input: { nums: [2, 7, 11, 15], target: 9 }, expected: [0, 1], hidden: false },
  { input: { nums: [3, 2, 4], target: 6 }, expected: [1, 2], hidden: false },
  { input: { nums: [3, 3], target: 6 }, expected: [0, 1], hidden: false },
  { input: { nums: [0, 4, 3, 0], target: 0 }, expected: [0, 3], hidden: false },
  { input: { nums: [-1, -2, -3, -4, -5], target: -8 }, expected: [2, 4], hidden: false },
  { input: { nums: [5, 75, 25], target: 100 }, expected: [1, 2], hidden: false },
  { input: { nums: [10, 20, 30, 40, 50], target: 90 }, expected: [3, 4], hidden: false },
  { input: { nums: [1, 2, 3, 4, 5], target: 3 }, expected: [0, 1], hidden: false },
  // 2 hidden test cases
  { input: { nums: Array.from({length: 100}, (_, i) => i), target: 197 }, expected: [98, 99], hidden: true },
  { input: { nums: [150, 24, 79, 50, 88, 345, 3], target: 200 }, expected: [0, 3], hidden: true },
];

export const PROBLEM: Problem = {
    title: 'Two Sum',
    difficulty: 'Medium',
    description: 'Given an array of integers `nums` and an integer `target`, return indices of the two numbers such that they add up to `target`.\n\nYou may assume that each input would have exactly one solution, and you may not use the same element twice.\n\nYou can return the answer in any order.',
    examples: [
        {
            input: 'nums = [2, 7, 11, 15], target = 9',
            output: '[0, 1]',
            explanation: 'Because nums[0] + nums[1] == 9, we return [0, 1].'
        },
        {
            input: 'nums = [3, 2, 4], target = 6',
            output: '[1, 2]'
        },
        {
            input: 'nums = [3, 3], target = 6',
            output: '[0, 1]'
        }
    ],
    constraints: [
        '2 <= nums.length <= 10^4',
        '-10^9 <= nums[i] <= 10^9',
        '-10^9 <= target <= 10^9',
        'Only one valid answer exists.'
    ]
};

export const VIDEO_QUESTIONS: ExtendedVideoQuestion[] = [
    {
        id: 1,
        category: 'System Design',
        question: 'How would you design a URL shortening service like TinyURL? Describe the main components and data flow.',
        expectedKeywords: ['hash', 'redirect', 'database', 'API', 'load balancer', 'cache', 'scalability', 'collision'],
        fillerWords: ['um', 'like', 'you know', 'basically', 'so'],
        mockTranscript: "Okay, so for a URL shortener, I'd basically use a hashing function to convert the long URL into a short key. This key and the original URL would be stored in a database, maybe a NoSQL one for scalability. When a user hits the short URL, an API endpoint would look up the key in the database, fetch the long URL, and issue a 301 redirect. To handle high traffic, I'd put a load balancer in front of the application servers and use a cache like Redis to store popular URLs to reduce database hits. We also need to consider hash collisions."
    },
    {
        id: 2,
        category: 'API & Web Concepts',
        question: 'Explain the difference between REST and GraphQL. When would you choose one over the other?',
        expectedKeywords: ['endpoint', 'over-fetching', 'under-fetching', 'schema', 'query', 'mutation', 'resource', 'type system'],
        fillerWords: ['uh', 'like', 'I guess', 'kind of'],
        mockTranscript: "The main difference is that REST is based on resources and has multiple endpoints, like /users or /posts. With REST, you often face over-fetching or under-fetching data. GraphQL, on the other hand, uses a single endpoint and has a strong type system defined by a schema. The client can specify exactly what data it needs in a query, which solves those fetching problems. I'd choose GraphQL for complex applications with nested data, like a social media app, and REST for simpler, resource-oriented APIs."
    },
    {
        id: 3,
        category: 'Core CS Concepts',
        question: 'What is the purpose of database indexing? Explain how it works and the trade-offs involved.',
        expectedKeywords: ['performance', 'query speed', 'B-tree', 'lookup', 'write overhead', 'storage', 'data structure', 'trade-off'],
        fillerWords: ['um', 'like', 'actually', 'sort of'],
        mockTranscript: "The main purpose of indexing is to dramatically improve the performance of database queries. It's like an index in a book; instead of scanning the whole table, the database can use the index, which is usually a B-tree data structure, to quickly find the location of the data. This makes lookups much faster. The trade-off, however, is that indexes take up extra storage space. Also, while they speed up reads, they add overhead to write operations like inserts and updates, because the index itself also needs to be updated."
    }
];

export const HR_QUESTION = {
    id: 'hr-1',
    category: 'Behavioral Question',
    question: 'Tell me about a time you faced a significant challenge at work and how you handled it. What was the outcome?',
    softSkills: ['problem-solving', 'adaptability', 'communication', 'leadership', 'teamwork', 'resilience'],
    starKeywords: {
        situation: ['project', 'task', 'team', 'deadline', 'client'],
        task: ['goal', 'objective', 'had to', 'needed to', 'responsible for'],
        action: ['I decided', 'we implemented', 'I developed', 'we communicated', 'I analyzed'],
        result: ['outcome', 'achieved', 'improved', 'led to', 'as a result']
    },
    mockTranscript: "In my previous role, we were facing a tight deadline on a critical client project, and a key team member left unexpectedly. That was the situation. My task was to ensure we still met the deadline without compromising quality. As the team lead, the action I took was to first redistribute the workload, but I also organized a meeting to communicate the challenge and brainstorm solutions with the team. I developed a revised timeline and we implemented a new pair-programming approach to get the new member up to speed. As a result, not only did we successfully deliver the project on time, but the outcome was that the team's morale actually improved due to better teamwork and adaptability."
};

export const SCORING_WEIGHTS = {
    coding: 0.40,
    technical: 0.35,
    hr: 0.25,
};