
import React, { useState, useEffect } from 'react';

interface TimerProps {
  initialMinutes: number;
}

const Timer: React.FC<TimerProps> = ({ initialMinutes }) => {
  const [seconds, setSeconds] = useState(initialMinutes * 60);

  useEffect(() => {
    if (seconds <= 0) return;

    const interval = setInterval(() => {
      setSeconds(prev => prev - 1);
    }, 1000);

    return () => clearInterval(interval);
  }, [seconds]);

  const formatTime = () => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${String(mins).padStart(2, '0')}:${String(secs).padStart(2, '0')}`;
  };

  const timeColorClass = seconds < 300 ? 'text-red-500' : 'text-gray-300';

  return (
    <div className={`text-lg font-mono font-semibold ${timeColorClass}`}>
      {formatTime()}
    </div>
  );
};

export default Timer;
