import React from 'react';

const RecordingIcon: React.FC<React.SVGProps<SVGSVGElement>> = (props) => (
    <svg 
        xmlns="http://www.w3.org/2000/svg" 
        viewBox="0 0 24 24" 
        fill="currentColor" 
        {...props}
    >
        <circle cx="12" cy="12" r="10" className="animate-pulse" />
    </svg>
);

export default RecordingIcon;
