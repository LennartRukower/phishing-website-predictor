import React from 'react';

function Card({title, content}) {
    return (
        <div className="bg-white shadow-md rounded-lg p-4">
            <h2 className="text-xl font-bold mb-2">{title}</h2>
            {content}
        </div>
    );
}

export default Card;