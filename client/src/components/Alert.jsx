import React from 'react';

function Alert({color, message}) {

    function getColor(color) {
        switch (color) {
            case 'red':
                return 'border-red-400 text-red-700';
            case 'green':
                return 'border-green-400 text-green-700';
            case 'orange':
                return 'border-orange-400 text-orange-700';
            default:
                return 'border-blue-400 text-blue-700';
        }
    }

    return (
        <div className={`border ${getColor(color)} bg-white px-4 py-5 rounded relative flex items-center justify-center`} role="alert">
            <p>{message}</p>
        </div>
    );
}

export default Alert;