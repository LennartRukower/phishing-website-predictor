import React from "react";

function Card({ title, content, clickable, isSelected, onClick, onButtonClick }) {
    function getStyle() {
        if (clickable) {
            if (isSelected) {
                return "bg-white shadow-md rounded-lg p-4 cursor-pointer hover:bg-gray-100 border-2 border-purple-500";
            }
            return "bg-white shadow-md rounded-lg p-4 cursor-pointer hover:bg-gray-100 border-2 border-transparent";
        }
        return "bg-white shadow-md rounded-lg p-4";
    }

    function handleButtonClick(event) {
        event.stopPropagation(); // Stop event propagation
        onButtonClick();
    }

    return (
        <div id="card" className={getStyle()} onClick={onClick}>
            {title ? (
                <div>
                    <div className="flex justify-between items-start">
                        <h2 className="text-lg font-semibold mb-2 h-14 overflow-hidden">{title}</h2>
                        {/* Icon button */}
                        {/* TODO: Add correct button styling and icon */}
                        <button
                            id="button"
                            className="hover:cursor-pointer"
                            onClick={handleButtonClick}
                        >
                            <svg
                                className="w-6 h-6 text-gray-500 hover:text-purple-500"
                                fill="none"
                                stroke="currentColor"
                                viewBox="0 0 24 24"
                                xmlns="http://www.w3.org/2000/svg"
                            >
                                <path
                                    strokeLinecap="round"
                                    strokeLinejoin="round"
                                    strokeWidth="2"
                                    d="M12 4v2m0 6h.01m-6.938 4.938a9 9 0 1112.728 0M12 14v4"
                                ></path>
                            </svg>
                        </button>
                    </div>
                </div>
            ) : null}
            {content}
        </div>
    );
}

export default Card;
