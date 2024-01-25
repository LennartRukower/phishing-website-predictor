import React from "react";

function Card({ title, content, clickable, isSelected, onClick }) {
    function getStyle() {
        if (clickable) {
            if (isSelected) {
                return "bg-white shadow-md rounded-lg p-4 cursor-pointer hover:bg-gray-100 border-2 border-purple-500";
            }
            return "bg-white shadow-md rounded-lg p-4 cursor-pointer hover:bg-gray-100 border-2 border-transparent";
        }
        return "bg-white shadow-md rounded-lg p-4";
    }
    return (
        <div className={getStyle()} onClick={onClick}>
            {title ? (
                <h2 className="text-lg font-semibold mb-2 h-14 overflow-hidden">{title}</h2>
            ) : null}
            {content}
        </div>
    );
}

export default Card;
