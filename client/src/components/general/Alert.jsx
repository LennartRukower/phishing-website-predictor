import React from "react";

function Alert({ color, message, showDetails }) {
    function getColor(color) {
        switch (color) {
            case "red":
                return "border-red-400 text-red-700";
            case "green":
                return "border-green-400 text-green-700";
            case "orange":
                return "border-orange-400 text-orange-700";
            default:
                return "border-blue-400 text-blue-700";
        }
    }

    function getButtonColor() {
        switch (color) {
            case "red":
                return "bg-red-500 hover:bg-red-700";
            case "green":
                return "bg-green-500 hover:bg-green-700";
            case "orange":
                return "bg-orange-500 hover:bg-orange-700";
            default:
                return "bg-blue-500 hover:bg-blue-700";
        }
    }

    return (
        <div
            className={`border ${getColor(
                color
            )} bg-white px-4 py-5 rounded relative flex items-center justify-between`}
            role="alert"
        >
            <p>{message}</p>
            {showDetails ? (
                <button
                    className={`${getButtonColor()} text-white font-bold py-2 px-4 rounded`}
                    type="button"
                >
                    Details
                </button>
            ) : null}
        </div>
    );
}

export default Alert;
