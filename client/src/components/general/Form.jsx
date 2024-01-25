// Form renders a form element which renders given children and makes a given action when submitted

import React from "react";

function Form({ children, onSubmit, onError, disabledSubmit }) {
    function getStyle() {
        if (disabledSubmit) {
            return "bg-purple-500 hover:bg-purple-700 text-white font-bold py-2 px-4 rounded opacity-50 cursor-not-allowed";
        }
        return "bg-purple-500 hover:bg-purple-700 text-white font-bold py-2 px-4 rounded";
    }

    return (
        <form className="w-full" onSubmit={onSubmit} onError={onError}>
            <div>{children}</div>
            <div className="flex flex-row justify-center">
                <button className={getStyle()} type="submit" disabled={disabledSubmit}>
                    Submit
                </button>
            </div>
        </form>
    );
}

export default Form;
