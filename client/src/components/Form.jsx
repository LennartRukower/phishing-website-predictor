// Form renders a form element which renders given children and makes a given action when submitted

import React from 'react';

function Form({children, onSubmit, onError}) {
    return (
        <form onSubmit={onSubmit} onError={onError}>
            {children}
            <button className="bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded" type="submit">Submit</button>
        </form>
    );
}

export default Form;