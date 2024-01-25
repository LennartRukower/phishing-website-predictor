// Form renders a form element which renders given children and makes a given action when submitted

import React from 'react';

function Form({children, onSubmit, onError}) {
    return (
        <form onSubmit={onSubmit} onError={onError}>
            {children}
            <button className="bg-purple-500 hover:bg-purple-700 text-white font-bold py-2 px-4 rounded" type="submit">Submit</button>
        </form>
    );
}

export default Form;