import { useEffect, useState } from "react";
import Alert from "./components/Alert";
import Card from "./components/Card";
import Form from "./components/Form";

function App() {
  const [models, setModels] = useState([]);
  const [result, setResult] = useState(null);
  const [status, setStatus] = useState("idle");
  useEffect(() => {
    fetch("http://127.0.0.1:5000/models")
      .then(res => res.json())
      .then(data => setModels(data))
      .catch(err => setModels(["FNN", "CNN", "RNN"]));
  }
  , []);

  function renderAlert() {
    switch (status) {
      case "idle":
        return null;
      case "pending":
        return <Alert color="blue" message="Pending message"/>;
      case "error":
        return <Alert color="red" message="Error message"/>;
      case "success":
        return <Alert color="green" message="Success message"/>;
      default:
        return null;
    }
  }

  return (
    <div className="mx-auto w-2/4">
        <Card title="Title" content={<Form action=" http://127.0.0.1:5000/models">
          <input type="text" placeholder="Enter your name" className="border border-gray-400 p-2 w-full"/>
          <select className="border border-gray-400 p-2 w-full">
            {models.map(model => <option value={model}>{model}</option>)}
          </select>
        </Form>} />

        <br/>
        {renderAlert()}
    </div>
  );
}

export default App;
