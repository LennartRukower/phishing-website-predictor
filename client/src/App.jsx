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
      .catch(err => setModels(["FNN", "SVM"]));
  }, []);

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

  function handleSubmit(event) {
    event.preventDefault();
    setStatus("pending");
    fetch("http://127.0.0.1:5000/models", {
      method: "POST",
      body: JSON.stringify({
        name: "",
        model: ""
      })
    })
      .then(res => res.json())
      .then(data => {
        setResult(data);
        setStatus("success");
      })
      .catch(err => setStatus("error"));
  }

  return (
    <div className="mx-auto w-2/4 mt-20">
        <Card title="Phishing URL Predictor" content={
          <Form onSubmit={handleSubmit}>
            <input type="text" placeholder="Enter a url" className="border border-gray-400 p-2 w-full rounded mb-1"/>
            <select className="border border-gray-400 p-2 w-full rounded mb-1">
              {models.map(model => <option value={model}>{model}</option>)}
            </select>
          </Form>
        } />
        <br/>
        {renderAlert()}
    </div>
  );
}

export default App;
