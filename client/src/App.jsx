import { useEffect, useState } from "react";
import Alert from "./components/general/Alert";
import Card from "./components/general/Card";
import Form from "./components/general/Form";
import ModelCard from "./components/ModelCard";

function App() {
    const [models, setModels] = useState([]);
    const [result, setResult] = useState(null);
    const [status, setStatus] = useState("idle");
    const [error, setError] = useState(null);

    const [selectedModel, setSelectedModel] = useState(null);
    const [url, setUrl] = useState("");

    useEffect(() => {
        fetch("http://127.0.0.1:5000/models")
            .then((res) => res.json())
            .then((data) => {
                if (data.error) throw Error(data.error);
                setModels(data);
            })
            .catch((err) => console.log(err));
    }, []);

    function generateSuccessMessage() {
        const prediction = result.pred === 1 ? "phishing" : "legitimate";
        return `The url is a ${prediction} website!`;
    }

    function renderAlert() {
        switch (status) {
            case "idle":
                return null;
            case "pending":
                return <Alert color="blue" message="Loading model results..." />;
            case "error":
                return <Alert color="orange" message={error} />;
            case "success":
                const successMessage = generateSuccessMessage();
                if (result.pred === 1) {
                    return <Alert color="red" message={successMessage} />;
                }
                return <Alert color="green" message={successMessage} />;
            default:
                return null;
        }
    }

    function handleSubmit(event) {
        event.preventDefault();
        setStatus("pending");
        fetch("http://127.0.0.1:5000/predict", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                url: url,
                model: selectedModel,
            }),
        })
            .then((res) => res.json())
            .then((data) => {
                if (data.error) throw Error(data.error);
                setResult(data);
                setStatus("success");
            })
            .catch((err) => {
                setStatus("error");
                setError(err.message);
            });
    }

    return (
        <div className="mx-auto w-2/4 mt-20">
            {/* TODO: Add grid and justify center the cards */}
            <div className="flex flex-row justify-center">
                {models.map((mod) => (
                    <ModelCard
                        key={mod.name}
                        model={mod}
                        selectedModel={selectedModel}
                        setSelectedModel={setSelectedModel}
                    />
                ))}
            </div>
            <br />
            <div className="flex justify-center">
                <div className="w-1/2">
                    <Card
                        title="Phishing URL Predictor"
                        content={
                            <div className="flex flex-row justify-center w-full">
                                <Form
                                    onSubmit={handleSubmit}
                                    disabledSubmit={selectedModel === null || url === ""}
                                >
                                    <input
                                        type="text"
                                        placeholder="Enter a url"
                                        className="border border-gray-400 p-2 w-full rounded mb-1"
                                        onChange={(event) => setUrl(event.target.value)}
                                    />
                                </Form>
                            </div>
                        }
                    />
                </div>
            </div>
            <br />
            {renderAlert()}
        </div>
    );
}

export default App;
