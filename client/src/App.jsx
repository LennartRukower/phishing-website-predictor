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
        return (
            <p>
                The url is a <strong>{prediction}</strong> website!
            </p>
        );
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
                    return <Alert color="red" message={successMessage} showDetails />;
                }
                return <Alert color="green" message={successMessage} showDetails />;
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
        <div className="mx-auto w-1/2 mt-20">
            <div className="flex flex-row justify-center">
                <p className="text-gray-400">Select a model</p>
            </div>
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
            <div className="flex flex-row justify-center">
                <p className="text-gray-400">Enter a URL</p>
            </div>
            <div className="flex justify-center">
                <div className="w-2/3">
                    <Card
                        content={
                            <div className="flex flex-row justify-center w-full">
                                <Form
                                    onSubmit={handleSubmit}
                                    disabledSubmit={selectedModel === null || url === ""}
                                >
                                    <input
                                        type="text"
                                        placeholder="URL"
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
            <div className="flex flex-row justify-center">
                <p className="text-gray-400">The result is displayed here</p>
            </div>
            <div className="flex justify-center">
                <div className="w-2/3">{renderAlert()}</div>
            </div>
        </div>
    );
}

export default App;
