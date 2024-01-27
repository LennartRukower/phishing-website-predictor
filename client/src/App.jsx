import { useEffect, useState } from "react";
import Alert from "./components/general/Alert";
import Card from "./components/general/Card";
import Form from "./components/general/Form";
import ModelCard from "./components/ModelCard";
import ModelInfoPopUp from "./components/ModelInfoPopUp";
import ResultDetailsPopUp from "./components/ResultDetailsPopUp";
import ParallelCoordinatesChart from "./components/general/ParallelCoordinates";

function App() {
    const [models, setModels] = useState([]);
    const [result, setResult] = useState(null);
    const [status, setStatus] = useState("idle");
    const [error, setError] = useState(null);
    const [modelInfoOpen, setModelInfoOpen] = useState(false);
    const [modelInfo, setModelInfo] = useState(null);
    const [resultDetailsOpen, setResultDetailsOpen] = useState(false);

    const [selectedModel, setSelectedModel] = useState(null);
    const [url, setUrl] = useState("");

    const info = {
        ffnn: [
            { key: "input", text: "Number of input nodes", type: "number" },
            { key: "activations", text: "Nubmer of hidden layers", type: "list" },
            { key: "output", text: "Number of output nodes", type: "number" },
        ],
        rf: [
            { key: "nEstimators", text: "Number of estimators", type: "number" },
            { key: "maxDepth", text: "Maximum depth", type: "number" },
            { key: "minSamplesSplit", text: "Minimal sample split", type: "number" },
        ],
    };

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
                    return (
                        <Alert
                            color="red"
                            message={successMessage}
                            showDetails
                            onDetailsClick={() => setResultDetailsOpen(true)}
                        />
                    );
                }
                return (
                    <Alert
                        color="green"
                        message={successMessage}
                        showDetails
                        onDetailsClick={() => setResultDetailsOpen(true)}
                    />
                );
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
        <div>
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
                            handleModelInfoOpen={() => {
                                setModelInfo(mod);
                                setModelInfoOpen(true);
                            }}
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
            <ModelInfoPopUp
                isOpen={modelInfoOpen}
                onClose={() => setModelInfoOpen(false)}
                model={modelInfo}
                infoStructure={info[modelInfo?.name]}
            />
            <ResultDetailsPopUp
                isOpen={resultDetailsOpen}
                onClose={() => setResultDetailsOpen(false)}
                prediction={result?.pred === 1 ? "phishing" : "legitimate"}
                features={result?.features}
                confidence={result?.conf}
                usedModel={result?.model}
            />
        </div>
    );
}

export default App;
