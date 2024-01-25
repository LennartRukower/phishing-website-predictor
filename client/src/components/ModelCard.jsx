import Card from "./general/Card";

function ModelCard({ model, selectedModel, setSelectedModel }) {
    function generateModelName(model) {
        switch (model) {
            case "rf":
                return "Random Forest";
            case "ffnn":
                return "Feed Forward Neural Network";
            default:
                return "Unknown";
        }
    }

    function getStatItem(stat, value) {
        if (value === null) {
            return null;
        }
        return (
            <li>
                {stat}: {value}
            </li>
        );
    }

    return (
        <div className="w-1/4 m-1">
            <Card
                title={generateModelName(model.name)}
                content={
                    <div>
                        <hr className="my-2" />
                        <p className="h-20 overflow-hidden">{model.description}</p>
                        {model.stats ? (
                            <ul className="list-disc list-inside">
                                {getStatItem("Accuracy", model.stats.accuracy)}
                                {getStatItem("Precision", model.stats.precision)}
                                {getStatItem("Recall", model.stats.recall)}
                                {getStatItem("F1", model.stats.f1)}
                            </ul>
                        ) : null}
                    </div>
                }
                clickable
                isSelected={model.name === selectedModel}
                onClick={(event) => {
                    setSelectedModel(model.name);
                }}
            />
        </div>
    );
}

export default ModelCard;
