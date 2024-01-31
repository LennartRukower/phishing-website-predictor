import { mapModelName } from "../utils/mapperUtils";
import ParallelCoordinatesChart from "./general/ParallelCoordinates";
import PopUp from "./general/PopUp";

function ResultDetailsPopUp({
    isOpen,
    onClose,
    prediction,
    modelResults,
    features,
    usedModels,
    trainingData,
}) {
    function convertValuesToReadable(value) {
        if (value === false) return "No";
        if (value === true) return "Yes";
        return value;
    }

    function toPercents(value) {
        return `${(value * 100).toFixed(6)}%`;
    }

    function getModelNames() {
        // If only one model was used, return its name if not return the names of all models separated by & or ,
        if (usedModels.length === 1) return mapModelName(usedModels[0]);
        return usedModels.reduce((acc, model, index) => {
            if (index === usedModels.length - 1) {
                return `${acc} & ${mapModelName(model)}`;
            }
            if (index === 0) {
                return `${mapModelName(model)}`;
            }
            return `${acc}, ${mapModelName(model)}`;
        }, "");
    }

    return prediction && features ? (
        <PopUp isOpen={isOpen} onClose={onClose}>
            <h1 className="text-2xl font-bold mb-4">This website is a {prediction} website!</h1>
            <p className="text-lg">
                {modelResults.map((res, index) => (
                    <p key={res.model}>
                        The {mapModelName(res.model)} model is{" "}
                        <strong>{toPercents(res.conf)}</strong> confident that this website is a{" "}
                        <strong>{res.pred === 1 ? "phishing" : "legitimate"}</strong> website.
                    </p>
                ))}
            </p>
            <br />
            <hr className="border-1 border-gray-400" />
            <br />
            <p>
                The following chart shows the training data (10% of data labeled legitimate & 10% of
                data labeled phishing) used to train the{" "}
                {usedModels.length > 1 ? "models" : "model"}. The green line represents the features
                of the provided url.
            </p>
            <ParallelCoordinatesChart
                trainingData={trainingData}
                currentData={{ ...features, label: prediction }}
            />
            <br />
            <hr className="border-1 border-gray-400" />
            <br />
            <p>
                The following information was extracted from the url and the html code of the
                website:
            </p>
            <div className="w-full flex justify-center ">
                <div className="w-1/2">
                    {Object.entries(features).map(([key, value]) => (
                        <div key={key} className="flex flex-row justify-between">
                            <p className="text-lg">{key}</p>
                            <p className="text-lg">{convertValuesToReadable(value)}</p>
                        </div>
                    ))}
                </div>
            </div>
            <br />
        </PopUp>
    ) : null;
}

export default ResultDetailsPopUp;
