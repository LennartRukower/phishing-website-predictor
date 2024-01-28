import { mapModelName } from "../utils/mapperUtils";
import ParallelCoordinatesChart from "./general/ParallelCoordinates";
import PopUp from "./general/PopUp";

function ResultDetailsPopUp({
    isOpen,
    onClose,
    prediction,
    confidence,
    features,
    usedModel,
    trainingData,
}) {
    function convertValuesToReadable(value) {
        if (value === false) return "No";
        if (value === true) return "Yes";
        return value;
    }

    function toPercents(value) {
        console.log(value);
        return `${(value * 100).toFixed(6)}%`;
    }

    return prediction && features ? (
        <PopUp isOpen={isOpen} onClose={onClose}>
            <h1 className="text-2xl font-bold mb-4">
                This website is a <strong>{prediction}</strong> website!
            </h1>
            <p className="text-lg">
                The {mapModelName(usedModel)} model is <strong>{toPercents(confidence)}</strong>{" "}
                confident that the provided url belongs to a <strong>{prediction}</strong> website.
            </p>
            <p>
                The following information was extracted from the url and the html code of the
                website:
            </p>
            <br />
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
            <p>
                The following chart shows the training data (10% of data labeled legitimate & 10% of
                data labeled phishing) used to train the {mapModelName(usedModel)} model. The green
                line represents the features of the provided url.
            </p>
            <ParallelCoordinatesChart
                trainingData={trainingData}
                currentData={{ ...features, label: prediction }}
            />
        </PopUp>
    ) : null;
}

export default ResultDetailsPopUp;
