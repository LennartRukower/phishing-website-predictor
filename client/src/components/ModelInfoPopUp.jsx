import { mapModelName } from "../utils/mapperUtils";
import PopUp from "./general/PopUp";

function ModelInfoPopUp({ isOpen, onClose, model, infoStructure }) {
    const getInfoItemValue = (key) => {
        const item = model.info[key];
        const structure = infoStructure.find((item) => item.key === key);
        if (item) {
            return structure.type === "list" ? item.length : item !== -1 ? item : null;
        }
        return null;
    };

    return model ? (
        <PopUp isOpen={isOpen} onClose={onClose}>
            <h1 className="text-2xl font-bold text-blue-2 mb-4">{mapModelName(model.name)}</h1>
            <p className="text-lg text-blue-2">{model.description}</p>
            <br />
            {infoStructure ? (
                <div>
                    <p className="text-lg text-blue-2">Stats</p>
                    <ul className="list-disc list-inside">
                        {infoStructure.map((stat) => (
                            <li key={stat.key}>
                                {stat.text}: {getInfoItemValue(stat.key) || "N/A"}
                            </li>
                        ))}
                    </ul>
                </div>
            ) : null}
        </PopUp>
    ) : null;
}

export default ModelInfoPopUp;
