export function mapModelName(model) {
    switch (model) {
        case "rf":
            return "Random Forest";
        case "ffnn":
            return "Feed Forward Neural Network";
        default:
            return "Unknown";
    }
}