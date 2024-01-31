export function mapModelName(model) {
    switch (model) {
        case "rf":
            return "Random Forest";
        case "ffnn":
            return "Feed Forward Neural Network";
        case "svm":
            return "Support Vector Machine";
        default:
            return "Unknown";
    }
}