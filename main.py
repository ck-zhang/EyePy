import json
import src.data_processing.collect_data as collect_data
import src.training.train as train
import src.gaze_prediction.predict_gaze as predict_gaze


def main():
    with open("options.json", "r") as f:
        options = json.load(f)

    collect_data.collect_data(camera_index=options.get("camera_index", 1))

    train.train(
        alpha=options.get("alpha", 1.0),
        plot_graphs=options.get("plot_graphs", False),
        feature_scales=options.get("feature_scales", {}),
    )

    predict_gaze.predict_gaze(
        do_kde=options.get("do_kde", True),
        do_accuracy_test=options.get("do_accuracy_test", False),
        use_kalman_filter=options.get("use_kalman_filter", True),
        center_neon_circle=options.get("center_neon_circle", False),
        feature_scales=options.get("feature_scales", {}),
    )


if __name__ == "__main__":
    main()
