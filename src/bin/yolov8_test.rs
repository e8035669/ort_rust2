use std::error::Error;

use opencv::core::Point;
use opencv::highgui::{imshow, wait_key};
use opencv::imgcodecs::{imread, IMREAD_COLOR};
use opencv::imgproc::{put_text, rectangle, FONT_HERSHEY_SIMPLEX, LINE_8};
use ort_rust2::yolo_utils::{self, Yolov8, YOLOV8_CLASS_LABELS};

fn main() -> Result<(), Box<dyn Error>> {
    let orig_img = imread("./baseball.jpg", IMREAD_COLOR)?;
    let model = Yolov8::builder().with_model_path("yolov8m.onnx").build()?;

    let preprocess = yolo_utils::preprocess(&orig_img)?;
    let output = model.forward(&preprocess)?;
    let result = yolo_utils::postprocess(&output, &preprocess.imgsz)?;

    let mut drawed = orig_img.clone();
    for bbox in result.iter() {
        rectangle(
            &mut drawed,
            bbox.bbox.clone(),
            (0., 0., 255., 0.).into(),
            2,
            LINE_8,
            0,
        )?;
        let cls_name = YOLOV8_CLASS_LABELS[bbox.cls as usize];
        let b = bbox.bbox.clone();
        let pt = b.tl() + Point::new(0, -10);
        let text = format!("{}: {:.2}", cls_name, bbox.score);
        put_text(
            &mut drawed,
            text.as_str(),
            pt,
            FONT_HERSHEY_SIMPLEX,
            0.7,
            (0., 0., 255., 0.).into(),
            2,
            LINE_8,
            false,
        )?;
    }

    imshow("Window", &drawed)?;
    wait_key(0)?;

    Ok(())
}
