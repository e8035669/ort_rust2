use std::error::Error;

use cv_convert::TryFromCv;
use itertools::izip;
use ndarray::prelude::{s, Array, Axis, Ix3, Ix4};
use ndarray_stats::QuantileExt;
use opencv::core as cv;
use opencv::dnn::nms_boxes_batched;
use opencv::highgui::{imshow, named_window, wait_key, WINDOW_NORMAL};
use opencv::imgproc::{
    cvt_color, put_text, rectangle, resize, COLOR_BGR2RGB, FONT_HERSHEY_SIMPLEX, INTER_CUBIC,
    LINE_8,
};
use opencv::prelude::*;
use opencv::videoio::{VideoCapture, CAP_ANY};
use ort::{inputs, CUDAExecutionProvider, GraphOptimizationLevel::Level3, Session};

#[rustfmt::skip]
const YOLOV8_CLASS_LABELS: [&str; 80] = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
	"fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant",
	"bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
	"sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
	"wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
	"carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet",
	"tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
	"book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
];

fn main() -> Result<(), Box<dyn Error>> {
    let model = Session::builder()?
        .with_intra_threads(4)?
        .with_execution_providers([CUDAExecutionProvider::default().build()])?
        .with_optimization_level(Level3)?
        .with_model_from_file("yolov8m.onnx")?;

    let mut cap = VideoCapture::new(0, CAP_ANY)?;
    named_window("Window", WINDOW_NORMAL)?;

    while cap.is_opened()? {
        let mut orig_img = Mat::default();
        let ret = cap.read(&mut orig_img)?;
        if !ret {
            break;
        }

        let imgsz = orig_img.size()?;
        let mut resized = Mat::default();
        resize(
            &orig_img,
            &mut resized,
            (640, 640).into(),
            0.,
            0.,
            INTER_CUBIC,
        )?;
        let mut rgb_img = Mat::default();
        cvt_color(&resized, &mut rgb_img, COLOR_BGR2RGB, 0)?;

        let tensor = Array::<u8, Ix3>::try_from_cv(&rgb_img)?;
        let tensor = tensor.mapv(|x| f32::from(x) / 255.0);
        let tensor: Array<f32, Ix3> = tensor.permuted_axes([2, 0, 1]);
        let tensor: Array<f32, Ix4> = tensor.insert_axis(Axis(0)).to_owned();

        let outputs = model.run(inputs!["images"=>tensor.view()]?)?;

        let output = outputs["output0"].extract_tensor::<f32>()?;
        let output = output.view().t().remove_axis(Axis(2)).to_owned();

        println!("output shape: {:?}", output.shape());
        let mut bboxes: cv::Vector<cv::Rect> = cv::Vector::new();
        let mut scores: cv::Vector<f32> = cv::Vector::new();
        let mut classes: cv::Vector<i32> = cv::Vector::new();
        let mut indices: cv::Vector<i32> = cv::Vector::new();

        for row in output.axis_iter(Axis(0)) {
            let confidences = row.slice(s![4..]);
            let cls = confidences.argmax()?;
            let conf = confidences[[cls]];
            if conf > 0.5 {
                let xc = row[0] / 640. * (imgsz.width as f32);
                let yc = row[1] / 640. * (imgsz.height as f32);
                let w = row[2] / 640. * (imgsz.width as f32);
                let h = row[3] / 640. * (imgsz.height as f32);
                let x = xc - w / 2.;
                let y = yc - h / 2.;

                bboxes.push(cv::Rect::new(x as i32, y as i32, w as i32, h as i32));
                scores.push(conf);
                classes.push(cls as i32);
            }
        }

        println!("{:?}", bboxes);
        println!("{:?}", scores);
        println!("{:?}", classes);

        nms_boxes_batched(&bboxes, &scores, &classes, 0.5, 0.5, &mut indices, 1.0, 0)?;

        let bboxes: Vec<cv::Rect> = indices
            .iter()
            .map(|i| bboxes.get(i as _).unwrap().clone())
            .collect();
        let scores: Vec<f32> = indices
            .iter()
            .map(|i| scores.get(i as _).unwrap().clone())
            .collect();
        let classes: Vec<i32> = indices
            .iter()
            .map(|i| classes.get(i as _).unwrap().clone())
            .collect();

        for (b, s, c) in izip!(&bboxes, &scores, &classes) {
            println!("{:?}, {}, {}", b, s, c);
        }

        let mut drawed = orig_img.clone();
        for (b, s, c) in izip!(&bboxes, &scores, &classes) {
            rectangle(
                &mut drawed,
                b.clone(),
                (0., 0., 255., 0.).into(),
                2,
                LINE_8,
                0,
            )?;

            let cls_name = YOLOV8_CLASS_LABELS[c.to_owned() as usize];
            let pt = b.tl() + cv::Point::new(0, -10);
            let text = format!("{}: {:.2}", cls_name, s);
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
        if wait_key(1)? == 27 {
            break;
        }
    }

    Ok(())
}
