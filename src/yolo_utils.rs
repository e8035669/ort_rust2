use std::error::Error;
use std::path::{Path, PathBuf};

use anyhow::anyhow;
use cv_convert::TryFromCv;
use itertools::izip;
use ndarray::{s, Array, Axis, Ix2, Ix3, Ix4};
use ndarray_stats::QuantileExt;
use opencv::core as cv;
use opencv::dnn::nms_boxes_batched;
use opencv::imgproc::{cvt_color, resize, COLOR_BGR2RGB, INTER_CUBIC};
use opencv::prelude::*;
use ort::{inputs, CUDAExecutionProvider, GraphOptimizationLevel::Level3, Session};

pub struct PreprocessInfo {
    pub imgsz: cv::Size,
    pub tensor: Array<f32, Ix4>,
}

pub fn preprocess(bgr_img: &impl cv::ToInputArray) -> Result<PreprocessInfo, Box<dyn Error>> {
    let bgr_img = bgr_img.input_array()?.get_mat(-1)?;
    let imgsz = bgr_img.size()?;
    let mut resized = cv::Mat::default();
    resize(
        &bgr_img,
        &mut resized,
        (640, 640).into(),
        0.,
        0.,
        INTER_CUBIC,
    )?;
    let mut rgb_img = cv::Mat::default();
    cvt_color(&resized, &mut rgb_img, COLOR_BGR2RGB, 0)?;
    let tensor: Array<u8, Ix3> = Array::try_from_cv(&rgb_img)?;
    let tensor: Array<f32, Ix3> = tensor.mapv(|x| f32::from(x) / 255.0);
    let tensor: Array<f32, Ix3> = tensor.permuted_axes([2, 0, 1]);
    let tensor: Array<f32, Ix4> = tensor.insert_axis(Axis(0)).to_owned();

    Ok(PreprocessInfo { imgsz, tensor })
}

pub struct PredictResult {
    pub bbox: cv::Rect,
    pub score: f32,
    pub cls: i32,
}

impl PredictResult {
    pub fn new(bbox: cv::Rect, score: f32, cls: i32) -> Self {
        Self { bbox, score, cls }
    }
}

pub fn postprocess(
    output: &Array<f32, Ix2>,
    imgsz: &cv::Size,
) -> Result<Vec<PredictResult>, Box<dyn Error>> {
    let mut bboxes = cv::Vector::<cv::Rect>::new();
    let mut scores = cv::Vector::<f32>::new();
    let mut classes = cv::Vector::<i32>::new();
    let mut indices = cv::Vector::<i32>::new();

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

    let result: Vec<_> = izip!(&bboxes, &scores, &classes)
        .into_iter()
        .map(|(b, s, c)| PredictResult::new(b.to_owned(), s.to_owned(), c.to_owned()))
        .collect();

    Ok(result)
}

#[rustfmt::skip]
pub const YOLOV8_CLASS_LABELS: [&str; 80] = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
	"fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant",
	"bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
	"sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
	"wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
	"carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet",
	"tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
	"book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
];

pub struct Yolov8 {
    pub sess: Session,
}

pub struct Yolov8Builder {
    model_path: Option<PathBuf>,
}

impl Yolov8 {
    pub fn builder() -> Yolov8Builder {
        Yolov8Builder { model_path: None }
    }

    pub fn detect(
        &self,
        img: &impl cv::ToInputArray,
    ) -> Result<Vec<PredictResult>, Box<dyn Error>> {
        let info = preprocess(img)?;
        let tensor = info.tensor.view();
        let outputs = self.sess.run(inputs!["images" => tensor]?)?;
        let output = outputs["output0"].extract_tensor::<f32>()?;
        // (1, 84, N) -> (N, 84, 1) -> (N, 84)
        let output: Array<f32, Ix2> = output
            .view()
            .t()
            .index_axis(Axis(2), 0)
            .into_dimensionality()?
            .to_owned();
        let pred_ret = postprocess(&output, &info.imgsz)?;

        Ok(pred_ret)
    }

    pub fn forward(&self, info: &PreprocessInfo) -> Result<Array<f32, Ix2>, Box<dyn Error>> {
        let tensor = info.tensor.view();
        let outputs = self.sess.run(inputs!["images" => tensor]?)?;
        let output = outputs["output0"].extract_tensor::<f32>()?;
        let output = output
            .view()
            .t()
            .index_axis(Axis(2), 0)
            .into_dimensionality()?
            .to_owned();
        Ok(output)
    }
}

impl Yolov8Builder {
    pub fn with_model_path<P>(mut self, path: P) -> Self
    where
        P: AsRef<Path>,
    {
        self.model_path = Some(path.as_ref().to_path_buf());
        self
    }

    pub fn build(self) -> Result<Yolov8, Box<dyn Error>> {
        let sess = Session::builder()?
            .with_intra_threads(4)?
            .with_inter_threads(4)?
            .with_execution_providers([CUDAExecutionProvider::default().build()])?
            .with_optimization_level(Level3)?
            .with_model_from_file(
                self.model_path
                    .ok_or(anyhow!("model_path is not set"))?
                    .as_path(),
            )?;
        Ok(Yolov8 { sess })
    }
}
