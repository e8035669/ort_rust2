use std::error::Error;
use std::path::{Path, PathBuf};

use anyhow::anyhow;
use cv_convert::TryFromCv;
use ndarray::{Array, Axis, Ix3, Ix4};
use opencv::core as cv;
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

pub fn postprocess() {}

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

    pub fn detect(&self, img: &impl cv::ToInputArray) -> Result<(), Box<dyn Error>> {
        let info = preprocess(img).unwrap();
        let tensor = info.tensor.view();
        let outputs = self.sess.run(inputs!["images" => tensor]?)?;
        let output = outputs["output0"].extract_tensor::<f32>()?;
        // (1, 84, N) -> (N, 84, 1) -> (N, 84)
        let output = output.view().t().index_axis(Axis(2), 0).to_owned();

        Ok(())
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
