use std::error::Error;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc;
use std::sync::Arc;
use std::thread;

use ndarray::{Array, Ix2};
use opencv::core::{Mat, Point};
use opencv::highgui::{imshow, wait_key};
use opencv::imgcodecs::{imread, IMREAD_COLOR};
use opencv::imgproc::{put_text, rectangle, FONT_HERSHEY_SIMPLEX, LINE_8};
use opencv::prelude::*;
use opencv::videoio::{VideoCapture, CAP_ANY};
use ort_rust2::yolo_utils::{self, Yolov8, YOLOV8_CLASS_LABELS};

struct ImageSender {
    cap: VideoCapture,
    tx: mpsc::Sender<Mat>,
    should_stop: Arc<AtomicBool>,
}

impl ImageSender {
    fn new(cap: VideoCapture, tx: mpsc::Sender<Mat>, should_stop: Arc<AtomicBool>) -> Self {
        Self {
            cap,
            tx,
            should_stop,
        }
    }
    fn run(&mut self) {
        loop {
            if self.should_stop.load(Ordering::Relaxed) {
                break;
            }
            let mut img = Mat::default();
            let ret = self.cap.read(&mut img);
            if let Err(e) = ret {
                println!("Error {:?}", e);
                break;
            }
            let ret = ret.unwrap();
            if !ret {
                println!("VideoCapture closed");
                break;
            }
            let ret = self.tx.send(img);
            if let Err(e) = ret {
                println!("Send error {:?}", e);
                break;
            }
        }
        let ret = self.cap.release();
        if let Err(e) = ret {
            println!("Error {:?}", e);
        }
    }
}

type Preprocessed = (Mat, yolo_utils::PreprocessInfo);
type Predicted = (Mat, yolo_utils::PreprocessInfo, Array<f32, Ix2>);

struct Preprocesser {
    rx: mpsc::Receiver<Mat>,
    tx: mpsc::Sender<Preprocessed>,
}

impl Preprocesser {
    fn new(rx: mpsc::Receiver<Mat>, tx: mpsc::Sender<Preprocessed>) -> Self {
        Self { rx, tx }
    }

    fn run(&mut self) {
        for img in self.rx.iter() {
            let prep = yolo_utils::preprocess(&img);
            let prep = match prep {
                Err(e) => {
                    println!("Error on preprocess {:?}", e);
                    break;
                }

                Ok(prep) => prep,
            };

            let ret = self.tx.send((img, prep));
            if let Err(e) = ret {
                println!("Error on send {:?}", e);
                break;
            }
        }
    }
}

struct ModelInference {
    model: Yolov8,
    rx: mpsc::Receiver<Preprocessed>,
    tx: mpsc::Sender<Predicted>,
}

impl ModelInference {
    fn new(model: Yolov8, rx: mpsc::Receiver<Preprocessed>, tx: mpsc::Sender<Predicted>) -> Self {
        Self { model, rx, tx }
    }

    fn run(&mut self) {
        for (img, info) in self.rx.iter() {
            let output = self.model.forward(&info);
            let output = match output {
                Err(e) => {
                    println!("Error forward {:?}", e);
                    break;
                }
                Ok(o) => o,
            };

            let ret = self.tx.send((img, info, output));
            if let Err(e) = ret {
                println!("Error send {:?}", e);
                break;
            }
        }
    }
}

struct ImageShow {
    rx: mpsc::Receiver<Predicted>,
    should_stop: Arc<AtomicBool>,
}

impl ImageShow {
    fn new(rx: mpsc::Receiver<Predicted>, should_stop: Arc<AtomicBool>) -> Self {
        Self { rx, should_stop }
    }
    fn run(&mut self) {
        for (img, info, output) in self.rx.iter() {
            let pred_ret = yolo_utils::postprocess(&output, &info.imgsz);
            let result = match pred_ret {
                Err(e) => {
                    println!("Error postprocess {:?}", e);
                    break;
                }
                Ok(p) => p,
            };
            let mut drawed = img.clone();
            for bbox in result.iter() {
                let ret = rectangle(
                    &mut drawed,
                    bbox.bbox.clone(),
                    (0., 0., 255., 0.).into(),
                    2,
                    LINE_8,
                    0,
                );
                if let Err(e) = ret {
                    println!("rectangle {:?}", e);
                    break;
                }
                let cls_name = YOLOV8_CLASS_LABELS[bbox.cls as usize];
                let b = bbox.bbox.clone();
                let pt = b.tl() + Point::new(0, -10);
                let text = format!("{}: {:.2}", cls_name, bbox.score);
                let ret = put_text(
                    &mut drawed,
                    text.as_str(),
                    pt,
                    FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0., 0., 255., 0.).into(),
                    2,
                    LINE_8,
                    false,
                );
                if let Err(e) = ret {
                    println!("put_text {:?}", e);
                    break;
                }
            }
            let ret = imshow("Window", &drawed);
            if let Err(e) = ret {
                println!("imshow {:?}", e);
                break;
            }
            let ret = wait_key(1);
            let ret = match ret {
                Err(e) => {
                    println!("wait_key {:?}", e);
                    break;
                }
                Ok(i) => i,
            };
            if ret == 27 {
                println!("Send should_stop");
                self.should_stop.store(true, Ordering::Relaxed);
            }
        }

        self.should_stop.store(true, Ordering::Relaxed);
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let mut cap = VideoCapture::default()?;
    cap.open(0, CAP_ANY)?;

    let model = Yolov8::builder().with_model_path("yolov8m.onnx").build()?;

    let should_stop = Arc::new(AtomicBool::new(false));
    let (tx1, rx1) = mpsc::channel();

    let should_stop_clone = Arc::clone(&should_stop);
    let thread1 = thread::spawn(move || {
        let mut image_sender = ImageSender::new(cap, tx1, should_stop_clone);
        image_sender.run();
    });

    let (tx2, rx2) = mpsc::channel();
    let thread2 = thread::spawn(move || {
        let mut preprocesser = Preprocesser::new(rx1, tx2);
        preprocesser.run();
    });

    let (tx3, rx3) = mpsc::channel();
    let thread3 = thread::spawn(move || {
        let mut model_inference = ModelInference::new(model, rx2, tx3);
        model_inference.run();
    });

    let should_stop_clone = Arc::clone(&should_stop);
    let thread4 = thread::spawn(move || {
        let mut image_show = ImageShow::new(rx3, should_stop_clone);
        image_show.run();
    });

    let ret1 = thread1.join();
    if let Err(e) = ret1 {
        println!("Err thread1 {:?}", e);
    }
    let ret2 = thread2.join();
    if let Err(e) = ret2 {
        println!("Err thread2 {:?}", e);
    }
    let ret3 = thread3.join();
    if let Err(e) = ret3 {
        println!("Err thread3 {:?}", e);
    }
    let ret4 = thread4.join();
    if let Err(e) = ret4 {
        println!("Err thread4 {:?}", e);
    }

    Ok(())
}
