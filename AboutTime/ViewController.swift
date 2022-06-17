//
//  ViewController.swift
//  AboutTime
//
//  Created by Eugene Dorfman on 15/6/22.
//

import UIKit
import CoreML
import Vision
import CoreGraphics

class ViewController: UIViewController, UIImagePickerControllerDelegate, UINavigationControllerDelegate {
    @IBOutlet weak var imageView1: UIImageView!
    @IBOutlet weak var imageView2: UIImageView!
    @IBOutlet weak var timeLabel: UILabel!
    var imgPicker = UIImagePickerController()
    var queue = DispatchQueue(label: "background")
    var animate = false
    
    @IBAction func getWatchImageCamera(_ sender: Any) {
        presentImagePicker(sourceType: .camera)
    }
    
    @IBAction func getWatchImageLibrary(_ sender: Any) {
        presentImagePicker(sourceType: .photoLibrary)
    }
    
    func presentImagePicker(sourceType: UIImagePickerController.SourceType) {
        self.imgPicker.sourceType = sourceType
        self.imgPicker.delegate = self
        self.imgPicker.allowsEditing = true
        self.present(self.imgPicker, animated: true)
    }
    
    //MARK:UIImagePickerControllerDelegate
    func imagePickerController(_ picker: UIImagePickerController, didFinishPickingMediaWithInfo info: [UIImagePickerController.InfoKey : Any]) {
        var image : UIImage!

        if let img = info[UIImagePickerController.InfoKey.editedImage] as? UIImage {
            image = img
        } else if let img = info[UIImagePickerController.InfoKey.originalImage] as? UIImage {
            image = img
        }
        
        picker.dismiss(animated: true,completion: {[weak self] in
            self?.processImage(image)
        })
    }

    func processImage(_ img: UIImage) {
        self.imageView2.alpha = 0
        self.imageView1.image = img
        self.imageView1.alpha = 1
        
        let format: (Int) -> String  = { (n: Int) -> String in
            return String(format: "%02d", n)
        }
        
        UIView.animate(withDuration: 0.4, delay: 0, options: [.repeat, .autoreverse]) {
            self.imageView1.alpha = 0
            self.timeLabel.alpha = 0
        }
        
        queue.async {
            let processedImage = ImageProcessor.warp(img)
            var timeText: String?
            do {
                let model = try Model()
                let pred = try model.prediction(input: ModelInput(input: processedImage.buffer()!))
                let result = pred.var_842.argmax()
                
                let h: Int = result / 60
                let m: Int = result % 60
                timeText = "\(format(h)):\(format(m))"
            } catch {
                
            }
            DispatchQueue.main.async {
                self.imageView2.image = processedImage
                self.imageView1.layer.removeAllAnimations()
                self.timeLabel.layer.removeAllAnimations()
                self.timeLabel.text = timeText
                
                UIView.animate(withDuration: 0.5) { [weak self] in
                    self?.imageView1.alpha = 0
                    self?.imageView2.alpha = 1
                    self?.timeLabel.alpha = 1
                }
            }
        }
    }
}

