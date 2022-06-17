//
//  ImageProcessor.swift
//  AboutTime
//
//  Created by Eugene Dorfman on 16/6/22.
//

import Foundation
import Accelerate
import UIKit

public enum ImageProcessor {
    public static func warp(_ img: UIImage) -> UIImage {
        var processedImage: UIImage = img.imageWith(newSize: CGSize(width: 224.0, height: 224.0))
        do {
            //img = img.imageWith(newSize: CGSize(width: CGFloat(224), height: CGFloat(224)))
            let modelstn = try ModelSTN()
            let pred = try modelstn.prediction(input: ModelSTNInput(input: processedImage.buffer()!))
            let result = pred.var_842
            var Minv_pred:[Float] = Array(UnsafeBufferPointer(start: result.dataPointer.assumingMemoryBound(to: Float.self), count: 8))
            //print("Minv_pred=\(Minv_pred)")
            //Minv_pred:  [1.1207, 0.0022, 0.1211, 0.0039, 1.2326, 0.1812, 0.0231, 0.2054]
            Minv_pred.append(1.0)
            processedImage = warp(img: processedImage, Minv_pred: Minv_pred)
        } catch {
            print("\(error)")
        }
        return processedImage
    }

    static func warp(img: UIImage, Minv_pred: [Float], sz: Float = 224) -> UIImage {
        let s = sz/2
        let t:Float = 1.0
        let left:[Float] = [s, 0, t*s, 0, s, t*s, 0, 0, 1]
        let right:[Float] = [1.0/s, 0, -t, 0, 1.0/s, -t, 0, 0, 1]
        var Minv_pred2:[Float] = Array(repeating: 0, count: 9)
        vDSP_mmul(left, 1, Minv_pred, 1, &Minv_pred2, 1, 3, 3, 3)
        vDSP_mmul(Minv_pred2, 1, right, 1, &Minv_pred2, 1, 3, 3, 3)
        //print("Minv_pred2=\(Minv_pred2)")
        /*Minv_pred2: tensor([[[ 1.1438e+00,  2.0757e-01, -2.5782e+01],
                 [ 2.6999e-02,  1.4380e+00, -3.1782e+01],
                 [ 2.0620e-04,  1.8340e-03,  7.7150e-01]]])
        */
        let img_ = warp_perspective(img: img, M: Minv_pred2, dsize: (sz, sz))
        return img_
    }
    
    
    static func warp_perspective(img: UIImage,  M: [Float], dsize: (Float, Float)) -> UIImage {
        let H = Float(img.size.height)
        let W = Float(img.size.width)
        let dst_norm_trans_src_norm = normalize_homography(dst_pix_trans_src_pix: M, dsize_src: (H, W), dsize_dst: dsize)
        //print("dst_norm_trans_src_norm=\(dst_norm_trans_src_norm)")
        //dst_norm_trans_src_norm=tensor([[[1.1208, 0.0031, 0.1211],[0.0040, 1.2335, 0.1810],[0.0230, 0.2045, 0.9990]]])
        
        var src_norm_trans_dst_norm = invert(matrix: dst_norm_trans_src_norm)
        //print("src_norm_trans_dst_norm=\(src_norm_trans_dst_norm)")
        //src_norm_trans_dst_norm=tensor([[[ 8.9447e-01,1.6236e-02,-1.1139e-01],[ 1.1772e-04,8.3581e-01,-1.5142e-01],[-2.0611e-02, -1.7147e-01,  1.0346e+00]]])

        vDSP_mtrans(src_norm_trans_dst_norm, 1, &src_norm_trans_dst_norm, 1, 3, 3)
        
        let filter = WarpFilter(transform: src_norm_trans_dst_norm.map({ CGFloat($0) }))
        filter.inputImage = CIImage(image: img, options: [.applyOrientationProperty: true])
        return UIImage(ciImage:filter.outputImage!, scale: img.scale, orientation: UIImage.Orientation.up)
    }
    
    static func invert(matrix : [Double]) -> [Double] {
        var inMatrix = matrix
        var N = __CLPK_integer(sqrt(Double(matrix.count)))
        var pivots = Array<__CLPK_integer>(repeating: __CLPK_integer(0), count: Int(N))
        var workspace = Array<Double>(repeating: Double(0), count: Int(N))
        var error : __CLPK_integer = 0
        var N1 = __CLPK_integer(N)
        var N2 = __CLPK_integer(N)
        dgetrf_(&N1, &N2, &inMatrix, &N, &pivots, &error)
        dgetri_(&N1, &inMatrix, &N2, &pivots, &workspace, &N, &error)
        return inMatrix
    }

    static func invert(matrix: [Float]) -> [Float] {
        let doublematrix = matrix.map({ Double($0) })
        let inverted = invert(matrix: doublematrix)
        let floatmatrix = inverted.map({ Float($0) })
        return floatmatrix
    }

    static func normal_transform_pixel(height: Float, width: Float, eps: Float = 1e-14) -> [Float] {
        var tr_mat: [[Float]] = [[1.0, 0.0, -1.0], [0.0, 1.0, -1.0], [0.0, 0.0, 1.0]]
        let width_denom: Float = width == 1.0 ? eps : width - 1.0
        let height_denom: Float = height == 1.0 ? eps : height - 1.0
        tr_mat[0][0] *=  2.0 / width_denom
        tr_mat[1][1] *=  2.0 / height_denom
        return tr_mat.flatMap( { $0 } )
    }

    static func normalize_homography(dst_pix_trans_src_pix: [Float], dsize_src: (Float, Float), dsize_dst: (Float, Float)) -> [Float] {
        let (src_h, src_w) = dsize_src
        let (dst_h, dst_w) = dsize_dst
        let src_norm_trans_src_pix = normal_transform_pixel(height: src_h, width: src_w)
        //print("src_norm_trans_src_pix=\(src_norm_trans_src_pix)")
        /*src_norm_trans_src_pix=tensor([[[ 0.0090,  0.0000, -1.0000],[ 0.0000,  0.0090, -1.0000],[ 0.0000,  0.0000,  1.0000]]])*/
        
        let src_pix_trans_src_norm = invert(matrix: src_norm_trans_src_pix)
        //print("src_pix_trans_src_norm=\(src_pix_trans_src_norm)")
        /*src_pix_trans_src_norm=tensor([[[111.5000,  -0.0000, 111.5000],[  0.0000, 111.5000, 111.5000],[0.0000,0.0000,1.0000]]])*/
        
        let dst_norm_trans_dst_pix = normal_transform_pixel(height: dst_h, width: dst_w)
        //print("dst_norm_trans_dst_pix=\(dst_norm_trans_dst_pix)")
        /*dst_norm_trans_dst_pix=tensor([[[ 0.0090,  0.0000, -1.0000],[ 0.0000,  0.0090, -1.0000],[ 0.0000,  0.0000,  1.0000]]])*/
        
        //dst_norm_trans_src_norm: torch.Tensor = dst_norm_trans_dst_pix @ (dst_pix_trans_src_pix @ src_pix_trans_src_norm)
        var dst_norm_trans_src_norm = [Float](repeating: Float(0), count: 9)
        vDSP_mmul(dst_pix_trans_src_pix,1,src_pix_trans_src_norm,1,&dst_norm_trans_src_norm,1,3,3,3)
        vDSP_mmul(dst_norm_trans_dst_pix,1,dst_norm_trans_src_norm,1,&dst_norm_trans_src_norm,1,3,3,3)

        return dst_norm_trans_src_norm
    }


}
