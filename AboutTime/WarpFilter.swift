import Foundation
import CoreImage

public class WarpFilter: CIFilter {
    private let kernel: CIWarpKernel
    public var inputImage: CIImage?
    var transform: [CGFloat] //transform matrix in row major order
    
    public init(transform: [CGFloat]) {
        print(Bundle.main)
        let url = Bundle.main.url(forResource: "default", withExtension: "metallib")!
        let data = try! Data(contentsOf: url)
        kernel = try! CIWarpKernel(functionName: "warp", fromMetalLibraryData: data)
        self.transform = transform
        super.init()
    }
    
    required init?(coder: NSCoder) {
        fatalError("init(coder:) has not been implemented")
    }
    
    public override var outputImage: CIImage? {
        guard let inputImage = self.inputImage else { return nil }
        let inputExtent = inputImage.extent

        let roiCallback: CIKernelROICallback = { _, rect -> CGRect in
            return rect
        }
        let vec = CIVector(values: transform, count: transform.count)
        return self.kernel.apply(extent: inputExtent, roiCallback: roiCallback, image: inputImage, arguments: [vec, inputImage.extent.size.width])
    }
}
