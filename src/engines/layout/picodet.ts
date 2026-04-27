/**
 * PicoDet layout detection engine using ONNX Runtime.
 *
 * Runs PaddlePaddle's PicoDet layout model on CPU via onnxruntime-node.
 * Detects document layout regions: text, title, list, table, figure, etc.
 */

import sharp from "sharp";
import { LayoutEngine, LayoutRegion, LayoutLabel } from "./interface.js";
import { getModelPath, getModelMeta } from "./model.js";
import { decodePostProcessedOutputs, decodePicoDetOutputs, nms } from "./postprocess.js";

// Lazy import for onnxruntime-node (optional dependency)
let ort: typeof import("onnxruntime-node") | null = null;

async function getOrt() {
  if (!ort) {
    try {
      ort = await import("onnxruntime-node");
    } catch {
      throw new Error(
        "onnxruntime-node is required for layout detection. Install it:\n" +
          "  npm install onnxruntime-node"
      );
    }
  }
  return ort;
}

/** ImageNet normalization constants. */
const IMAGENET_MEAN = [0.485, 0.456, 0.406];
const IMAGENET_STD = [0.229, 0.224, 0.225];

export interface PicoDetOptions {
  /** Path to ONNX model file, or a registered model name. */
  modelPath?: string;
  /** Confidence score threshold for detections. */
  scoreThreshold?: number;
  /** IoU threshold for NMS. */
  nmsThreshold?: number;
  /** Maximum detections to keep after NMS. */
  maxDetections?: number;
}

export class PicoDetEngine implements LayoutEngine {
  name = "PicoDet";

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  private session: any = null;
  private modelPath?: string;
  private scoreThreshold: number;
  private nmsThreshold: number;
  private maxDetections: number;
  private modelMeta: ReturnType<typeof getModelMeta>;
  private modelNameOrPath?: string;

  constructor(options: PicoDetOptions = {}) {
    this.modelNameOrPath = options.modelPath;
    this.scoreThreshold = options.scoreThreshold ?? 0.3;
    this.nmsThreshold = options.nmsThreshold ?? 0.5;
    this.maxDetections = options.maxDetections ?? 100;
    this.modelMeta = getModelMeta(options.modelPath);
  }

  /**
   * Lazily initialize the ONNX session (load model on first use).
   */
  private async ensureSession() {
    if (this.session) return;

    const ortModule = await getOrt();

    // Resolve model path (downloads if needed)
    this.modelPath = await getModelPath(this.modelNameOrPath);

    // Create inference session with CPU provider
    this.session = await ortModule.InferenceSession.create(this.modelPath, {
      executionProviders: ["cpu"],
      graphOptimizationLevel: "all",
    });
  }

  /**
   * Detect layout regions in a page image.
   *
   * @param image - Page image as a Buffer (PNG, JPG, etc.)
   * @returns Array of detected layout regions with labels and bounding boxes.
   */
  async detect(image: Buffer): Promise<LayoutRegion[]> {
    await this.ensureSession();
    const ortModule = await getOrt();

    const [inputH, inputW] = this.modelMeta.inputSize;

    // Get original image dimensions
    const metadata = await sharp(image).metadata();
    const origW = metadata.width!;
    const origH = metadata.height!;

    // Preprocess: resize, normalize, HWC→CHW
    const inputTensor = await this.preprocess(image, inputH, inputW);

    // Scale factors for coordinate mapping
    const scaleY = inputH / origH;
    const scaleX = inputW / origW;

    // Run inference
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const feeds: Record<string, any> = {};
    const inputNames = this.session!.inputNames;

    if (inputNames.includes("image")) {
      feeds["image"] = new ortModule.Tensor("float32", inputTensor, [1, 3, inputH, inputW]);
    } else if (inputNames.includes("x")) {
      feeds["x"] = new ortModule.Tensor("float32", inputTensor, [1, 3, inputH, inputW]);
    } else {
      feeds[inputNames[0]] = new ortModule.Tensor("float32", inputTensor, [1, 3, inputH, inputW]);
    }

    // For post-processed models, scale_factor tells the NMS layer how to
    // map boxes back to original image coordinates.
    // scale_factor = [h_scale, w_scale] where scale = input_size / orig_size
    if (inputNames.includes("scale_factor")) {
      const scaleFactor = new Float32Array([scaleY, scaleX]);
      feeds["scale_factor"] = new ortModule.Tensor("float32", scaleFactor, [1, 2]);
    }

    const results = await this.session!.run(feeds);
    const outputNames = this.session!.outputNames;

    let detections;

    if (this.modelMeta.hasPostProcess || outputNames.length <= 2) {
      // Post-processed output: boxes [N, 6] = [class_id, score, x1, y1, x2, y2]
      // Coordinates are already in original image space (scale_factor applied by NMS)
      const boxesData = results[outputNames[0]].data as Float32Array;
      let numBoxes: number;

      if (outputNames.length === 2 && results[outputNames[1]]) {
        numBoxes = (results[outputNames[1]].data as Int32Array)[0] || 0;
      } else {
        const dims = results[outputNames[0]].dims;
        numBoxes = dims[0] as number;
      }

      detections = decodePostProcessedOutputs(boxesData, numBoxes, this.scoreThreshold);

      // Boxes are already in original image coords — map labels directly
      const regions: LayoutRegion[] = detections.map((det) => ({
        label: (this.modelMeta.labels[det.classId] || "text") as LayoutLabel,
        bbox: [
          Math.max(0, det.x1),
          Math.max(0, det.y1),
          Math.min(origW, det.x2),
          Math.min(origH, det.y2),
        ] as [number, number, number, number],
        confidence: det.score,
      }));

      return regions;
    } else {
      // Raw output format: per-level scores and bboxes (DFL encoded)
      const strides = [8, 16, 32, 64];
      const numLevels = strides.length;
      const scoreArrays: Float32Array[] = [];
      const bboxArrays: Float32Array[] = [];

      for (let i = 0; i < numLevels; i++) {
        scoreArrays.push(results[outputNames[i]].data as Float32Array);
        bboxArrays.push(results[outputNames[i + numLevels]].data as Float32Array);
      }

      detections = decodePicoDetOutputs(
        scoreArrays,
        bboxArrays,
        strides,
        inputH,
        inputW,
        this.modelMeta.numClasses,
        this.scoreThreshold
      );
    }

    // Apply NMS (only for raw outputs — post-processed models have NMS built in)
    detections = nms(detections, this.nmsThreshold, this.maxDetections);

    // Scale bounding boxes from input coords to original image coords
    const regions: LayoutRegion[] = detections.map((det) => ({
      label: (this.modelMeta.labels[det.classId] || "text") as LayoutLabel,
      bbox: [
        Math.max(0, det.x1 / scaleX),
        Math.max(0, det.y1 / scaleY),
        Math.min(origW, det.x2 / scaleX),
        Math.min(origH, det.y2 / scaleY),
      ] as [number, number, number, number],
      confidence: det.score,
    }));

    return regions;
  }

  /**
   * Preprocess an image for PicoDet inference.
   * Resize → normalize (ImageNet) → HWC to CHW → Float32Array
   */
  private async preprocess(
    image: Buffer,
    targetH: number,
    targetW: number
  ): Promise<Float32Array> {
    // Resize and get raw RGB pixels
    const { data, info } = await sharp(image)
      .resize(targetW, targetH, { fit: "fill" })
      .removeAlpha()
      .raw()
      .toBuffer({ resolveWithObject: true });

    const pixels = new Uint8Array(data.buffer, data.byteOffset, data.byteLength);
    const numPixels = info.width * info.height;

    // Create CHW float32 tensor with ImageNet normalization
    const tensor = new Float32Array(3 * numPixels);

    for (let i = 0; i < numPixels; i++) {
      const r = pixels[i * 3 + 0] / 255.0;
      const g = pixels[i * 3 + 1] / 255.0;
      const b = pixels[i * 3 + 2] / 255.0;

      tensor[0 * numPixels + i] = (r - IMAGENET_MEAN[0]) / IMAGENET_STD[0];
      tensor[1 * numPixels + i] = (g - IMAGENET_MEAN[1]) / IMAGENET_STD[1];
      tensor[2 * numPixels + i] = (b - IMAGENET_MEAN[2]) / IMAGENET_STD[2];
    }

    return tensor;
  }

  async close(): Promise<void> {
    if (this.session) {
      await this.session.release();
      this.session = null;
    }
  }
}
