/**
 * Post-processing for PicoDet layout detection:
 * - GFL DFL (Distribution Focal Loss) bounding box decoding
 * - Non-Maximum Suppression (NMS)
 */

interface RawDetection {
  classId: number;
  score: number;
  x1: number;
  y1: number;
  x2: number;
  y2: number;
}

/** Number of DFL regression bins per side (PicoDet default). */
const REG_MAX = 7; // 8 bins (0..7), so reg_max=7
const NUM_BINS = REG_MAX + 1; // 8

/**
 * Decode PicoDet raw outputs (without built-in post-processing).
 *
 * Output format per FPN level:
 * - Scores: [1, num_anchors, num_classes] — raw logits
 * - Bboxes: [1, num_anchors, 4 * (reg_max+1)] — DFL distribution
 *
 * DFL decoding: For each of the 4 sides (left, top, right, bottom),
 * apply softmax over reg_max+1 bins, then compute expected distance
 * as the weighted sum of bin indices.
 */
export function decodePicoDetOutputs(
  scores: Float32Array[],
  bboxes: Float32Array[],
  strides: number[],
  inputHeight: number,
  inputWidth: number,
  numClasses: number,
  scoreThreshold: number
): RawDetection[] {
  const detections: RawDetection[] = [];

  for (let level = 0; level < strides.length; level++) {
    const stride = strides[level];
    const featH = Math.ceil(inputHeight / stride);
    const featW = Math.ceil(inputWidth / stride);
    const numAnchors = featH * featW;
    const bboxChannels = NUM_BINS * 4; // 32

    const scoreData = scores[level];
    const bboxData = bboxes[level];

    for (let anchor = 0; anchor < numAnchors; anchor++) {
      // Find best class from score logits
      let maxScore = -Infinity;
      let maxClassId = 0;

      for (let c = 0; c < numClasses; c++) {
        const score = scoreData[anchor * numClasses + c];
        if (score > maxScore) {
          maxScore = score;
          maxClassId = c;
        }
      }

      // Apply sigmoid to get probability
      const prob = sigmoid(maxScore);
      if (prob < scoreThreshold) continue;

      // Anchor center point on the input image
      const anchorY = Math.floor(anchor / featW);
      const anchorX = anchor % featW;
      const centerX = (anchorX + 0.5) * stride;
      const centerY = (anchorY + 0.5) * stride;

      // Decode DFL distributions for 4 sides: left, top, right, bottom
      const bboxOffset = anchor * bboxChannels;
      const distLeft = decodeDFL(bboxData, bboxOffset + 0 * NUM_BINS);
      const distTop = decodeDFL(bboxData, bboxOffset + 1 * NUM_BINS);
      const distRight = decodeDFL(bboxData, bboxOffset + 2 * NUM_BINS);
      const distBottom = decodeDFL(bboxData, bboxOffset + 3 * NUM_BINS);

      // Convert distances to box coordinates
      const x1 = centerX - distLeft * stride;
      const y1 = centerY - distTop * stride;
      const x2 = centerX + distRight * stride;
      const y2 = centerY + distBottom * stride;

      detections.push({
        classId: maxClassId,
        score: prob,
        x1: Math.max(0, x1),
        y1: Math.max(0, y1),
        x2: Math.min(inputWidth, x2),
        y2: Math.min(inputHeight, y2),
      });
    }
  }

  return detections;
}

/**
 * Decode a single DFL (Distribution Focal Loss) distribution.
 * Applies softmax over NUM_BINS values, then computes the expected
 * distance as the weighted sum of bin indices (0, 1, ..., reg_max).
 */
function decodeDFL(data: Float32Array, offset: number): number {
  // Softmax
  let maxVal = -Infinity;
  for (let i = 0; i < NUM_BINS; i++) {
    if (data[offset + i] > maxVal) maxVal = data[offset + i];
  }

  let sumExp = 0;
  const exps = new Float64Array(NUM_BINS);
  for (let i = 0; i < NUM_BINS; i++) {
    exps[i] = Math.exp(data[offset + i] - maxVal);
    sumExp += exps[i];
  }

  // Weighted sum of indices = expected distance
  let dist = 0;
  for (let i = 0; i < NUM_BINS; i++) {
    dist += (exps[i] / sumExp) * i;
  }

  return dist;
}

/**
 * Decode outputs from PicoDet models exported WITH post-processing.
 * These models output two tensors:
 * - boxes: [num_detections, 6] where each row is [class_id, score, x1, y1, x2, y2]
 * - num_boxes: [1] scalar with number of valid detections
 */
export function decodePostProcessedOutputs(
  boxes: Float32Array,
  numBoxes: number,
  scoreThreshold: number
): RawDetection[] {
  const detections: RawDetection[] = [];

  for (let i = 0; i < numBoxes; i++) {
    const offset = i * 6;
    const classId = Math.round(boxes[offset + 0]);
    const score = boxes[offset + 1];
    const x1 = boxes[offset + 2];
    const y1 = boxes[offset + 3];
    const x2 = boxes[offset + 4];
    const y2 = boxes[offset + 5];

    if (score >= scoreThreshold) {
      detections.push({ classId, score, x1, y1, x2, y2 });
    }
  }

  return detections;
}

/**
 * Non-Maximum Suppression to remove overlapping detections.
 */
export function nms(
  detections: RawDetection[],
  iouThreshold: number,
  maxDetections: number = 100
): RawDetection[] {
  if (detections.length === 0) return [];

  // Sort by score descending
  const sorted = [...detections].sort((a, b) => b.score - a.score);
  const kept: RawDetection[] = [];
  const suppressed = new Set<number>();

  for (let i = 0; i < sorted.length && kept.length < maxDetections; i++) {
    if (suppressed.has(i)) continue;

    kept.push(sorted[i]);

    for (let j = i + 1; j < sorted.length; j++) {
      if (suppressed.has(j)) continue;

      // Only suppress same-class detections
      if (sorted[i].classId !== sorted[j].classId) continue;

      const iou = computeIoU(sorted[i], sorted[j]);
      if (iou > iouThreshold) {
        suppressed.add(j);
      }
    }
  }

  return kept;
}

function computeIoU(a: RawDetection, b: RawDetection): number {
  const interX1 = Math.max(a.x1, b.x1);
  const interY1 = Math.max(a.y1, b.y1);
  const interX2 = Math.min(a.x2, b.x2);
  const interY2 = Math.min(a.y2, b.y2);

  const interW = Math.max(0, interX2 - interX1);
  const interH = Math.max(0, interY2 - interY1);
  const interArea = interW * interH;

  const areaA = (a.x2 - a.x1) * (a.y2 - a.y1);
  const areaB = (b.x2 - b.x1) * (b.y2 - b.y1);
  const unionArea = areaA + areaB - interArea;

  return unionArea > 0 ? interArea / unionArea : 0;
}

function sigmoid(x: number): number {
  return 1 / (1 + Math.exp(-x));
}
