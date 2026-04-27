/**
 * Layout detection engine interface and types.
 *
 * Layout engines detect structural regions in document page images
 * (tables, figures, formulas, text blocks, etc.) and return labeled
 * bounding boxes with confidence scores.
 */

/** Labels detected by PicoDet layout models. */
export type LayoutLabel =
  | "text"
  | "title"
  | "figure"
  | "table"
  | "formula"
  | "list"
  | "figure_caption"
  | "table_caption"
  | "header"
  | "footer"
  | "reference"
  | "abstract"
  | "algorithm"
  | "paragraph_title"
  | "doc_title"
  | "footnote"
  | "seal"
  | "image"
  | "chart"
  | "number"
  | "content"
  | "figure_title"
  | "table_title";

/** A detected layout region on a page. */
export interface LayoutRegion {
  /** The semantic label of this region. */
  label: LayoutLabel;
  /** Bounding box in pixel coordinates [x1, y1, x2, y2]. */
  bbox: [number, number, number, number];
  /** Detection confidence score (0.0 to 1.0). */
  confidence: number;
}

/** Layout detection engine interface. */
export interface LayoutEngine {
  /** Engine name for logging. */
  name: string;
  /** Detect layout regions in a page image. */
  detect(image: Buffer): Promise<LayoutRegion[]>;
  /** Clean up resources. */
  close?(): Promise<void>;
}

/**
 * Region types that are considered "complex" and should be routed
 * to a vision model rather than handled by grid projection.
 */
export const DEFAULT_VISION_REGION_TYPES: LayoutLabel[] = [
  "image",
  "figure",
  "chart",
  "table",
  "formula",
];
