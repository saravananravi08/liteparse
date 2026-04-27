/**
 * Vision model engine interface.
 *
 * Vision engines receive cropped regions of document pages (images, charts,
 * tables, formulas) and return text descriptions, markdown tables, or LaTeX.
 */

import { LayoutLabel } from "../layout/interface.js";

/** Vision model engine interface. */
export interface VisionEngine {
  /** Engine name for logging. */
  name: string;

  /**
   * Describe/extract content from a cropped image region.
   *
   * @param image - Cropped region image as a Buffer (PNG).
   * @param regionType - The layout label of this region (table, figure, etc.).
   * @param customPrompt - Optional custom prompt override.
   * @returns Extracted text/markdown/LaTeX content.
   */
  describe(image: Buffer, regionType: LayoutLabel, customPrompt?: string): Promise<string>;
}

/**
 * Default prompts for each region type.
 * These instruct the vision model on what output format to produce.
 */
export const DEFAULT_VISION_PROMPTS: Partial<Record<LayoutLabel, string>> = {
  table:
    "Extract this table into clean markdown table format. Preserve all data accurately. Only output the markdown table, no explanation.",
  figure:
    "Describe this figure in detail. Include what it shows, any labels, axes, legends, and key data points. Be concise but thorough.",
  image:
    "Describe this image in detail. What does it show? Include any visible text, labels, or important visual elements.",
  chart:
    "Describe this chart in detail. Include the chart type, axes labels, data trends, and key values. Be concise but thorough.",
  formula:
    "Convert this mathematical formula to LaTeX notation. Only output the LaTeX, wrapped in $$ delimiters.",
  algorithm:
    "Describe this algorithm. Include the pseudocode or steps if visible. Use markdown code block format.",
};
