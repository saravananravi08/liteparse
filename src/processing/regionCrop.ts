/**
 * Utility for cropping layout regions from rendered page images.
 */

import sharp from "sharp";
import { LayoutRegion } from "../engines/layout/interface.js";

/**
 * Crop a region from a page image.
 *
 * @param pageImage - Full page image as a Buffer.
 * @param region - Layout region with bounding box in pixel coordinates.
 * @param padding - Extra padding around the region in pixels.
 * @returns Cropped image as a PNG Buffer.
 */
export async function cropRegion(
  pageImage: Buffer,
  region: LayoutRegion,
  padding: number = 5
): Promise<Buffer> {
  const metadata = await sharp(pageImage).metadata();
  const imgW = metadata.width!;
  const imgH = metadata.height!;

  const [x1, y1, x2, y2] = region.bbox;

  // Apply padding and clamp to image bounds
  const left = Math.max(0, Math.floor(x1 - padding));
  const top = Math.max(0, Math.floor(y1 - padding));
  const right = Math.min(imgW, Math.ceil(x2 + padding));
  const bottom = Math.min(imgH, Math.ceil(y2 + padding));

  const width = right - left;
  const height = bottom - top;

  if (width <= 0 || height <= 0) {
    throw new Error(`Invalid crop region: ${JSON.stringify(region.bbox)}`);
  }

  return sharp(pageImage)
    .extract({ left, top, width, height })
    .png()
    .toBuffer();
}
