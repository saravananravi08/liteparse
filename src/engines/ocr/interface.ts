export interface OcrEngine {
  name: string;
  recognize(image: string | Buffer, options: OcrOptions): Promise<OcrResult[]>;
  recognizeBatch(images: (string | Buffer)[], options: OcrOptions): Promise<OcrResult[][]>;
  /** Terminate and release all resources (e.g., WASM memory). Call between chunks. */
  terminate?(): Promise<void>;
}

export interface OcrOptions {
  language: string | string[];
  correctRotation?: boolean;
}

export interface OcrResult {
  text: string;
  bbox: [number, number, number, number];
  confidence: number;
}
