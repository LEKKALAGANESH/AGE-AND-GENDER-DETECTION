export interface FaceResult {
  age: number;
  age_min?: number;
  age_max?: number;
  gender: string;
  gender_confidence?: number;
  confidence: number;
  region: [number, number, number, number];
  emotion?: string;               // detected expression
  emotion_confidence?: number;    // expression detection confidence
}

export interface AnalyzeResponse {
  results: FaceResult[];
  face_count: number;
  processing_time_ms: number;
}

/** Resolve optional age range fields with fallback defaults. */
export function getAgeRange(face: FaceResult): { min: number; max: number } {
  return {
    min: face.age_min ?? face.age - 5,
    max: face.age_max ?? face.age + 5,
  };
}

/** Resolve optional gender confidence with fallback to face detection confidence. */
export function getGenderConfidence(face: FaceResult): number {
  return face.gender_confidence ?? face.confidence;
}
