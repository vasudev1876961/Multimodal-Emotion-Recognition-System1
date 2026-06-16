def fuse_emotions(face_result, text_result):

    face_emotion, face_conf = face_result
    text_emotion, text_conf = text_result

    # Weighted fusion
    FACE_WEIGHT = 0.6
    TEXT_WEIGHT = 0.4

    # Same prediction
    if face_emotion == text_emotion:

        final_confidence = (
            FACE_WEIGHT * face_conf +
            TEXT_WEIGHT * text_conf
        )

        return {
            "final_emotion": face_emotion,
            "confidence": round(final_confidence, 3),
            "decision": "Both models agreed"
        }

    # Different predictions
    weighted_face = FACE_WEIGHT * face_conf
    weighted_text = TEXT_WEIGHT * text_conf

    if weighted_face >= weighted_text:

        return {
            "final_emotion": face_emotion,
            "confidence": round(weighted_face, 3),
            "decision": "Face model dominated"
        }

    else:

        return {
            "final_emotion": text_emotion,
            "confidence": round(weighted_text, 3),
            "decision": "Text model dominated"
        }