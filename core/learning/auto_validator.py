import logging
from datetime import datetime
from typing import Optional

from core.learning.training_database import get_training_database
from core.learning.quality_metrics import get_quality_calculator
from core.synthesis.feedback_loop import get_feedback_loop

logger = logging.getLogger(__name__)


def validate_pending_predictions(history_snapshot, min_age_hours: float = 6, limit: int = 50) -> int:
    """
    Validate pending predictions using history snapshot data.

    Returns number of validated predictions.
    """
    if not history_snapshot or not history_snapshot.get_latest():
        logger.warning("Learning validation skipped: no history snapshot")
        return 0

    snap = history_snapshot.get_latest()
    db = get_training_database()
    qc = get_quality_calculator()
    fb = get_feedback_loop()

    pending = db.get_pending_predictions(min_age_hours=min_age_hours, limit=limit)
    if not pending:
        return 0

    validated_count = 0
    now_ts = datetime.now().timestamp()

    for pred in pending:
        coin_id = pred.get("coin_id")
        df = snap.by_coin.get(coin_id)
        if df is None or df.empty:
            continue

        pred_ts = float(pred.get("timestamp") or 0)
        if pred_ts <= 0:
            continue

        # Filter data after prediction time
        df_slice = df[df["timestamp"] >= datetime.fromtimestamp(pred_ts)]
        if df_slice.empty:
            continue

        max_price = float(df_slice["price"].max())
        min_price = float(df_slice["price"].min())
        final_price = float(df_slice["price"].iloc[-1])
        time_to_result_hours = (now_ts - pred_ts) / 3600.0

        metrics = qc.calculate_all_metrics(
            signal_type=pred.get("signal_type", "hold"),
            predicted_confidence=float(pred.get("confidence") or 0.5),
            entry_price=float(pred.get("entry_price") or final_price),
            target_price=float(pred.get("target_price") or final_price),
            stop_loss=float(pred.get("stop_loss") or final_price),
            max_price=max_price,
            min_price=min_price,
            final_price=final_price,
            time_to_result_days=time_to_result_hours / 24.0,
        )

        db.update_result(
            prediction_id=int(pred["id"]),
            max_price=max_price,
            min_price=min_price,
            final_price=final_price,
            time_to_result_hours=time_to_result_hours,
            quality_score=metrics.overall_quality,
            quality_metrics=metrics.to_dict(),
        )

        # Update adaptive weights if we have component scores
        component_scores = pred.get("component_scores") or {}
        if component_scores:
            try:
                fb.update_component_weights(component_scores, metrics.overall_quality)
            except Exception as e:
                logger.warning(f"Adaptive weights update failed: {e}")

        validated_count += 1

    if validated_count:
        logger.info(f"Learning validation: validated {validated_count} predictions")
    return validated_count
