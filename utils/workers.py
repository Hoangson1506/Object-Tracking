"""
Background worker utilities.

This module contains worker functions that run in background threads
for async processing of tasks like saving violations.
"""

import queue
from utils import MinioClient, get_logger, log_violation, log_upload


def violation_save_worker(save_queue: queue.Queue) -> None:
    """
    Background worker for saving violation data to storage.

    This worker runs in a separate thread and processes violation data
    from the queue, uploading proofs, retraining data, and video clips
    to MinIO storage.

    Args:
        save_queue: Queue containing violation data dictionaries.
                    Send None to stop the worker.

    Expected queue item format:
        {
            'vehicle_id': int,
            'identifier': str,  # license plate or vehicle id
            'violation_type': str,
            'frame': np.ndarray,
            'bbox': tuple,
            'bboxes': list,
            'frame_buffer': list,
            'fps': int,
            'proof_crop': np.ndarray
        }
    """
    logger = get_logger("violation_worker", file_logging=True)
    
    try:
        client = MinioClient()
    except Exception as e:
        logger.error(f"Failed to initialize MinIO client: {e}")
        return

    while True:
        data = save_queue.get()
        
        # None is the signal to stop the worker
        if data is None:
            logger.info("Received stop signal, shutting down worker")
            break

        try:
            vehicle_id = data['vehicle_id']
            identifier = data['identifier']
            violation_type = data['violation_type']
            frame = data['frame']
            bbox = data['bbox']
            bboxes = data['bboxes']
            frame_buffer = data['frame_buffer']
            fps = data['fps']
            proof_crop = data['proof_crop']

            # Log the violation
            log_violation(logger, vehicle_id, violation_type, identifier)

            # Save proof crop
            success = client.save_proof(proof_crop, identifier, violation_type)
            log_upload(logger, "proofs", f"{violation_type}_{identifier}", success)
            
            # Save retraining data
            success = client.save_retraining_data(frame, vehicle_id, bbox)
            log_upload(logger, "retraining", f"train_{vehicle_id}", success)
            
            # Save labeled proof
            success = client.save_labeled_proof(frame, identifier, violation_type, bbox)
            log_upload(logger, "proofs", f"{violation_type}_{identifier}_labeled", success)
            
            # Save video proof if buffer available
            if frame_buffer:
                success = client.save_video_proof(frame_buffer, identifier, violation_type, bboxes, fps)
                log_upload(logger, "proofs", f"{violation_type}_{identifier}.mp4", success)
            
            logger.info(f"Saved all proofs for violation ID: {identifier}")
            
        except Exception as e:
            logger.error(f"Error saving violation: {e}")
        finally:
            save_queue.task_done()
